import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
TARGET_DEADLINE = 145.0  # 5s safety margin

def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """
    Verifies if U, V, W form a valid tensor decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R = Ua.shape[1]
    
    # Reshape to 3D
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    
    # T[i, k, j] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: einsum indices
    # U3[i,j,r], V3[j,k,r], W3[k,i,r]
    # We want sum over r.
    # Result shape (n, p, m) corresponding to (i, k, j)
    T = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3, optimize=True)
    
    # Target is 1 for all entries
    err = float(np.max(np.abs(T - 1.0)))
    
    return (err < tol), err

def _current_rank55(n, m, p):
    """
    Reconstructs the known rank-55 solution from the seed.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Block 1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1; W[k1 + 2][col] += 1
        col += 1
        # Block 2
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0 + 2][col] += 1; W[k1 + 2][col] -= 1
        col += 1
        # Block 3
        U[0][col] += 1
        V[k1][col] += 1; V[10 + k1][col] -= 1
        W[k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
        # Block 4
        U[4][col] += 1
        V[10 + k0][col] += 1; V[k0][col] -= 1
        W[k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        # Block 5
        U[0][col] += 1; U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] -= 1; W[k1][col] += 1
        col += 1
        # Block 6
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1 + 2][col] += 1
        col += 1
        # Block 7
        U[1][col] += 1; U[4][col] -= 1
        V[10 + k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1
        col += 1
        # Block 8
        U[2][col] += 1; V[20 + k0][col] += 1; W[k0][col] += 1
        col += 1
        # Block 9
        U[2][col] += 1; V[20 + k1][col] += 1; W[k1][col] += 1
        col += 1
        # Block 10
        U[5][col] += 1; V[20 + k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        # Block 11
        U[5][col] += 1; V[20 + k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
    return {"U": U, "V": V, "W": W}

def _solve_W(U, V, n, m, p, R):
    """
    Solves for W given U and V.
    For each (i, k), we have a system M @ w = 1_m
    where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    W = np.zeros((p * n, R))
    
    for i in range(n):
        for k in range(p):
            # Build M (m x R)
            M = np.zeros((m, R))
            for j in range(m):
                # Vectorized multiplication of rows
                M[j] = Ua[i * m + j] * Va[j * p + k]
            
            try:
                # Solve M @ w = ones(m)
                # Use lstsq for robustness
                w, residuals, rank, s = np.linalg.lstsq(M, np.ones(m), rcond=1e-12)
            except Exception:
                return None
            
            # Check if solution is valid (residual should be 0)
            # We check the actual residual by recomputing
            resid = M @ w - np.ones(m)
            if np.linalg.norm(resid) > 1e-9:
                return None
                
            W[k * n + i] = w
            
    return W.tolist()

def _polish_rational(U, V, W, n, m, p):
    """
    Snaps entries to nearest 0.5 and verifies.
    """
    def snap(M):
        a = np.array(M, dtype=float)
        b = np.round(a * 2.0) / 2.0
        return b.tolist()

    Us, Vs, Ws = snap(U), snap(V), snap(W)
    ok, err = _verify_full(Us, Vs, Ws, n, m, p, tol=1e-6)
    if ok:
        return Us, Vs, Ws
    return None

def _eliminate_columns(U, V, W, n, m, p, rng, deadline=TARGET_DEADLINE):
    """
    Tries to remove redundant columns.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    R = len(U[0])
    improved = True
    
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        order = list(range(R))
        rng.shuffle(order)
        
        for r in order:
            if time.time() > deadline - 2:
                break
            # Try removing column r
            U2 = [row[:r] + row[r + 1:] for row in U]
            V2 = [row[:r] + row[r + 1:] for row in V]
            W2 = _solve_W(U2, V2, n, m, p, R - 1)
            
            if W2 is None:
                continue
                
            ok, _ = _verify_full(U2, V2, W2, n, m, p)
            if ok:
                U, V, W = U2, V2, W2
                R -= 1
                improved = True
                break
                
    return U, V, W

def _rand_sparse(nrows, rank, density, rng, vals):
    A = np.zeros((nrows, rank))
    for r in range(nrows):
        for c in range(rank):
            if rng.random() < density:
                A[r, c] = vals[rng.randrange(len(vals))]
    return A.tolist()

def _try_rank(R, n, m, p, rng, deadline=TARGET_DEADLINE):
    """
    Attempts to find a rank R solution.
    """
    nm, mp = n * m, m * p
    
    # Define value sets to try
    val_sets = [
        [-1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, -0.5, 0.5, 1.0],
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    ]
    
    densities = [0.1, 0.2, 0.3]
    
    max_iters = 2000
    
    for vset in val_sets:
        for dens in densities:
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                U = _rand_sparse(nm, R, dens, rng, vset)
                V = _rand_sparse(mp, R, dens, rng, vset)
                
                W = _solve_W(U, V, n, m, p, R)
                if W is None:
                    continue
                
                ok, _ = _verify_full(U, V, W, n, m, p)
                if ok:
                    # Try to polish
                    polished = _polish_rational(U, V, W, n, m, p)
                    if polished is not None:
                        return polished
                    # If polish fails, return as is (floats are allowed if they are effectively rational? 
                    # The prompt says "integer/half-integer entries". So we MUST return snapped values.
                    # If snap fails, this solution is invalid for the output contract.
                    # But maybe a slight adjustment works? No, strict contract.
                    # So if _polish_rational fails, we discard.
                    pass
                    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(42) # Fixed seed for reproducibility
    
    # Start with known solution
    best = _current_rank55(n, m, p)
    best_rank = 55
    
    # 1. Eliminate columns from seed
    Ue, Ve, We = _eliminate_columns(best["U"], best["V"], best["W"], n, m, p, rng)
    ok, _ = _verify_full(Ue, Ve, We, n, m, p)
    if ok and len(Ue[0]) < best_rank:
        best = {"U": Ue, "V": Ve, "W": We}
        best_rank = len(Ue[0])
        
    # 2. Polish the best found so far
    polished = _polish_rational(best["U"], best["V"], best["W"], n, m, p)
    if polished is not None:
        best = {"U": polished[0], "V": polished[1], "W": polished[2]}
        
    # 3. Search for lower ranks
    # We go down to a lower bound. 
    # Lower bound for M(n,m,p) is n+m+p-2. Here 2+3+10-2=13.
    # Let's try down to 40 to be safe and fast.
    for target_rank in range(best_rank - 1, 39, -1):
        if time.time() > TARGET_DEADLINE - 10:
            break
            
        result = _try_rank(target_rank, n, m, p, rng)
        if result is not None:
            U, V, W = result
            # Try to eliminate further
            U2, V2, W2 = _eliminate_columns(U, V, W, n, m, p, rng)
            ok2, _ = _verify_full(U2, V2, W2, n, m, p)
            if ok2:
                best = {"U": U2, "V": V2, "W": W2}
                best_rank = len(U2[0])
            else:
                # If elimination broke it, keep the uneliminated one if it was valid
                # But _try_rank already verified and polished.
                # So we should keep the polished one from _try_rank.
                best = {"U": U, "V": V, "W": W}
                best_rank = len(U[0])
                
            # Break after finding a lower rank?
            # We might find an even lower rank if we continue.
            # But we break to save time for the next lower rank attempt?
            # No, if we found rank 54, we should try 53 next.
            # The loop handles that.
            
    # Final check
    if time.time() > TARGET_DEADLINE - 1:
        return best
        
    return best

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    ok, err = _verify_full(res["U"], res["V"], res["W"], n, m, p)
    print(f"Rank: {len(res['U'][0])}, Valid: {ok}, MaxErr: {err:.2e}")
