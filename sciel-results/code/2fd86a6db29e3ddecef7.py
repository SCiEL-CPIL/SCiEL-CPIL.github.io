"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B.
    C[i,k] = sum_{j=0}^{m-1} A[i,j]*B[j,k].
    Tensor entry T[i*m+j, j*p+k, k*n+i] must equal 1 for all i,j,k, 0 otherwise.
    We check by reconstructing the tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    and verifying all required entries equal 1 and no spurious non-zero elsewhere.
    """
    # Build the full tensor as a 3D array
    # T shape: (n*m, m*p, p*n)
    T = [[0]* (p*n) for _ in range(m*p)]
    # Actually we need 3D; use nested lists: T[a][b][c]
    T = [[[0]*(p*n) for _ in range(m*p)] for _ in range(n*m)]
    for r in range(rank):
        # collect non-zero entries in each factor for this r
        u_nz = [(i, U[i][r]) for i in range(n*m) if U[i][r] != 0]
        v_nz = [(j, V[j][r]) for j in range(m*p) if V[j][r] != 0]
        w_nz = [(k, W[k][r]) for k in range(p*n) if W[k][r] != 0]
        for (a, ua) in u_nz:
            for (b, vb) in v_nz:
                for (c, wc) in w_nz:
                    T[a][b][c] += ua * vb * wc
    # Check required entries: for each i in [0,n), j in [0,m), k in [0,p):
    #   a = i*m + j, b = j*p + k, c = k*n + i  -> T[a][b][c] must be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T[a][b][c] != 1:
                    return False
    # Check no spurious non-zeros: for any (a,b,c) not of the form above, T must be 0
    # Build set of required (a,b,c)
    required = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                required.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            row = T[a][b]
            for c in range(p*n):
                if row[c] != 0 and (a,b,c) not in required:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10.
    Groups p=10 columns into 5 pairs; per pair uses 7 shared + 4 direct = 11 terms.
    """
    rank = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0
    for t in range(5):
        k0 = 2*t
        k1 = 2*t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*10+k0][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M2: (A10 + A11) * B0,k0
        U[1*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1
        W[k1*2+1][col] -= 1
        col += 1

        # M3: A00 * (B0,k1 - B1,k1)
        U[0*3+0][col] += 1
        V[0*10+k1][col] += 1
        V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M4: A11 * (B1,k0 - B0,k0)
        U[1*3+1][col] += 1
        V[1*10+k0][col] += 1
        V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1
        W[k0*2+1][col] += 1
        col += 1

        # M5: (A00 + A01) * B1,k1
        U[0*3+0][col] += 1
        U[0*3+1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1
        W[k1*2+0][col] += 1
        col += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1*3+0][col] += 1
        U[0*3+0][col] -= 1
        V[0*10+k0][col] += 1
        V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0*3+1][col] += 1
        U[1*3+1][col] -= 1
        V[1*10+k0][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1

        # Direct terms for row 2 (A02, A12) with B2
        U[0*3+2][col] += 1
        V[2*10+k0][col] += 1
        W[k0*2+0][col] += 1
        col += 1

        U[0*3+2][col] += 1
        V[2*10+k1][col] += 1
        W[k1*2+0][col] += 1
        col += 1

        U[1*3+2][col] += 1
        V[2*10+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1

        U[1*3+2][col] += 1
        V[2*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

import numpy as np

def _verify_np(U, V, W, n, m, p, rank):
    """Fast numpy-based verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficient: for each r, outer product
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    # Check required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if abs(T[a,b,c] - 1.0) > 1e-9:
                    return False
    # Check spurious entries
    required_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                required_mask[a,b,c] = True
    if np.any(np.abs(T[~required_mask]) > 1e-9):
        return False
    return True

def _try_rank54_random(n, m, p, deadline):
    """Random sparse linear algebra approach for rank-54."""
    rank = 54
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Initialize with random sparse 0/1 entries
    rng = np.random.default_rng(42)
    for r in range(rank):
        # Each factor has a few non-zero entries
        for _ in range(rng.integers(2, 5)):
            U[rng.integers(n*m)][r] = rng.choice([1, -1])
        for _ in range(rng.integers(2, 5)):
            V[rng.integers(m*p)][r] = rng.choice([1, -1])
        for _ in range(rng.integers(2, 5)):
            W[rng.integers(p*n)][r] = rng.choice([1, -1])
    
    best_residual = None
    best_U, best_V, best_W = None, None, None
    
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        # Compute residual
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        # Target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    target[a,b,c] = 1.0
        
        residual = T - target
        frob_norm = np.linalg.norm(residual)
        
        if best_residual is None or frob_norm < best_residual:
            best_residual = frob_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if frob_norm < 1e-9:
                # Check exactness
                if _verify_np(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, rank):
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        # Gradient-based update: try flipping or zeroing entries
        # For each factor, compute gradient of ||T - target||^2 w.r.t. each entry
        # d(T)/dU[a,r] = V[:,r] * W[:,r] (outer product)
        # Gradient of 0.5*||T-target||^2 w.r.t. U[a,r] = sum_{b,c} (T[a,b,c]-target[a,b,c]) * V[b,r] * W[c,r]
        
        # Pick a random entry to modify
        factor = rng.integers(3)  # 0=U, 1=V, 2=W
        if factor == 0:
            idx = rng.integers(n*m)
            r = rng.integers(rank)
            # Gradient
            grad = np.sum(residual[idx,:,:] * V[:,r] * W[:,r])
            # Try flipping sign or zeroing
            if U[idx, r] != 0:
                if rng.random() < 0.5:
                    U[idx, r] = -U[idx, r]  # flip sign
                else:
                    U[idx, r] = 0  # zero out
            else:
                U[idx, r] = rng.choice([1, -1])  # set to +/-1
        elif factor == 1:
            idx = rng.integers(m*p)
            r = rng.integers(rank)
            grad = np.sum(residual[:,idx,:] * U[:,r] * W[:,r])
            if V[idx, r] != 0:
                if rng.random() < 0.5:
                    V[idx, r] = -V[idx, r]
                else:
                    V[idx, r] = 0
            else:
                V[idx, r] = rng.choice([1, -1])
        else:
            idx = rng.integers(p*n)
            r = rng.integers(rank)
            grad = np.sum(residual[:,:,idx] * U[:,r] * V[:,r])
            if W[idx, r] != 0:
                if rng.random() < 0.5:
                    W[idx, r] = -W[idx, r]
                else:
                    W[idx, r] = 0
            else:
                W[idx, r] = rng.choice([1, -1])
    
    # Final check on best solution
    if best_U is not None:
        if _verify_np(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, rank):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Try rank-54 random search first (most valuable if it works)
    if time.time() < deadline - 20:
        try:
            res54 = _try_rank54_random(n, m, p, deadline)
            if res54 is not None:
                return res54
        except Exception:
            pass

    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
            # Also try numpy verification
            if _verify_np(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
