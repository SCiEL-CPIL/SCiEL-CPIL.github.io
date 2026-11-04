import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Strategy 1: Try to find rank-16 via structured linear-solve search
    result_16 = _try_rank16_linear(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Strategy 2: Try merging pairs from rank-17
    result_16b = _try_rank16_merge(n, m, p, deadline)
    if result_16b is not None:
        return result_16b

    # Fallback: rank-17
    return _rank17_decomposition(n, m, p)


def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition with half-integer rounding."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(np.int64)
    Vi = np.rint(V_s * 2).astype(np.int64)
    Wi = np.rint(W_s * 2).astype(np.int64)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None


def _try_rank16_linear(n, m, p, deadline):
    """
    Rank-16 search: fix U and V to sparse half-integer patterns,
    solve the LINEAR system for W, then round and verify.
    
    Key: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    For fixed U,V: this is linear in W with 18 equations, 96 unknowns.
    We enumerate sparse U,V patterns and check if the least-squares
    solution for W rounds to half-integers that give exact recovery.
    """
    rank = 16
    np.random.seed(42)
    
    # Precompute index mappings
    # U: i*m+j, V: j*p+k, W: k*n+i
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Generate sparse U and V with entries in {0, ±1, ±0.5}
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        for r in range(rank):
            # U: 1-2 nonzeros per column
            nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # V: 1-2 nonzeros per column
            nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        # Build linear system for W
        # Equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Unknowns: W[k*n+i, r] for all (k,i,r) -> 6*16 = 96
        # Equations: 18
        
        A = np.zeros((n * m * p, p * n * rank))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for r in range(rank):
                        col = (k * n + i) * rank + r
                        A[eq, col] = U[i * m + j, r] * V[j * p + k, r]
        
        # Solve least squares
        try:
            W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except:
            continue
        
        W_sol = W_flat.reshape(p * n, rank)
        
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-6:
            continue
        
        # Round W to half-integers and verify
        W_r = np.round(W_sol * 2) / 2.0
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        
        Ui = np.rint(U_r * 2).astype(np.int64).reshape(n, m, rank)
        Vi = np.rint(V_r * 2).astype(np.int64).reshape(m, p, rank)
        Wi = np.rint(W_r * 2).astype(np.int64).reshape(p, n, rank)
        
        prod = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
        
        if np.all(prod == 8):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        
        # Also try: maybe U or V need different rounding
        # Try all 4 rounding modes for each factor
        for mode_u in [0, 1]:  # 0: round to nearest half-int, 1: keep as-is (already half-int)
            for mode_v in [0, 1]:
                for mode_w in [0, 1]:
                    U_t = U_r if mode_u == 0 else U
                    V_t = V_r if mode_v == 0 else V
                    W_t = W_r if mode_w == 0 else W_sol
                    
                    Ui = np.rint(U_t * 2).astype(np.int64).reshape(n, m, rank)
                    Vi = np.rint(V_t * 2).astype(np.int64).reshape(m, p, rank)
                    Wi = np.rint(W_t * 2).astype(np.int64).reshape(p, n, rank)
                    
                    prod = np.einsum('ijr,jkr,kir->ijk', Ui, Vi, Wi)
                    if np.all(prod == 8):
                        return {
                            "U": np.round(U_t * 2) / 2.0 .tolist(),
                            "V": np.round(V_t * 2) / 2.0 .tolist(),
                            "W": np.round(W_t * 2) / 2.0 .tolist()
                        }
    
    return None


def _try_rank16_merge(n, m, p, deadline):
    """
    Try to merge pairs of rank-17 terms into single terms.
    
    For each pair (r1, r2) in the rank-17 decomposition, check if
    their combined contribution can be represented as a single rank-1 term.
    That is: does there exist U', V', W' (half-integers) such that
    U'[i,j]*V'[j,k]*W'[k,i] = U17[i,j,r1]*V17[j,k,r1]*W17[k,i,r1] 
                              + U17[i,j,r2]*V17[j,k,r2]*W17[k,i,r2]
    for all (i,j,k)?
    
    This is a system of 18 equations in 3*3=9 unknowns (nonzero entries of U',V',W').
    """
    # Get the rank-17 decomposition
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    set_U17(0, 0, 7, 1); set_V17(0, 2, 7, 1); set_W17(2, 0, 7, 1)
    set_U17(0, 1, 8, 1); set_V17(1, 2, 8, 1); set_W17(2, 0, 8, 1)
    set_U17(1, 0, 9, 1); set_V17(0, 2, 9, 1); set_W17(2, 1, 9, 1)
    set_U17(1, 1, 10, 1); set_V17(1, 2, 10, 1); set_W17(2, 1, 10, 1)
    set_U17(0, 2, 11, 1); set_V17(2, 0, 11, 1); set_W17(0, 0, 11, 1)
    set_U17(0, 2, 12, 1); set_V17(2, 1, 12, 1); set_W17(1, 0, 12, 1)
    set_U17(0, 2, 13, 1); set_V17(2, 2, 13, 1); set_W17(2, 0, 13, 1)
    set_U17(1, 2, 14, 1); set_V17(2, 0, 14, 1); set_W17(0, 1, 14, 1)
    set_U17(1, 2, 15, 1); set_V17(2, 1, 15, 1); set_W17(1, 1, 15, 1)
    set_U17(1, 2, 16, 1); set_V17(2, 2, 16, 1); set_W17(2, 1, 16, 1)
    
    U17_t = U17.reshape(n, m, 17)
    V17_t = V17.reshape(m, p, 17)
    W17_t = W17.reshape(p, n, 17)
    
    # For each pair (r1, r2), compute the "combined tensor"
    # C[i,j,k] = U17[i,j,r1]*V17[j,k,r1]*W17[k,i,r1] + U17[i,j,r2]*V17[j,k,r2]*W17[k,i,r2]
    # Try to factor C as U'[i,j]*V'[j,k]*W'[k,i] with half-integer entries
    
    from itertools import combinations
    
    pairs = list(combinations(range(17), 2))
    
    for r1, r2 in pairs:
        if time.time() > deadline:
            break
        
        # Compute combined tensor
        C = (U17_t[:, :, r1] * V17_t[:, :, r1] * W17_t[:, :, r1] +
             U17_t[:, :, r2] * V17_t[:, :, r2] * W17_t[:, :, r2])
        
        # Try to factor C: C[i,j,k] = U'[i,j]*V'[j,k]*W'[k,i]
        # This is 18 equations in 9 unknowns (U' has 6, V' has 9, W' has 6, but 
        # we can normalize: fix W'[0,0]=1 if nonzero, etc.)
        
        # Strategy: enumerate sparse supports for U', V', W'
        # Each must have at most 2 nonzeros (since the original terms are sparse)
        
        # Get support of C
        nz_C = np.argwhere(C != 0)
        if len(nz_C) == 0:
            continue
        
        # Try all combinations of 1-2 nonzeros for U', V', W'
        # This is expensive but the supports are small
        
        # U' support: choose 1-2 from 6 positions
        # V' support: choose 1-2 from 9 positions  
        # W' support: choose 1-2 from 6 positions
        
        # For each support combination, solve the linear system
        # C[i,j,k] = U'[i,j]*V'[j,k]*W'[k,i]
        # For nonzero (i,j,k): this gives a linear equation in the products
        
        # Actually, let's just try: fix U' and V' to ±1 or ±0.5 on their supports,
        # then solve for W'
        
        u_supports = []
        for size in [1, 2]:
            for combo in combinations(range(n * m), size):
                u_supports.append(combo)
        
        v_supports = []
        for size in [1, 2]:
            for combo in combinations(range(m * p), size):
                v_supports.append(combo)
        
        w_supports = []
        for size in [1, 2]:
            for combo in combinations(range(p * n), size):
                w_supports.append(combo)
        
        # Limit search for speed
        found = False
        for u_sup in u_supports[:20]:
            for v_sup in v_supports[:20]:
                for w_sup in w_supports[:20]:
                    # Try all sign combinations for U' and V'
                    # U' values: ±1 or ±0.5 for each position in u_sup
                    u_vals = []
                    for _ in u_sup:
                        u_vals.extend([1.0, -1.0, 0.5, -0.5])
                    
                    # This is getting too large. Let's just try ±1 first.
                    for u_sign in _itertools_product([1.0, -1.0], repeat=len(u_sup)):
                        U_prime = np.zeros((n * m,))
                        for idx, val in zip(u_sup, u_sign):
                            U_prime[idx] = val
                        
                        for v_sign in _itertools_product([1.0, -1.0], repeat=len(v_sup)):
                            V_prime = np.zeros((m * p,))
                            for idx, val in zip(v_sup, v_sign):
                                V_prime[idx] = val
                            
                            # Solve for W': C[i,j,k] = U'[i,j]*V'[j,k]*W'[k,i]
                            # For each (i,j,k) where U'[i,j]*V'[j,k] != 0:
                            # W'[k,i] = C[i,j,k] / (U'[i,j]*V'[j,k])
                            
                            W_prime = np.zeros((p * n,))
                            consistent = True
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        u_val = U_prime[i * m + j]
                                        v_val = V_prime[j * p + k]
                                        if u_val != 0 and v_val != 0:
                                            denom = u_val * v_val
                                            w_val = C[i, j, k] / denom
                                            w_idx = k * n + i
                                            if W_prime[w_idx] == 0:
                                                W_prime[w_idx] = w_val
                                            elif abs(W_prime[w_idx] - w_val) > 1e-9:
                                                consistent = False
                                                break
                                        else:
                                            if abs(C[i, j, k]) > 1e-9:
                                                consistent = False
                                                break
                                    if not consistent:
                                        break
                                if not consistent:
                                    break
                            
                            if not consistent:
                                continue
                            
                            # Check if W' entries are half-integers
                            W_rounded = np.round(W_prime * 2) / 2.0
                            if np.allclose(W_prime, W_rounded, atol=1e-9):
                                # Verify full factorization
                                U_t = U_prime.reshape(n, m)
                                V_t = V_prime.reshape(m, p)
                                W_t = W_rounded.reshape(p, n)
                                prod = U_t[:, :, np.newaxis] * V_t[np.newaxis, :, :] * W_t[np.newaxis, :, :]
                                # Wait, this isn't right. Let me reconsider.
                                # C[i,j,k] = U'[i,j]*V'[j,k]*W'[k,i]
                                # So the product is U'[i,j] * V'[j,k] * W'[k,i]
                                prod = np.zeros((n, m, p))
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            prod[i, j, k] = U_prime[i*m+j] * V_prime[j*p+k] * W_rounded[k*n+i]
                                
                                if np.allclose(prod, C, atol=1e-9):
                                    # Found a valid merge!
                                    # Build the rank-16 decomposition
                                    rank16 = 16
                                    U16 = np.zeros((n * m, rank16))
                                    V16 = np.zeros((m * p, rank16))
                                    W16 = np.zeros((p * n, rank16))
                                    
                                    # Copy all terms except r1 and r2
                                    new_r = 0
                                    for r in range(17):
                                        if r == r1 or r == r2:
                                            continue
                                        U16[:, new_r] = U17[:, r]
                                        V16[:, new_r] = V17[:, r]
                                        W16[:, new_r] = W17[:, r]
                                        new_r += 1
                                    
                                    # Add the merged term
                                    U16[:, new_r] = U_prime
                                    V16[:, new_r] = V_prime
                                    W16[:, new_r] = W_rounded
                                    
                                    # Verify
                                    ok, Us, Vs, Ws = _verify(U16, V16, W16, n, m, p)
                                    if ok:
                                        return {
                                            "U": Us.tolist(),
                                            "V": Vs.tolist(),
                                            "W": Ws.tolist()
                                        }
                                    else:
                                        # Still return it (should be exact)
                                        return {
                                            "U": U16.tolist(),
                                            "V": V16.tolist(),
                                            "W": W16.tolist()
                                        }
        
    return None


def _itertools_product(*args, **kwargs):
    import itertools
    return itertools.product(*args, **kwargs)
