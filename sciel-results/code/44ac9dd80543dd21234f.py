"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time

# EVOLVE-BLOCK-START

def _build_strassen_21():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    # Check tensor identity: sum_r U[a,r]*V[b,r]*W[c,r] == delta_{a, c mod n} * delta_{b, c // n}
    # Actually: C[i,j] = sum_k A[i,k] B[k,j]
    # Tensor: T[i,k,j] = 1 if i==i and k==k and j==j ... 
    # Standard: T_{i,k,j} = delta_{i,i'} * delta_{k,k'} where C[i',j'] = sum_{i,k} A[i,k]B[k,j']
    # The decomposition: T_{i,k,j} = sum_r U[i,k,r] V[k,j,r] W[i,j,r]
    # So for each (i,k,j), sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So W index is (k, i) -> row k*n + i
    # U index: (i, k) for A[i,k] -> row i*m + k? Or (n*m, rank)
    # V index: (k, j) for B[k,j] -> row k*p + j? Or (m*p, rank)
    # W index: (j, i) for C[i,j] -> row j*n + i? Or (p*n, rank)
    # The contract says: W[k*n+i, r] where k is the column of C (which is j in my notation) and i is row of C
    # So W row = j*n + i
    # U row = i*m + k (A is n x m, so n*m entries)
    # V row = k*p + j (B is m x p, so m*p entries)
    # Tensor identity: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    # Wait, that's not right. The matrix multiplication tensor has T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]B[k,j]
    # So T[i,k,j] = 1 for all valid i,k,j? No.
    # T[i,k,j] is the coefficient such that C[i,j] = sum_{i',k',j'} T[i',k',j'] A[i',k'] B[k',j']
    # C[i,j] = sum_k A[i,k] B[k,j], so T[i,k,j] = 1 for all i,k,j in range.
    # So the condition is: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for ALL i in [0,n), k in [0,m), j in [0,p)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if s != 1:
                    return False
    return True

def _try_remove_product(U, V, W, n, m, p, deadline):
    """Try to remove one product by expressing it as a linear combination of others."""
    rank = 21
    nU = n * m
    nV = m * p
    nW = p * n
    
    for removed in range(rank):
        if time.time() > deadline:
            break
        # Try to find coefficients c_r for r != removed such that:
        # For all (i,k,j): sum_{r != removed} c_r * U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r] = U[i*m+k][removed] * V[k*p+j][removed] * W[j*n+i][removed]
        # This is a system of n*m*p = 24 equations in 20 unknowns (c_r for r != removed)
        # But it's not linear in c_r! It's linear in c_r only if we fix the products.
        # Actually it IS linear in c_r: sum_r c_r * (U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]) = U[i*m+k][removed] * V[k*p+j][removed] * W[j*n+i][removed]
        # So we have a linear system: M * c = b, where M[j_eq, r] = U[i*m+k][r]*V[k*p+j][r]*W[j*n+i][r] for r != removed
        # and b[j_eq] = U[i*m+k][removed]*V[k*p+j][removed]*W[j*n+i][removed]
        
        # Build the system
        idx = [r for r in range(rank) if r != removed]
        # M is 24 x 20, b is 24 x 1
        # Use numpy least squares
        import numpy as np
        M = np.zeros((n*m*p, len(idx)))
        b = np.zeros(n*m*p)
        eq = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    for col, r in enumerate(idx):
                        M[eq, col] = U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    b[eq] = U[i*m+k][removed] * V[k*p+j][removed] * W[j*n+i][removed]
                    eq += 1
        
        # Solve least squares
        try:
            c, residuals, rank_svd, sv = np.linalg.lstsq(M, b, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        residual = M @ c - b
        if np.max(np.abs(residual)) < 1e-9:
            # We found a valid rank-20 decomposition!
            new_rank = rank - 1
            new_U = [[0] * new_rank for _ in range(nU)]
            new_V = [[0] * new_rank for _ in range(nV)]
            new_W = [[0] * new_rank for _ in range(nW)]
            new_idx = 0
            for r in range(rank):
                if r == removed:
                    continue
                for a in range(nU):
                    new_U[a][new_idx] = U[a][r] + c[new_idx] * 0  # c is already the coefficient
                    # Wait, I need to add c_r * U[a][r] to the kept products
                new_idx += 1
            # Actually, the new decomposition has rank new_rank with the same U,V,W but scaled
            # For kept product r: new_U[a][new_idx] = U[a][r], new_V[b][new_idx] = V[b][r], new_W[c][new_idx] = W[c][r] * c_r?
            # No. The product is sum_r c_r * U[a][r]*V[b][r]*W[c][r]. We can absorb c_r into any one of U, V, or W.
            # Let's absorb into W: new_W[c][new_idx] = W[c][r] * c_r
            new_idx = 0
            for r in range(rank):
                if r == removed:
                    continue
                for a in range(nU):
                    new_U[a][new_idx] = U[a][r]
                for b_idx in range(nV):
                    new_V[b_idx][new_idx] = V[b_idx][r]
                for c_idx in range(nW):
                    new_W[c_idx][new_idx] = W[c_idx][r] * c[new_idx]
                new_idx += 1
            return new_U, new_V, new_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    U, V, W = _build_strassen_21()
    
    # Try to find rank-20
    result = _try_remove_product(U, V, W, n, m, p, deadline)
    if result is not None:
        new_U, new_V, new_W = result
        # Verify
        if _verify(new_U, new_V, new_W, n, m, p, 20):
            return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fallback to rank-21
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
