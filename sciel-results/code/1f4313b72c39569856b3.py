"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import numpy as np

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

def _verify_full(U, V, W, n, m, p, rank):
    """
    Strict verification of the tensor decomposition.
    Checks that sum_r U[i*m+k, r] * V[k*p+j, r] * W[j'*n+i', r] == delta_{i,i'} * delta_{j,j'}
    for all i, k, j, i', j'.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i, k, j, i', j'] = sum_r U[i,k,r] V[k,j,r] W[j',i',r]
    # We want T[i, k, j, i', j'] == 1 if i==i' and j==j', else 0.
    
    # Efficient check:
    # For each r, we have U_r (4x1), V_r (12x1), W_r (12x1).
    # This is not a simple matrix multiplication.
    # We can iterate over i, k, j and check the resulting vector in C.
    
    for i in range(n):
        for k in range(m):
            # For fixed i, k, the contribution to C is V[k*p:(k+1)*p, :] * W.T * U[i*m+k, :] ? No.
            # C[i, j'] = sum_k A[i, k] B[k, j']
            # Decomposition: C[i, j'] = sum_r U[i*m+k, r] V[k*p+j', r] W[j'*n+i, r]
            # Wait, W index is j'*n + i.
            # So for fixed i, k, we sum over r and j'.
            # Let's just do the full O(n*m*p*n*p*rank) check if rank is small.
            # 2*2*6*2*6*21 = 5040 ops. Very fast.
            for j in range(p):
                # Check diagonal: should be 1
                s_diag = 0
                for r in range(rank):
                    s_diag += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if not np.isclose(s_diag, 1.0):
                    return False
                
                # Check off-diagonal: should be 0
                # For any i' != i or j' != j
                for i_p in range(n):
                    for j_p in range(p):
                        if i_p == i and j_p == j:
                            continue
                        s_off = 0
                        for r in range(rank):
                            s_off += U[i*m+k][r] * V[k*p+j][r] * W[j_p*n+i_p][r]
                        if not np.isclose(s_off, 0.0):
                            return False
    return True

def _try_remove_product(U, V, W, n, m, p, deadline):
    rank = 21
    nU = n * m
    nV = m * p
    nW = p * n
    
    for removed in range(rank):
        if time.time() > deadline:
            break
        
        idx = [r for r in range(rank) if r != removed]
        M = np.zeros((n*m*p, len(idx)))
        b = np.zeros(n*m*p)
        
        eq = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    # We are trying to match the contribution of product 'removed'
                    # to the diagonal term C[i,j] from A[i,k]B[k,j]
                    # Wait, the linear system approach assumes we can express the *tensor* of the removed product
                    # as a linear combination of the others.
                    # The tensor of product r is T_r[i,k,j] = U[i,m+k,r] * V[k,p+j,r] * W[j,n+i,r].
                    # We want sum_{r!=removed} c_r T_r[i,k,j] = T_removed[i,k,j].
                    # This must hold for ALL i,k,j.
                    # So we build a system for each (i,k,j).
                    
                    # M[eq, col] = T_idx[col][i,k,j]
                    # b[eq] = T_removed[i,k,j]
                    
                    for col, r in enumerate(idx):
                        M[eq, col] = U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    b[eq] = U[i*m+k][removed] * V[k*p+j][removed] * W[j*n+i][removed]
                    eq += 1
        
        try:
            c, residuals, rank_svd, sv = np.linalg.lstsq(M, b, rcond=None)
        except:
            continue
        
        residual = M @ c - b
        if np.max(np.abs(residual)) < 1e-9:
            # Check if the solution is valid for ALL tensor entries, not just diagonal?
            # The system was built only for diagonal entries (i,k,j -> C[i,j]).
            # It does NOT enforce off-diagonal terms to be 0.
            # So we MUST verify the full tensor.
            new_rank = rank - 1
            new_U = [[0] * new_rank for _ in range(nU)]
            new_V = [[0] * new_rank for _ in range(nV)]
            new_W = [[0] * new_rank for _ in range(nW)]
            
            new_idx = 0
            for r in range(rank):
                if r == removed:
                    continue
                coeff = c[new_idx]
                for a in range(nU):
                    new_U[a][new_idx] = U[a][r]
                for b_idx in range(nV):
                    new_V[b_idx][new_idx] = V[b_idx][r]
                for c_idx in range(nW):
                    new_W[c_idx][new_idx] = W[c_idx][r] * coeff
                new_idx += 1
            
            if _verify_full(new_U, new_V, new_W, n, m, p, new_rank):
                return new_U, new_V, new_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _build_strassen_21()
    
    # Try to find rank-20
    result = _try_remove_product(U, V, W, n, m, p, deadline)
    if result is not None:
        new_U, new_V, new_W = result
        # _verify_full is already called in _try_remove_product, but double check
        if _verify_full(new_U, new_V, new_W, n, m, p, 20):
            return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fallback to rank-21
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
