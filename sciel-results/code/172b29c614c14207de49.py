import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    def build_rank50():
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        
        def add_strassen_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            # M1 = (a+d)(e+h)
            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1; W[c11_idx, col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1; W[c11_idx, col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01_idx, col] += 1; W[c11_idx, col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00_idx, col] += 1; W[c10_idx, col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1; W[c01_idx, col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1
        
        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)
            
            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)
        
        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Vectorized verification using the trace convention."""
        r = U.shape[1]
        # Build contraction matrices
        # For each (i, k), check sum over j, r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta
        # We check: for each (i, j, jp, k), sum_r U[i*m+j, r]*V[jp*p+k, r]*W[k*n+i, r] == delta_{j,jp}
        
        # More efficient: for each r, form outer products and accumulate
        # C[i,k] contribution from term r: (U[:,r] reshaped as 2x4) * (V[:,r] reshaped as 4x7)
        # Actually the decomposition is: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not standard. Let me re-read the verify function.
        
        # The verify checks: for each (i, j, jp, k):
        #   sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c] == (1.0 if j==jp else 0.0)
        
        # This means: the tensor T[i,j,jp,k] = sum_c U[i*m+j,c] V[jp*p+k,c] W[k*n+i,c]
        # should equal delta_{j,jp}
        
        # Let's do it with matrix operations for speed
        # For each c, define A_c = U[:,c] (8-vector), B_c = V[:,c] (28-vector), W_c = W[:,c] (14-vector)
        # T[i,j,jp,k] = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[k*n+i,c]
        
        # Reshape: U is 8xr, V is 28xr, W is 14xr
        # U[i*m+j, c] -> for fixed i, this is a 4xr matrix (j, c)
        # V[jp*p+k, c] -> for fixed k, this is a 4xr matrix (jp, c)  
        # W[k*n+i, c] -> for fixed (i,k), this is an xr vector (c)
        
        # T[i,j,jp,k] = sum_c U[i, j, c] * V[k, jp, c] * W[i, k, c]
        # where U[i,j,c] = U[i*m+j, c], V[k,jp,c] = V[jp*p+k, c], W[i,k,c] = W[k*n+i, c]
        
        # For each (i,k): T[i, :, :, k] = sum_c U[i,:,c] * V[k,:,c] * W[i,k,c]
        # This is an outer product weighted by W[i,k,c]
        
        # Let's just do it directly but vectorized over c
        # Shape: (n, m, m, p) tensor
        # T = zeros((n, m, m, p))
        # for c in range(r):
        #     Uc = U[:, c].reshape(n, m)  # (2,4)
        #     Vc = V[:, c].reshape(p, m)  # (7,4) -- wait, V[jp*p+k, c], so V[k, jp, c]
        #     Actually V index is jp*p+k, so V[:, c] reshaped as (m, p) gives V[jp, k, c]
        #     Vc = V[:, c].reshape(m, p)  # (4, 7), Vc[jp, k] = V[jp*p+k, c]
        #     Wc = W[:, c].reshape(p, n)  # (7, 2), Wc[k, i] = W[k*n+i, c]
        #     T[i, j, jp, k] += Uc[i, j] * Vc[jp, k] * Wc[k, i]
        
        # Vectorized: for each c, the contribution to T is:
        # T_c[i,j,jp,k] = Uc[i,j] * Vc[jp,k] * Wc[k,i]
        # This is a 4D tensor. We can compute it as:
        # For fixed c: outer(Uc, Vc, Wc) with appropriate indices
        # T_c = np.einsum('ij,jk,ki->ijpk', Uc, Vc, Wc) -- no, that's not right either
        
        # Let me just loop over c since r=50 is small
        T = np.zeros((n, m, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)       # (2, 4)
            Vc = V[:, c].reshape(m, p)       # (4, 7)
            Wc = W[:, c].reshape(p, n)       # (7, 2)
            # T[i,j,jp,k] += Uc[i,j] * Vc[jp,k] * Wc[k,i]
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            T[i, j, jp, k] += Uc[i, j] * Vc[jp, k] * Wc[k, i]
        
        # Check T[i,j,jp,k] == delta_{j,jp}
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        if abs(T[i, j, jp, k] - expected) > tol:
                            return False
        return True
    
    # Build the rank-50 decomposition
    U, V, W = build_rank50()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = 50
    
    # MPS-style rank reduction: check if any column is linearly dependent
    # Strategy: for each column c, check if removing it still allows valid reconstruction
    # by seeing if the "residual" can be absorbed by other columns
    
    if time.time() < deadline - 10:
        # Try to find a redundant column using SVD on the factor structure
        # The key insight: if column c is a linear combination of other columns
        # in a way that preserves the tensor identity, we can remove it.
        
        # For each column c, try removing it and checking if the tensor is still valid
        # This is O(r * verification_cost) which might be too slow for r=50
        # Instead, use a smarter approach: check rank of the "interaction" matrices
        
        # The tensor T[i,j,jp,k] = sum_c U[i*m+j,c] V[jp*p+k,c] W[k*n+i,c]
        # For this to equal delta_{j,jp}, we need specific rank conditions.
        
        # Quick check: see if any column has all zeros in one of the matrices
        # (which would make it trivially removable)
        for c in range(50):
            if np.all(U[:, c] == 0) or np.all(V[:, c] == 0) or np.all(W[:, c] == 0):
                # This column contributes nothing, remove it
                mask = np.ones(50, dtype=bool)
                mask[c] = False
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = 49
                    break
        
        # Try pairwise column elimination: can columns c1 and c2 be merged?
        # This is more complex; skip for now due to time constraints
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
