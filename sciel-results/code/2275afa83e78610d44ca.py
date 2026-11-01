import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def build_rank32():
        """
        Construct the rank-32 decomposition using Strassen blocks.
        4 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x1 (rank 4).
        Total rank: 4*7 + 4 = 32.
        """
        rank = 32
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Strassen 2x2x2 decomposition
        # Indices for U: (i, j) where i in 0..1, j in 0..1 -> idx = i*m+j
        # Indices for V: (j, k) where j in 0..1, k in 0..1 -> idx = j*p+k
        # Indices for W: (k, i) where k in 0..1, i in 0..1 -> idx = k*n+i
        
        def add_strassen_block(k_offset):
            nonlocal col
            # c=0: M1 = (a00+a11)(b00+b11)
            U[0, col] = 1; U[3, col] = 1
            V[0*p + k_offset, col] = 1; V[1*p + k_offset+1, col] = 1
            W[k_offset*n + 0, col] = 1; W[(k_offset+1)*n + 1, col] = 1
            col += 1
            
            # c=1: M2 = (a10+a11)b00
            U[2, col] = 1; U[3, col] = 1
            V[0*p + k_offset, col] = 1
            W[k_offset*n + 1, col] = 1; W[(k_offset+1)*n + 1, col] = -1
            col += 1
            
            # c=2: M3 = a00(b10-b11)
            U[0, col] = 1
            V[0*p + k_offset+1, col] = 1; V[1*p + k_offset+1, col] = -1
            W[(k_offset+1)*n + 0, col] = 1; W[(k_offset+1)*n + 1, col] = 1
            col += 1
            
            # c=3: M4 = a11(b01-b00)
            U[3, col] = 1
            V[1*p + k_offset, col] = 1; V[0*p + k_offset, col] = -1
            W[k_offset*n + 0, col] = 1; W[k_offset*n + 1, col] = 1
            col += 1
            
            # c=4: M5 = (a00+a01)b11
            U[0, col] = 1; U[1, col] = 1
            V[1*p + k_offset+1, col] = 1
            W[k_offset*n + 0, col] = -1; W[(k_offset+1)*n + 0, col] = 1
            col += 1
            
            # c=5: M6 = (a10-a00)(b00+b10)
            U[2, col] = 1; U[0, col] = -1
            V[0*p + k_offset, col] = 1; V[0*p + k_offset+1, col] = 1
            W[(k_offset+1)*n + 1, col] = 1
            col += 1
            
            # c=6: M7 = (a01-a11)(b01+b11)
            U[1, col] = 1; U[3, col] = -1
            V[1*p + k_offset, col] = 1; V[1*p + k_offset+1, col] = 1
            W[k_offset*n + 0, col] = 1
            col += 1

        def add_schoolbook_block(k_offset):
            nonlocal col
            # c=0: a00*b00
            U[0, col] = 1
            V[0*p + k_offset, col] = 1
            W[k_offset*n + 0, col] = 1
            col += 1
            
            # c=1: a01*b10
            U[1, col] = 1
            V[1*p + k_offset, col] = 1
            W[k_offset*n + 0, col] = 1
            col += 1
            
            # c=2: a10*b00
            U[2, col] = 1
            V[0*p + k_offset, col] = 1
            W[k_offset*n + 1, col] = 1
            col += 1
            
            # c=3: a11*b10
            U[3, col] = 1
            V[1*p + k_offset, col] = 1
            W[k_offset*n + 1, col] = 1
            col += 1

        # Apply 4 Strassen blocks for k=0..7
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        add_strassen_block(6)
        
        # Apply 1 Schoolbook block for k=8
        add_schoolbook_block(8)
        
        return U, V, W

    def compute_residual(U, V, W):
        """
        Compute the sum of squared residuals for the tensor decomposition.
        T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
        where i corresponds to (i_idx, j_idx) in U, 
        j corresponds to (j_idx, k_idx) in V,
        k corresponds to (k_idx, i_idx) in W.
        
        Actually, the indices are:
        U row: i*m + j
        V row: j*p + k
        W row: k*n + i
        
        The tensor entry T[i, j, k] should be 1.
        We compute M[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        """
        # Efficiently compute the tensor contraction
        # M[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We only need to check the specific indices (i*m+j, j*p+k, k*n+i)
        
        residual_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Dot product of the rows
                    val = np.dot(U[idx_u], V[idx_v]) # This is wrong, need to sum over c with W
                    # val = sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    val = np.einsum('c,c,c->', U[idx_u], V[idx_v], W[idx_w])
                    residual_sum += (val - 1.0) ** 2
        return residual_sum

    def try_reduce_rank(U0, V0, W0, time_budget):
        """
        Attempt to reduce rank by 1 using optimization.
        """
        start_time = time.time()
        rank = U0.shape[1]
        nm, mp, pn = U0.shape[0], V0.shape[0], W0.shape[0]
        
        # Precompute indices for fast residual calculation
        # We will use a vectorized approach for the objective function
        # Flatten U, V, W into a single parameter vector x
        
        def objective(x, target_col, lam):
            U_flat = x[:nm * rank]
            V_flat = x[nm * rank : nm * rank + mp * rank]
            W_flat = x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Compute residual
            # To speed up, we can compute the full tensor M and sum squares
            # M[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
            # But we only need the diagonal-like indices.
            # Let's just loop, it's small (2*2*9 = 36 terms)
            res_sq = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u] * V[idx_v], W[idx_w])
                        res_sq += (val - 1.0) ** 2
            
            # Penalty for target column
            col_penalty = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            
            return res_sq + lam * col_penalty

        # Try eliminating different columns
        # Heuristic: Columns from the schoolbook part are less "entangled"
        # The last 4 columns (28-31) are from the schoolbook block.
        # Let's try eliminating one of those first.
        
        candidate_cols = [28, 29, 30, 31]
        # Also try some Strassen columns, maybe they can be merged
        # But Strassen columns are critical for the 2x2x2 blocks.
        
        for target_col in candidate_cols:
            if time.time() - start_time > time_budget * 0.7:
                break
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Try a few lambda values
            for lam in [1.0, 10.0, 100.0]:
                if time.time() - start_time > time_budget * 0.8:
                    break
                
                try:
                    result = minimize(objective, x0, args=(target_col, lam), method='L-BFGS-B', 
                                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9})
                except Exception:
                    continue
                
                U_flat = result.x[:nm * rank]
                V_flat = result.x[nm * rank : nm * rank + mp * rank]
                W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                
                U_opt = U_flat.reshape(nm, rank)
                V_opt = V_flat.reshape(mp, rank)
                W_opt = W_flat.reshape(pn, rank)
                
                # Check if the target column is small enough
                col_norm = np.sqrt(np.sum(U_opt[:, target_col]**2) + np.sum(V_opt[:, target_col]**2) + np.sum(W_opt[:, target_col]**2))
                
                if col_norm < 1e-4:
                    # Try to drop the column
                    U_new = np.delete(U_opt, target_col, axis=1)
                    V_new = np.delete(V_opt, target_col, axis=1)
                    W_new = np.delete(W_opt, target_col, axis=1)
                    
                    # Verify the new decomposition
                    # We need to check if the residual is close to 0
                    # And if the values are close to integers or half-integers?
                    # The task says "integer/half-integer entries".
                    # Optimization might yield floats. We need to round.
                    
                    # Let's check the residual of the rounded version
                    # First, check raw residual
                    res_raw = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.dot(U_new[idx_u] * V_new[idx_v], W_new[idx_w])
                                res_raw += (val - 1.0) ** 2
                    
                    if res_raw < 1e-6:
                        # Try rounding to nearest 0.5
                        U_rounded = np.round(U_new * 2) / 2
                        V_rounded = np.round(V_new * 2) / 2
                        W_rounded = np.round(W_new * 2) / 2
                        
                        res_rounded = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    idx_u = i * m + j
                                    idx_v = j * p + k
                                    idx_w = k * n + i
                                    val = np.dot(U_rounded[idx_u] * V_rounded[idx_v], W_rounded[idx_w])
                                    res_rounded += (val - 1.0) ** 2
                        
                        if res_rounded < 1e-9:
                            # Success!
                            return U_rounded, V_rounded, W_rounded
        
        return None

    # 1. Build the rank-32 solution
    U32, V32, W32 = build_rank32()
    
    # Verify rank 32
    res32 = compute_residual(U32, V32, W32)
    if res32 > 1e-9:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # 2. Try to reduce to rank 31
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        result_31 = try_reduce_rank(U32, V32, W32, deadline - elapsed - 10)
        if result_31 is not None:
            U31, V31, W31 = result_31
            # Final verification
            res31 = compute_residual(U31, V31, W31)
            if res31 < 1e-9:
                return {"U": U31, "V": V31, "W": W31}

    # 3. Return rank 32
    return {"U": U32, "V": V32, "W": W32}

# EVOLVE-BLOCK-END
