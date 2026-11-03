import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor for the full problem
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook fallback (rank = n*m*p = 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        P = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(P, T_full, atol=tol)

    def project_to_half_integers(U, V, W):
        """Project matrices to nearest half-integers."""
        U_proj = np.round(2 * U) / 2.0
        V_proj = np.round(2 * V) / 2.0
        W_proj = np.round(2 * W) / 2.0
        return U_proj, V_proj, W_proj

    def find_decomposition(n_sub, m_sub, p_sub, target_rank, max_iter=200, trials=50):
        """
        Find a decomposition for a sub-problem n_sub x m_sub x p_sub.
        Returns U, V, W if found, else None.
        """
        nm_sub = n_sub * m_sub
        mp_sub = m_sub * p_sub
        pn_sub = p_sub * n_sub
        
        T_sub = np.zeros((nm_sub, mp_sub, pn_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0

        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        for trial in range(trials):
            U = np.random.randn(nm_sub, target_rank)
            V = np.random.randn(mp_sub, target_rank)
            W = np.random.randn(pn_sub, target_rank)
            
            # Normalize to avoid divergence
            for r in range(target_rank):
                norm_U = np.linalg.norm(U[:, r])
                norm_V = np.linalg.norm(V[:, r])
                norm_W = np.linalg.norm(W[:, r])
                if norm_U > 1e-9 and norm_V > 1e-9 and norm_W > 1e-9:
                    # Scale such that product of norms is 1? Or just normalize each to 1?
                    # Let's just run ALS
                    pass
            
            for it in range(max_iter):
                # Update U
                # T_sub[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Fix V, W. Solve for U.
                # For each i, U[i, :] is a vector.
                # T_sub[i, :, :] = sum_r U[i, r] (V[:, r] W[:, r])
                # Let M_VW[j, k, r] = V[j, r] W[k, r]
                # Then T_sub[i, j, k] = sum_r U[i, r] M_VW[j, k, r]
                # This is a linear system for U[i, :]
                # Reshape T_sub to (nm_sub, mp_sub * pn_sub)
                # M_VW to (mp_sub * pn_sub, target_rank)
                VW = np.einsum('jr,kr->jkr', V, W) # mp_sub x pn_sub x target_rank
                VW_flat = VW.transpose(2, 0, 1).reshape(target_rank, mp_sub * pn_sub).T # (mp_sub*pn_sub, target_rank)
                T_mat = T_sub.reshape(nm_sub, mp_sub * pn_sub)
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW_flat, T_mat, rcond=None)
                except:
                    U_new = U
                
                # Update V
                UW = np.einsum('ir,kr->ikr', U, W) # nm_sub x pn_sub x target_rank
                UW_flat = UW.transpose(2, 0, 1).reshape(target_rank, nm_sub * pn_sub).T # (nm_sub*pn_sub, target_rank)
                T_mat_V = T_sub.transpose(1, 0, 2).reshape(mp_sub, nm_sub * pn_sub)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW_flat, T_mat_V, rcond=None)
                except:
                    V_new = V
                
                # Update W
                UV = np.einsum('ir,jr->ijr', U, V) # nm_sub x mp_sub x target_rank
                UV_flat = UV.transpose(2, 0, 1).reshape(target_rank, nm_sub * mp_sub).T # (nm_sub*mp_sub, target_rank)
                T_mat_W = T_sub.transpose(2, 0, 1).reshape(pn_sub, nm_sub * mp_sub)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV_flat, T_mat_W, rcond=None)
                except:
                    W_new = W
                
                U, V, W = U_new, V_new, W_new
                
                # Check error
                P = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(P - T_sub)
                if err < 1e-8:
                    return U, V, W
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # If we didn't find an exact one, check if the best one is close enough to be projected
        if best_U is not None and best_err < 1e-4:
             U_proj, V_proj, W_proj = project_to_half_integers(best_U, best_V, best_W)
             P = np.einsum('ur,vr,wr->uvw', U_proj, V_proj, W_proj)
             if np.allclose(P, T_sub, atol=1e-8):
                 return U_proj, V_proj, W_proj
                 
        return None

    start_time = time.time()
    deadline = 145.0
    
    best_rank = rank_sb
    best_result = (U_sb, V_sb, W_sb)
    
    # Strategy 1: Decompose into 2x3x2 (rank 10) blocks and 2x3x1 (rank 6)
    # 7 = 2 + 2 + 2 + 1
    # Need to find rank 10 decomposition for 2x3x2
    if time.time() < deadline - 10:
        print("Searching for 2x3x2 rank 10 decomposition...")
        dec_232 = find_decomposition(2, 3, 2, 10, max_iter=100, trials=20)
        if dec_232:
            U_232, V_232, W_232 = dec_232
            rank_232 = U_232.shape[1]
            # 2x3x1 is always rank 6
            U_231 = np.zeros((6, 6))
            V_231 = np.zeros((6, 6))
            W_231 = np.zeros((2, 6))
            for i in range(2):
                for j in range(3):
                    idx = i * 3 + j
                    U_231[idx, idx] = 1.0
                    V_231[j, idx] = 1.0
                    W_231[i, idx] = 1.0
            
            # Construct full decomposition
            total_rank = rank_232 * 3 + 6
            U_full = np.zeros((6, total_rank))
            V_full = np.zeros((21, total_rank))
            W_full = np.zeros((14, total_rank))
            
            col_offset = 0
            # Block 1: cols 0,1 of B (k=0,1)
            # V indices for block 1: j*2 + k. j in 0..2, k in 0..1.
            # U indices: i*3 + j.
            # W indices: k*2 + i.
            
            # We need to map the sub-problem indices to the full problem indices.
            # Sub-problem: n=2, m=3, p=2.
            # U_sub[i*3+j, r] -> U_full[i*3+j, col_offset + r]
            # V_sub[j*2+k, r] -> V_full[j*2+k, col_offset + r] for k in 0,1
            # W_sub[k*2+i, r] -> W_full[k*2+i, col_offset + r] for k in 0,1
            
            U_full[:, col_offset:col_offset+rank_232] = U_232
            # V_232 has size 6x10. Rows correspond to j*2+k.
            # In full V, rows are j*7+k.
            # For k=0,1, the rows are j*7+0, j*7+1.
            # We need to scatter V_232 into V_full.
            for j in range(3):
                for k in range(2):
                    row_sub = j * 2 + k
                    row_full = j * 7 + k
                    V_full[row_full, col_offset:col_offset+rank_232] = V_232[row_sub, :]
            
            # W_232 has size 4x10. Rows correspond to k*2+i.
            # In full W, rows are k*2+i? No, full W is k*n+i = k*2+i.
            # So W_232 rows map directly to W_full rows for k=0,1.
            W_full[0:4, col_offset:col_offset+rank_232] = W_232
            
            col_offset += rank_232
            
            # Block 2: cols 2,3 of B (k=2,3)
            U_full[:, col_offset:col_offset+rank_232] = U_232
            for j in range(3):
                for k in range(2):
                    row_sub = j * 2 + k
                    row_full = j * 7 + (k + 2)
                    V_full[row_full, col_offset:col_offset+rank_232] = V_232[row_sub, :]
            # W_232 rows k*2+i. Here k is local 0,1. Global k is 2,3.
            # W_full row index is k_global * 2 + i.
            # So for local k=0 (global 2), rows are 2*2+i = 4+i.
            # For local k=1 (global 3), rows are 3*2+i = 6+i.
            W_full[4:8, col_offset:col_offset+rank_232] = W_232
            col_offset += rank_232
            
            # Block 3: cols 4,5 of B (k=4,5)
            U_full[:, col_offset:col_offset+rank_232] = U_232
            for j in range(3):
                for k in range(2):
                    row_sub = j * 2 + k
                    row_full = j * 7 + (k + 4)
                    V_full[row_full, col_offset:col_offset+rank_232] = V_232[row_sub, :]
            W_full[8:12, col_offset:col_offset+rank_232] = W_232
            col_offset += rank_232
            
            # Block 4: col 6 of B (k=6)
            # Use 2x3x1 decomposition
            U_full[:, col_offset:col_offset+6] = U_231
            # V_231 rows are j. Global V rows are j*7+6.
            for j in range(3):
                V_full[j*7+6, col_offset:col_offset+6] = V_231[j, :]
            # W_231 rows are i. Global W rows are 6*2+i = 12+i.
            W_full[12:14, col_offset:col_offset+6] = W_231
            
            # Verify
            if verify(U_full, V_full, W_full):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = (U_full, V_full, W_full)
                print(f"Found rank {total_rank} decomposition.")
                break # No need to try other strategies if we found a good one? Maybe try 35.

    # Strategy 2: Decompose into 2x3x3 (rank 15?) and 2x3x2 (rank 10)
    # 7 = 3 + 2 + 2
    if time.time() < deadline - 10:
        print("Searching for 2x3x3 rank 15 decomposition...")
        dec_233 = find_decomposition(2, 3, 3, 15, max_iter=100, trials=20)
        if dec_233 and dec_232: # Reuse dec_232 if found
            U_233, V_233, W_233 = dec_233
            rank_233 = U_233.shape[1]
            rank_232 = U_232.shape[1]
            
            total_rank = rank_233 + rank_232 * 2
            U_full = np.zeros((6, total_rank))
            V_full = np.zeros((21, total_rank))
            W_full = np.zeros((14, total_rank))
            
            col_offset = 0
            # Block 1: cols 0,1,2 (k=0,1,2)
            U_full[:, col_offset:col_offset+rank_233] = U_233
            for j in range(3):
                for k in range(3):
                    row_sub = j * 3 + k
                    row_full = j * 7 + k
                    V_full[row_full, col_offset:col_offset+rank_233] = V_233[row_sub, :]
            # W_233 rows k*2+i. k in 0,1,2.
            W_full[0:6, col_offset:col_offset+rank_233] = W_233
            col_offset += rank_233
            
            # Block 2: cols 3,4 (k=3,4)
            U_full[:, col_offset:col_offset+rank_232] = U_232
            for j in range(3):
                for k in range(2):
                    row_sub = j * 2 + k
                    row_full = j * 7 + (k + 3)
                    V_full[row_full, col_offset:col_offset+rank_232] = V_232[row_sub, :]
            W_full[6:10, col_offset:col_offset+rank_232] = W_232
            col_offset += rank_232
            
            # Block 3: cols 5,6 (k=5,6)
            U_full[:, col_offset:col_offset+rank_232] = U_232
            for j in range(3):
                for k in range(2):
                    row_sub = j * 2 + k
                    row_full = j * 7 + (k + 5)
                    V_full[row_full, col_offset:col_offset+rank_232] = V_232[row_sub, :]
            W_full[10:14, col_offset:col_offset+rank_232] = W_232
            
            if verify(U_full, V_full, W_full):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = (U_full, V_full, W_full)
                print(f"Found rank {total_rank} decomposition.")

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
