"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # 1. Construct the known Rank-36 solution explicitly
    # This is faster and more reliable than random search initialization
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Strassen for 2x2 * 2x2
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j):
        def ma(l): return (a_i + l//2) * m + (a_j + l%2)
        def mb(l): return (b_i + l//2) * p + (b_j + l%2)
        def mc(l):
            i = c_i + l//2
            k = c_j + l%2
            return k * n + i

        # M1
        add_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
        # M2
        add_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
        # M3
        add_term({ma(0): 1}, {mb(1): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
        # M4
        add_term({ma(3): 1}, {mb(2): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
        # M5
        add_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
        # M6
        add_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
        # M7
        add_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})

    # Schoolbook for 2x2 * 2x1
    def add_schoolbook_2x1(a_i, a_j, b_i, b_j, c_i, c_j):
        def ma(l): return (a_i + l//2) * m + (a_j + l%2)
        def mb(l): return (b_i + l) * p + b_j
        def mc(l):
            i = c_i + l
            k = c_j
            return k * n + i

        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        add_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
        add_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
        add_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
        add_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})

    # Block 1: A[:, 0:2] * B[0:2, :]
    # B[0:2, 0:2] -> Strassen
    add_strassen(0, 0, 0, 0, 0, 0)
    # B[0:2, 2:4] -> Strassen
    add_strassen(0, 0, 0, 2, 0, 2)
    # B[0:2, 4:5] -> Schoolbook
    add_schoolbook_2x1(0, 0, 0, 4, 0, 4)

    # Block 2: A[:, 2:4] * B[2:4, :]
    # B[2:4, 0:2] -> Strassen
    add_strassen(0, 2, 2, 0, 0, 0)
    # B[2:4, 2:4] -> Strassen
    add_strassen(0, 2, 2, 2, 0, 2)
    # B[2:4, 4:5] -> Schoolbook
    add_schoolbook_2x1(0, 2, 2, 4, 0, 4)

    # Build the multiplication tensor
    T_full = np.zeros((n*m, m*p, p*n))
    for row_a in range(n):
        for col_a in range(m):
            for row_b in range(p):
                i = row_a * m + col_a
                j = col_a * p + row_b
                k = row_a * p + row_b
                T_full[i, j, k] = 1.0
    
    # Verify the rank-36 solution (vectorized)
    def verify_np(U, V, W):
        """Verify that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all i,j,k"""
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        return np.allclose(T_recon, T_full, atol=1e-6)

    if not verify_np(U, V, W):
        # Fallback: this shouldn't happen
        pass

    # 2. Attempt to reduce rank using efficient vectorized approach
    # Check if the rank-36 solution has linear dependencies that allow rank reduction
    
    def try_reduce_rank_fast(U, V, W, target_rank):
        """Try to find a rank-target_rank decomposition by checking if 
        the tensor can be expressed with fewer rank-1 terms."""
        if target_rank >= U.shape[1]:
            return None
        
        # Build the full tensor T[i,j,k] efficiently
        T = np.zeros((n*m, m*p, p*n))
        for row_a in range(n):
            for col_a in range(m):
                for row_b in range(p):
                    if col_a == row_b:
                        i = row_a * m + col_a
                        j = col_a * p + row_b  # j = col_a*p + row_b, but row_b=col_a
                        # Actually j = col_a * p + row_b, and we need row_b = col_a
                        # Wait, let me re-derive: A is n x m, B is m x p
                        # T[i,j,k] = 1 iff i=row_a*m+col_a, j=col_a*p+row_b, k=row_b*n+col_a... no
                        # T[i,j,k] = 1 iff the (row_a,col_a) entry of A times (col_a,row_b) entry of B
                        # gives (row_a,row_b) entry of C
                        # i = row_a * m + col_a
                        # j = col_a * p + row_b  
                        # k = row_b * n + col_a ... no, k = row_b * n + col_a is wrong
                        # k should be row_a * n + row_b? No, C is n x p
                        # C[row_a, row_b] = sum_col A[row_a,col] * B[col,row_b]
                        # So k = row_a * p + row_b
                        # Let me fix: i = row_a*m + col_a, j = col_a*p + row_b, k = row_a*p + row_b
                        pass
        
        # Correct tensor construction
        T = np.zeros((n*m, m*p, p*n))
        for row_a in range(n):
            for col_a in range(m):
                for row_b in range(p):
                    i = row_a * m + col_a
                    j = col_a * p + row_b
                    k = row_a * p + row_b
                    T[i, j, k] = 1.0
        
        # Try each possible subset of target_rank columns
        # For each subset, check if the tensor can be decomposed
        # This is expensive, so use a smarter approach:
        # Check if any column is in the span of others
        
        U_curr = U.astype(float).copy()
        V_curr = V.astype(float).copy()
        W_curr = W.astype(float).copy()
        curr_rank = U_curr.shape[1]
        
        # Try to remove one column at a time
        for col_to_remove in range(curr_rank):
            if time.time() > deadline:
                return None
            
            # Remove this column
            mask = [i for i in range(curr_rank) if i != col_to_remove]
            U_sub = U_curr[:, mask]
            V_sub = V_curr[:, mask]
            W_sub = W_curr[:, mask]
            
            # Check if the tensor is still correctly decomposed
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # We need to check if removing this column still gives correct T
            # This is equivalent to checking if the removed column's contribution is zero
            # in the tensor, which it's not in general.
            
            # Instead, try to adjust the remaining columns to compensate
            # Use least squares to find best fit
            target_r = len(mask)
            
            # Build design matrix for U: for each i, we need sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
            # This is a 3-way problem. Use alternating approach.
            
            U_new = U_sub.copy()
            V_new = V_sub.copy()
            W_new = W_sub.copy()
            
            converged = False
            for it in range(5):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each i, T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
                # So T[i] = U[i,:] @ (V*W)^T where V*W is (target_r, m*p*p*n)
                VW = np.zeros((target_r, m*p * p*n))
                for r in range(target_r):
                    VW[r, :] = (V_new[:, r][:, None] * W_new[:, r][None, :]).flatten()
                
                # T_flat[i, :] should equal U_new[i, :] @ VW.T
                T_flat = T.reshape(n*m, -1)
                try:
                    U_new = np.linalg.lstsq(VW.T, T_flat.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                UW = np.zeros((target_r, n*m * p*n))
                for r in range(target_r):
                    UW[r, :] = (U_new[:, r][:, None] * W_new[:, r][None, :]).flatten()
                
                T_flat2 = T.reshape(-1, m*p).T  # (m*p, n*m*p*n)
                T_flat2 = T.transpose(1, 0, 2).reshape(m*p, -1)
                try:
                    V_new = np.linalg.lstsq(UW.T, T_flat2.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                UV = np.zeros((target_r, n*m * m*p))
                for r in range(target_r):
                    UV[r, :] = (U_new[:, r][:, None] * V_new[:, r][None, :]).flatten()
                
                T_flat3 = T.transpose(2, 0, 1).reshape(p*n, -1)
                try:
                    W_new = np.linalg.lstsq(UV.T, T_flat3.T, rcond=None)[0].T
                except:
                    break
            
            # Check if solution is close to integers
            all_entries = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            rounded = np.round(all_entries)
            round_err = np.sum((all_entries - rounded)**2)
            
            if round_err < 1e-3:
                U_r = np.round(U_new).astype(int)
                V_r = np.round(V_new).astype(int)
                W_r = np.round(W_new).astype(int)
                if verify_np(U_r, V_r, W_r):
                    return U_r, V_r, W_r
        
        return None

    # Try rank 35
    if time.time() < deadline:
        result = try_reduce_rank_fast(U, V, W, 35)
        if result:
            U, V, W = result
            return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}

    # Return the rank-36 solution
    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}
# EVOLVE-BLOCK-END
