"""Matrix multiplication 4x5x5 - rank-85 Strassen with dependency-based reduction."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    start_time = time.time()
    deadline = 140.0
    
    # Build rank-85 Strassen-based decomposition
    rank_base = 85
    U = np.zeros((nm, rank_base), dtype=int)
    V = np.zeros((mp, rank_base), dtype=int)
    W = np.zeros((pn, rank_base), dtype=int)
    
    # 2x2x2 Strassen factors
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    s_u_sparse = [[(i, v) for i, v in enumerate(s_u[:, k]) if v != 0] for k in range(7)]
    s_v_sparse = [[(i, v) for i, v in enumerate(s_v[:, k]) if v != 0] for k in range(7)]
    s_w_sparse = [[(i, v) for i, v in enumerate(s_w[:, k]) if v != 0] for k in range(7)]
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        for inner_k in range(7):
            col = col_start + inner_k
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for local_idx, u_val in s_u_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * a_br + lr
                    gj = 2 * a_bc + lc
                    U[gi * 5 + gj, col] += a_coeff * u_val
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for local_idx, v_val in s_v_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gj = 2 * b_br + lr
                    gk = 2 * b_bc + lc
                    V[gj * 5 + gk, col] += b_coeff * v_val
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for local_idx, w_val in s_w_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * c_br + lr
                    gk = 2 * c_bc + lc
                    W[gk * 4 + gi, col] += c_coeff * w_val
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
        add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)
    
    col = 49
    
    # 4x4x1: A[:4,:4] @ B[:4,4] -> C[:4,4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
    
    # 4x1x4: A[:4,4] * B[4,:4] -> C[:4,:4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
    
    # 4x1x1: A[:4,4] * B[4,4] -> C[:4,4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W, rank_base):
        # Fallback to schoolbook
        rank_schoolbook = n*m*p
        U = np.zeros((nm, rank_schoolbook), dtype=int)
        V = np.zeros((mp, rank_schoolbook), dtype=int)
        W = np.zeros((pn, rank_schoolbook), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # For a tensor decomposition, column c can be eliminated if there exist
    # coefficients alpha_d such that for all (i,j,k):
    # U[i,j,c]*V[j,k,c]*W[k,i,c] = sum_d alpha_d * U[i,j,d]*V[j,k,d]*W[k,i,d]
    # This is hard to check directly. Instead, try removing one column at a time
    # and see if we can solve for a correction.
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank_base
    
    # Try removing columns from the end (boundary terms are more likely redundant)
    for target_rank in range(rank_base - 1, max(rank_base - 20, 60), -1):
        if time.time() - start_time > deadline - 5:
            break
        
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        current_rank = target_rank + 1
        
        # Try to eliminate column current_rank-1
        # Check if it's "redundant" by seeing if the tensor it contributes
        # can be matched by other columns
        removed_col = current_rank - 1
        
        # The contribution of column c is: T_c[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c]
        # We need: T_removed = sum_{d != removed} alpha_d * T_d
        # This is a linear system in alpha_d
        
        # Build the linear system: for each (i,j,k), we have:
        # sum_d alpha_d * U[i,j,d]*V[j,k,d]*W[k,i,d] = U[i,j,removed]*V[j,k,removed]*W[k,i,removed]
        
        # Number of equations: n*m*p = 100
        # Number of unknowns: current_rank - 1 = target_rank
        # If target_rank < 100, we have more equations than unknowns (overdetermined)
        # Use least squares
        
        if current_rank - 1 >= 1:
            # Build matrix A where A[(i,j,k), d] = U[i,j,d]*V[j,k,d]*W[k,i,d]
            # and b[(i,j,k)] = U[i,j,removed]*V[j,k,removed]*W[k,i,removed]
            
            # Only consider columns 0..current_rank-2 (not the removed one)
            num_eqs = n * m * p
            num_unknowns = current_rank - 1
            
            if num_unknowns > 0:
                A_mat = np.zeros((num_eqs, num_unknowns))
                b_vec = np.zeros(num_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # b
                            b_vec[eq_idx] = U_try[i*m+j, removed_col] * V_try[j*p+k, removed_col] * W_try[k*n+i, removed_col]
                            # A columns
                            for d in range(num_unknowns):
                                A_mat[eq_idx, d] = U_try[i*m+j, d] * V_try[j*p+k, d] * W_try[k*n+i, d]
                            eq_idx += 1
                
                # Solve least squares
                try:
                    alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    alpha = None
                
                if alpha is not None:
                    # Check if the solution is exact (residual ~ 0)
                    residual = np.linalg.norm(A_mat @ alpha - b_vec)
                    if residual < 1e-9:
                        # Found an exact dependency! Remove the column
                        # Update: for each remaining column d, add alpha[d] * T_d to compensate
                        # Actually, we just remove the column and adjust others
                        # But the decomposition requires integer/half-integer entries
                        # If alpha has non-integer values, we can't use this directly
                        
                        # Check if alpha entries are close to integers or half-integers
                        alpha_rounded = np.round(alpha * 2) / 2
                        if np.allclose(alpha, alpha_rounded, atol=1e-9):
                            # Valid! Remove column and add contributions
                            for d in range(num_unknowns):
                                if alpha_rounded[d] != 0:
                                    U_try[:, d] += alpha_rounded[d] * U_try[:, removed_col]
                                    V_try[:, d] += alpha_rounded[d] * V_try[:, removed_col]
                                    W_try[:, d] += alpha_rounded[d] * W_try[:, removed_col]
                            
                            # Remove the column
                            keep = list(range(removed_col)) + list(range(removed_col + 1, current_rank))
                            U_try = U_try[:, keep]
                            V_try = V_try[:, keep]
                            W_try = W_try[:, keep]
                            
                            if verify(U_try, V_try, W_try, target_rank):
                                best_U = U_try.copy()
                                best_V = V_try.copy()
                                best_W = W_try.copy()
                                best_rank = target_rank
                                break
        
        if best_rank == target_rank:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
