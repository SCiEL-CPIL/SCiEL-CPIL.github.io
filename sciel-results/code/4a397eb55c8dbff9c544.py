import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Build the rank-42 Strassen decomposition
    rank42 = 42
    U42 = np.zeros((dim_A, rank42))
    V42 = np.zeros((dim_B, rank42))
    W42 = np.zeros((dim_C, rank42))
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        """Add a 2x2x2 Strassen block at position (i_start, j_start, k_start)"""
        # Strassen's 7 products for 2x2 matrix multiplication
        U42[(i_start+0)*m + (j_start+0), offset+0] += 1
        U42[(i_start+1)*m + (j_start+1), offset+0] += 1
        V42[(j_start+0)*p + (k_start+0), offset+0] += 1
        V42[(j_start+1)*p + (k_start+1), offset+0] += 1
        W42[(k_start+0)*n + (i_start+0), offset+0] += 1
        W42[(k_start+1)*n + (i_start+1), offset+0] += 1
        
        U42[(i_start+1)*m + (j_start+0), offset+1] += 1
        U42[(i_start+1)*m + (j_start+1), offset+1] += 1
        V42[(j_start+0)*p + (k_start+0), offset+1] += 1
        W42[(k_start+0)*n + (i_start+1), offset+1] += 1
        W42[(k_start+1)*n + (i_start+1), offset+1] -= 1
        
        U42[(i_start+0)*m + (j_start+0), offset+2] += 1
        V42[(j_start+0)*p + (k_start+1), offset+2] += 1
        V42[(j_start+1)*p + (k_start+1), offset+2] -= 1
        W42[(k_start+1)*n + (i_start+0), offset+2] += 1
        W42[(k_start+1)*n + (i_start+1), offset+2] += 1
        
        U42[(i_start+1)*m + (j_start+1), offset+3] += 1
        V42[(j_start+1)*p + (k_start+0), offset+3] += 1
        V42[(j_start+0)*p + (k_start+0), offset+3] -= 1
        W42[(k_start+0)*n + (i_start+0), offset+3] += 1
        W42[(k_start+0)*n + (i_start+1), offset+3] += 1
        
        U42[(i_start+0)*m + (j_start+0), offset+4] += 1
        U42[(i_start+0)*m + (j_start+1), offset+4] += 1
        V42[(j_start+1)*p + (k_start+1), offset+4] += 1
        W42[(k_start+0)*n + (i_start+0), offset+4] -= 1
        W42[(k_start+1)*n + (i_start+0), offset+4] += 1
        
        U42[(i_start+1)*m + (j_start+0), offset+5] += 1
        U42[(i_start+0)*m + (j_start+0), offset+5] -= 1
        V42[(j_start+0)*p + (k_start+0), offset+5] += 1
        V42[(j_start+0)*p + (k_start+1), offset+5] += 1
        W42[(k_start+1)*n + (i_start+1), offset+5] += 1
        
        U42[(i_start+0)*m + (j_start+1), offset+6] += 1
        U42[(i_start+1)*m + (j_start+1), offset+6] -= 1
        V42[(j_start+1)*p + (k_start+0), offset+6] += 1
        V42[(j_start+1)*p + (k_start+1), offset+6] += 1
        W42[(k_start+0)*n + (i_start+0), offset+6] += 1
    
    # Six 2x2x2 blocks covering the 2x4x6 multiplication
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    def verify_solution_fast(U, V, W, rank, atol=1e-9):
        """Fast verification using precomputed indices"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    total = np.dot(U[a, :rank], V[b, :rank] * W[w, :rank])
                    if abs(total - 1.0) > atol:
                        return False
        return True
    
    # Verify the rank-42 solution
    if not verify_solution_fast(U42, V42, W42, rank42):
        rank_school = n * m * p
        U_school = np.zeros((dim_A, rank_school))
        V_school = np.zeros((dim_B, rank_school))
        W_school = np.zeros((dim_C, rank_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Improved rank reduction: check if column r is a linear combination of others
    # simultaneously in U, V, and W. This is a system of linear equations.
    # For column r to be redundant: there exist c_s (s != r) such that:
    #   U[:,r] = sum_{s!=r} c_s * U[:,s]
    #   V[:,r] = sum_{s!=r} c_s * V[:,s]
    #   W[:,r] = sum_{s!=r} c_s * W[:,s]
    # This gives (dim_A + dim_B + dim_C) equations in (rank-1) unknowns.
    
    def find_redundant_columns(U, V, W, rank, atol=1e-8):
        """Find all columns that are linear combinations of the others."""
        redundant = set()
        for r in range(rank):
            # Build the system: find c such that U[:,r] - sum c_s U[:,s] = 0
            # for all s != r, and similarly for V, W
            # Stack: [U[:,s] for s != r] as columns, target is U[:,r]
            # Similarly for V and W
            
            # Build augmented system
            # Variables: c_0, ..., c_{rank-2} (coefficients for all columns except r)
            # Equations from U: U[:,r] = U[:, others] @ c
            # Equations from V: V[:,r] = V[:, others] @ c
            # Equations from W: W[:,r] = W[:, others] @ c
            
            others = [s for s in range(rank) if s != r]
            if len(others) == 0:
                continue
                
            # Build the linear system
            # A is (dim_A + dim_B + dim_C) x (rank - 1)
            # b is (dim_A + dim_B + dim_C)
            
            A_parts = []
            b_parts = []
            
            # U part
            U_others = U[:, others]  # (dim_A, rank-1)
            A_parts.append(U_others)
            b_parts.append(U[:, r])
            
            # V part
            V_others = V[:, others]  # (dim_B, rank-1)
            A_parts.append(V_others)
            b_parts.append(V[:, r])
            
            # W part
            W_others = W[:, others]  # (dim_C, rank-1)
            A_parts.append(W_others)
            b_parts.append(W[:, r])
            
            A = np.vstack(A_parts)  # (dim_A+dim_B+dim_C, rank-1)
            b = np.concatenate(b_parts)  # (dim_A+dim_B+dim_C,)
            
            # Solve least squares
            try:
                c, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=atol)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is good
            residual = np.linalg.norm(A @ c - b)
            norm_b = np.linalg.norm(b)
            if norm_b > 1e-12:
                rel_residual = residual / norm_b
            else:
                rel_residual = residual
            
            if rel_residual < 1e-6:
                redundant.add(r)
        
        return redundant
    
    # Iteratively remove redundant columns
    U = U42.copy()
    V = V42.copy()
    W = W42.copy()
    current_rank = rank42
    
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if current_rank == 0:
            break
        
        redundant = find_redundant_columns(U, V, W, current_rank)
        
        if not redundant:
            break
        
        # Remove all redundant columns at once
        keep = [i for i in range(current_rank) if i not in redundant]
        if len(keep) == current_rank:
            break
        
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        current_rank = len(keep)
    
    # Additionally, try zeroing individual columns as a fallback
    for _ in range(current_rank):
        if time.time() > deadline:
            break
        
        removed = False
        for col in range(current_rank - 1, -1, -1):
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, col] = 0
            V_test[:, col] = 0
            W_test[:, col] = 0
            if verify_solution_fast(U_test, V_test, W_test, current_rank, atol=1e-8):
                # Remove this column
                keep = [i for i in range(current_rank) if i != col]
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    # Round to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Final verification
    if not verify_solution_fast(U, V, W, current_rank, atol=1e-9):
        # Return the known rank-42 solution
        U, V, W = U42, V42, W42
        current_rank = rank42
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
