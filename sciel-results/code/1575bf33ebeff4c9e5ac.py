"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of matrices
    # A: n x m
    # B: m x p
    # C: n x p
    
    # Total elements
    dim_A = n * m
    dim_B = m * p
    dim_C = n * p
    
    # Construct the target tensor T
    # T[a, b, c] = 1 if C[c] depends on A[a] and B[b]
    # Indices:
    # a = i * m + j
    # b = j * p + k
    # c = i * p + k
    # Note: The task description says W[k*n+i, r].
    # Let's verify the W index mapping.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W row index is $k \cdot n + i$.
    # In standard matrix mult $C_{ik} = \sum A_{ij} B_{jk}$.
    # $C$ is $n \times p$. $C_{ik}$.
    # $W$ should correspond to $C$ elements.
    # If $W$ index is $k \cdot n + i$, it's column-major for C?
    # Let's check the seed code:
    # W[k * n + i][column] = 1
    # Yes, the seed uses $k \cdot n + i$.
    # So $W$ rows are indexed by $c' = k \cdot n + i$.
    # My previous derivation assumed $c = i \cdot p + k$ (row-major).
    # I must use $c' = k \cdot n + i$ for the W matrix rows.
    
    # Let's re-map indices for the linear system.
    # Equation: C_{ik} = sum_j A_{ij} B_{jk}
    # In terms of factors:
    # sum_r U_{a,r} V_{b,r} W_{c',r} = T_{a,b,c'}
    # where:
    # a = i * m + j
    # b = j * p + k
    # c' = k * n + i
    
    # We need to build T[a, b, c']
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_prime = k * n + i
                T[a, b, c_prime] = 1
    
    # Schoolbook solution
    rank_schoolbook = dim_A * dim_B // dim_A # n*m*p ? No.
    # Schoolbook rank is n*m*p
    rank_sb = n * m * p
    U_sb = np.zeros((dim_A, rank_sb))
    V_sb = np.zeros((dim_B, rank_sb))
    W_sb = np.zeros((dim_C, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_prime = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c_prime, col] = 1
                col += 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try to find lower rank
    # We search R from rank_sb - 1 down to some limit
    # Using the linear system approach
    
    # Pre-compute T vectors for each c'
    # T_c' is a vector of length dim_A * dim_B
    T_flat = T.reshape(dim_A * dim_B, dim_C)
    
    for R in range(rank_sb - 1, 10, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        restarts = 10
        found = False
        
        for _ in range(restarts):
            if time.time() > deadline:
                break
                
            # Random integer initialization for U and V
            # Small range to keep numbers manageable
            U = np.random.randint(-2, 3, size=(dim_A, R))
            V = np.random.randint(-2, 3, size=(dim_B, R))
            
            # Solve for W
            W = np.zeros((dim_C, R))
            valid = True
            
            for c_idx in range(dim_C):
                # Build A_c matrix
                # A_c[a * dim_B + b, r] = U[a, r] * V[b, r]
                # Vectorize:
                # U is (dim_A, R), V is (dim_B, R)
                # We need outer product structure per r?
                # A_c[:, r] = (U * V.T).flatten() ?
                # U[:, r] is length dim_A. V[:, r] is length dim_B.
                # Outer product U[:, r] * V[:, r].T is dim_A x dim_B.
                # Flatten to dim_A * dim_B.
                
                # Efficient construction:
                # A_c = np.zeros((dim_A * dim_B, R))
                # for r in range(R):
                #     A_c[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                # This is slow. Let's use broadcasting.
                # U_expanded: dim_A x 1 x R
                # V_expanded: 1 x dim_B x R
                # Prod: dim_A x dim_B x R
                # Reshape to dim_A*dim_B x R
                
                Prod = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(dim_A * dim_B, R)
                
                # Solve Prod @ W[c_idx, :] = T_flat[:, c_idx]
                # Use least squares
                try:
                    W_row, residuals, rank, sv = np.linalg.lstsq(Prod, T_flat[:, c_idx], rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Round to nearest 0.5
                W_row_rounded = np.round(W_row * 2) / 2.0
                W[c_idx, :] = W_row_rounded
                
                # Check exact match
                # Residual should be 0
                # Check: Prod @ W_row_rounded == T_flat[:, c_idx]
                if not np.allclose(Prod @ W_row_rounded, T_flat[:, c_idx], atol=1e-9):
                    valid = False
                    break
            
            if valid:
                # Verify entire tensor just in case
                # Reconstruct T_recon
                # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # Use einsum
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.array_equal(T_recon, T):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    found = True
                    break # Break restarts, go to lower R
        
        if found:
            continue # Try lower rank
        else:
            # If we failed to find for this R, we might try a few more restarts or break?
            # Given time constraints, we might break if we've tried many R's without success
            # But let's keep trying lower R as it becomes easier (more degrees of freedom? No, fewer constraints on rank, but harder to satisfy exactly with integers?)
            # Actually, lower rank is HARDER.
            # If we fail at R=44, we are unlikely to succeed at R=43 quickly.
            # However, the search space is discrete.
            # Let's just continue to the next lower R, but limit total time.
            pass

    # Convert to lists of lists with integer/half-integer values
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
