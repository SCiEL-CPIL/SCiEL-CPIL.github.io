import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
        
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Pre-compute the structure of the linear system A * W_flat = b
    # A is (num_eqs, pn * r)
    # b is (num_eqs,)
    # We will build A dynamically for each r, or pre-compute the sparsity pattern.
    # Since r changes, A width changes.
    
    # Indices for equations
    # eq corresponds to (i, j, j_prime, k_prime, I, K)
    # Coefficient for W[(K, I), c] is U[(i,j), c] * V[(j_prime, k_prime), c]
    # Target b is 1 if i==I and j==j_prime and k_prime==K else 0
    
    # We will search for ranks from 59 down to 50 (or until time runs out)
    # Rank 60 is schoolbook.
    
    for target_rank in range(rank_schoolbook - 1, 40, -1):
        if time.time() > deadline:
            break
            
        # Try to find a solution for this rank
        # We use random sparse integer U, V
        # Entries in {-1, 0, 1}
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate random sparse U, V
            # Density 0.2
            U_rand = np.random.choice([-1, 0, 1], size=(nm, target_rank), p=[0.1, 0.8, 0.1])
            V_rand = np.random.choice([-1, 0, 1], size=(mp, target_rank), p=[0.1, 0.8, 0.1])
            
            # Construct A and b
            # A shape: (3600, pn * target_rank)
            # b shape: (3600,)
            
            # Optimization: Pre-compute the non-zero indices of U and V?
            # Or just build A directly.
            # 3600 * 1180 ~ 4 million entries. Building it in Python loop is slow.
            # Use vectorized construction or sparse matrix.
            
            # Let's use scipy sparse if available, or just numpy if we can vectorize.
            # Vectorized construction of A is tricky due to the specific mapping.
            # However, we can pre-compute the "pattern" of which (U_idx, V_idx) pairs contribute to which (W_idx).
            
            # Alternative: Just use numpy dense if it's fast enough for 5000 attempts?
            # 5000 * (3600 * 1180) operations is too much for pure Python loops.
            # We need to vectorize the construction of A.
            
            # Let's pre-compute the mapping arrays once.
            # For each equation index e (0..3599):
            #   i, j, jp, kp, I, K
            #   W_row = K * n + I
            #   Coeffs for c: U[i*m+j, c] * V[jp*p+kp, c]
            
            # We can store the indices (i, j, jp, kp, I, K) in arrays.
            eqs_i = []
            eqs_j = []
            eqs_jp = []
            eqs_kp = []
            eqs_I = []
            eqs_K = []
            eqs_b = []
            
            # This loop runs once
            for I in range(n):
                for K in range(p):
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for kp in range(p):
                                    eqs_i.append(i)
                                    eqs_j.append(j)
                                    eqs_jp.append(jp)
                                    eqs_kp.append(kp)
                                    eqs_I.append(I)
                                    eqs_K.append(K)
                                    if i == I and j == jp and kp == K:
                                        eqs_b.append(1.0)
                                    else:
                                        eqs_b.append(0.0)
                                        
            eqs_i = np.array(eqs_i)
            eqs_j = np.array(eqs_j)
            eqs_jp = np.array(eqs_jp)
            eqs_kp = np.array(eqs_kp)
            eqs_I = np.array(eqs_I)
            eqs_K = np.array(eqs_K)
            eqs_b = np.array(eqs_b)
            
            num_eqs = len(eqs_i)
            
            # In the loop, we build A
            # A[e, (K*n + I)*r + c] = U[i*m+j, c] * V[jp*p+kp, c]
            
            # This is a sparse matrix.
            # Rows: num_eqs
            # Cols: pn * r
            # Non-zeros: num_eqs * r (since U and V are dense in c? No, U and V are sparse in rows, but dense in c? 
            # Wait, U is (nm, r). U[i*m+j, c] is non-zero for some c.
            # If U is sparse, the number of non-zeros in A is reduced.
            
            # Let's use scipy.sparse
            try:
                import scipy.sparse as sp
                import scipy.sparse.linalg as spla
            except ImportError:
                # Fallback to numpy dense, but limit attempts
                pass
                
            if 'sp' in locals():
                rows = []
                cols = []
                data = []
                
                # For each equation e, and each c, if U_val and V_val are non-zero, add to A
                # This is still O(num_eqs * r) which is 3600 * 59 ~ 200k.
                # We can vectorize this.
                
                # Get indices of non-zeros in U and V
                # This is getting complex. Let's stick to numpy dense for simplicity and speed of implementation, 
                # but limit attempts to 500 per rank if it's slow.
                
                pass
            
            # Simple Numpy Dense Approach
            # A = np.zeros((num_eqs, pn * target_rank))
            # This is 3600 x 1180 ~ 4.2 MB.
            # Filling it:
            # We can compute U_part = U_rand[eqs_i * m + eqs_j, :]  # Shape (3600, r)
            # V_part = V_rand[eqs_jp * p + eqs_kp, :]  # Shape (3600, r)
            # Coeffs = U_part * V_part  # Shape (3600, r)
            # Now we need to scatter Coeffs into A.
            # A[e, (K[e]*n + I[e])*r + c] = Coeffs[e, c]
            
            # This can be done with advanced indexing.
            # Let's try it.
            
            U_part = U_rand[eqs_i * m + eqs_j, :]
            V_part = V_rand[eqs_jp * p + eqs_kp, :]
            Coeffs = U_part * V_part  # (3600, r)
            
            # Construct row and col indices for A
            # Row indices: repeat 0..3599 for each c
            row_idx = np.repeat(np.arange(num_eqs), target_rank)
            # Col indices: 
            # Base col = (K[e]*n + I[e]) * r
            # Add c (0..r-1)
            base_cols = (eqs_K * n + eqs_I) * target_rank  # Shape (3600,)
            col_offsets = np.tile(np.arange(target_rank), num_eqs)
            col_idx = base_cols[row_idx // target_rank] + col_idx
            
            # Wait, row_idx // target_rank gives e.
            # base_cols[e] is the start of the block for equation e.
            # col_idx = base_cols[e] + c.
            
            A = np.zeros((num_eqs, pn * target_rank))
            A[row_idx, col_idx] = Coeffs[row_idx // target_rank, row_idx % target_rank]
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, eqs_b, rcond=None)
            except Exception:
                continue
                
            if residuals is None or len(residuals) == 0 or residuals[0] > 1e-9:
                continue
                
            W_candidate = W_flat.reshape(pn, target_rank)
            
            # Check if W_candidate is close to half-integers
            W_2 = W_candidate * 2
            W_rounded_2 = np.round(W_2)
            if np.allclose(W_2, W_rounded_2, atol=1e-6):
                # Check U and V are integers (they are)
                # Verify exactly
                U_int = U_rand
                V_int = V_rand
                W_int = W_rounded_2 / 2.0
                
                # Verify: sum_c U[i*m+j, c] V[jp*p+kp, c] W[kp*n+i, c] == delta
                # We can do this in float with high precision or using fractions.
                # Given the small size, we can just check the residual of the original system with the rounded values.
                W_flat_rounded = (W_int * 2).astype(int).astype(float) / 2.0
                # Reconstruct A * W_flat_rounded
                # But A was built from U_rand, V_rand which are integers.
                # So we can just check if A @ W_flat_rounded == eqs_b
                # Note: A is float, but entries are integers.
                # W_flat_rounded entries are half-integers.
                # Product is half-integers.
                
                residual_check = np.linalg.norm(A @ W_flat_rounded - eqs_b)
                if residual_check < 1e-6:
                    # Found a valid solution
                    best_rank = target_rank
                    best_solution = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    break # Move to next (lower) rank
        
    return best_solution
# EVOLVE-BLOCK-END
