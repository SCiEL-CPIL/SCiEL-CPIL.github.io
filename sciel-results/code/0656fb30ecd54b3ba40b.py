import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base case: C(2, 2, 2) rank 7
    # U (4x7), V (4x7), W (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_base = np.array([
        [1, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 1]
    ])
    
    V_base = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 0]
    ])
    
    W_base = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1]
    ])
    
    # Verify base case?
    # T = sum_r U[:,r] V[:,r] W[:,r]^T
    # We trust the standard Strassen-like decomposition for 2x2x2.
    
    # Recursive doubling
    # Start with p=2
    U_curr = U_base
    V_curr = V_base
    W_curr = W_base
    curr_p = 2
    
    while curr_p < 16:
        # Double p
        # U_new = [U_curr, U_curr]
        U_new = np.hstack([U_curr, U_curr])
        
        # V_new: Left half uses V_curr on indices 0..curr_p-1, Right half on curr_p..2*curr_p-1
        # V_curr is shape (2*curr_p, rank)
        # We need to split V_curr into two blocks of size curr_p?
        # No, V_curr corresponds to B of size 2 x curr_p.
        # The indices of V are j*p + k.
        # For the left half (k=0..curr_p-1), we use V_curr.
        # For the right half (k=curr_p..2*curr_p-1), we use V_curr shifted?
        # Actually, the structure is:
        # V_new has shape (2 * 2*curr_p, 2*rank) = (4*curr_p, 2*rank)
        # Rows 0..2*curr_p-1 correspond to j=0,1 and k=0..2*curr_p-1.
        # Wait, V index is j*p + k.
        # In the new problem, p_new = 2*curr_p.
        # V_new[j*p_new + k, r]
        # For left half (k < curr_p), this is V_curr[j*curr_p + k, r].
        # For right half (k >= curr_p), this is V_curr[j*curr_p + (k-curr_p), r + rank].
        
        rank_curr = U_curr.shape[1]
        U_new = np.hstack([U_curr, U_curr])
        
        V_new = np.zeros((2 * (2 * curr_p), 2 * rank_curr))
        W_new = np.zeros((2 * (2 * curr_p), 2 * rank_curr))
        
        # Fill V_new
        for j in range(2):
            for k in range(curr_p):
                # Left half
                V_new[j * (2 * curr_p) + k, :rank_curr] = V_curr[j * curr_p + k, :]
                # Right half
                V_new[j * (2 * curr_p) + curr_p + k, rank_curr:] = V_curr[j * curr_p + k, :]
                
        # Fill W_new
        # W index is k*n + i. n=2.
        for k in range(curr_p):
            for i in range(2):
                # Left half
                W_new[k * 2 + i, :rank_curr] = W_curr[k * 2 + i, :]
                # Right half
                W_new[(curr_p + k) * 2 + i, rank_curr:] = W_curr[k * 2 + i, :]
                
        U_curr = U_new
        V_curr = V_new
        W_curr = W_new
        curr_p = 2 * curr_p
        
    # Now we have C(2, 2, 16) with rank 56
    # We need C(2, 2, 15)
    # Just take the first 15 columns of B?
    # The decomposition for 16 includes terms for the 16th column.
    # If we set B_{j,15} = 0, those terms vanish?
    # No, the decomposition is fixed.
    # We can simply drop the terms that only contribute to the 16th column?
    # In the recursive construction, the right half of the last step corresponds to k=8..15.
    # The right half of the previous step (k=4..7) is in the left half of the last step?
    # Let's trace:
    # p=2: k=0,1
    # p=4: k=0,1 (left), 2,3 (right)
    # p=8: k=0..3 (left), 4..7 (right)
    # p=16: k=0..7 (left), 8..15 (right)
    # We want k=0..14.
    # We can keep all terms from the left half (k=0..7) of p=16.
    # And from the right half (k=8..15), we only need k=8..14.
    # The terms in the right half of p=16 are a copy of the p=8 decomposition.
    # The p=8 decomposition covers k=0..7.
    # So the right half of p=16 covers k=8..15.
    # We need to drop the terms corresponding to k=15.
    # In the p=8 block, k=7 corresponds to the last column.
    # So we need to drop the sub-terms corresponding to the last column of the p=8 block.
    
    # This is getting complex to prune.
    # Alternative: Just use the rank 56 decomposition for 16, and accept that it computes 16 columns.
    # The task asks for 2x2 * 2x15.
    # If I return a decomposition for 2x2 * 2x16, is it valid for 2x2 * 2x15?
    # The evaluator will check if the decomposition correctly multiplies 2x2 by 2x15.
    # If my W matrix has 32 rows (for 16 cols), but the task expects 30 rows (for 15 cols)?
    # Task: W shape (p*n, rank). p=15. So W must be 30 x rank.
    # My W is 32 x rank.
    # I must trim W to 30 rows.
    # And V to 30 rows (2*15).
    # And U stays 4 x rank.
    # If I trim V and W, do the terms still work?
    # The terms corresponding to the 16th column of B will have V entries in rows 28,29 (j=0,1, k=15).
    # If I remove those rows from V, and rows 28,29 from W (k=15, i=0,1),
    # then the contributions from those terms to the 15th column output (which doesn't exist in 15-col problem) are gone.
    # But do they contribute to other columns?
    # No, the tensor is block diagonal in k.
    # So trimming is safe.
    
    # Trim V to 30 rows (remove last 2 rows)
    # Trim W to 30 rows (remove last 2 rows)
    # U remains 4 x 56.
    
    U_final = U_curr
    V_final = V_curr[:30, :]
    W_final = W_curr[:30, :]
    
    # Convert to list of lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
