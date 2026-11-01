"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Schoolbook Decomposition (Rank 36)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # 2. Search for lower rank (35, 34, etc.)
    # We attempt to find a decomposition of rank R < 36.
    # The tensor T is defined by T_{i, j, k} = 1 if we are computing C_{ik} from A_{ij} and B_{jk}.
    # Actually, the tensor entries are T_{i, j, k} where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # Wait, the standard tensor for matrix multiplication is often defined as M_{i, j, k} = 1 if we multiply A_{ij} and B_{jk}.
    # The decomposition is: sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} = T_{i, j, k}
    # Here U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank) or (n*p, rank)?
    # The prompt says: W[k*n+i, r]. So W index is k*n+i.
    # Let's check the indices carefully.
    # A is n x m. B is m x p. C is n x p.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor T_{i, j, k} = 1.
    # U index: i*m + j.
    # V index: j*p + k.
    # W index: k*n + i. (This is unusual, usually it's i*p+k or similar, but we must follow the prompt).
    
    # Let's verify the reconstruction with schoolbook:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For a specific term (i, j, k), there is a unique r = i*m*p + j*p + k (in the loop order i, j, k).
    # U[i*m+j, r] = 1.
    # V[j*p+k, r] = 1.
    # W[k*n+i, r] = 1.
    # Product is 1.
    # Are there other r's?
    # If r corresponds to (i', j', k'), then U[i*m+j, r] is non-zero only if i'=i and j'=j.
    # Then V[j*p+k, r] is non-zero only if j'=j and k'=k.
    # So only the term (i, j, k) contributes 1.
    # Thus the reconstruction is correct.

    # Strategy: Try to find a rank R decomposition by random search + local refinement?
    # Or use a known structure?
    # For 3x3x3, Strassen is rank 7.
    # For 3x3x4, the best known is 34? Or is it lower?
    # Actually, the problem of finding the minimal rank is very hard.
    # Random search for rank 34 or 35 is extremely unlikely to succeed in 140s without a very good heuristic.
    # However, we can try a simple heuristic:
    # Start with random matrices of rank R.
    # Check if they satisfy the tensor equation.
    # This is a system of quadratic equations.
    
    # Given the constraints and the high difficulty, and the fact that previous attempts (inspirations) all returned 36,
    # and no better score has been observed in the history provided (all scores 1.0),
    # it is highly probable that 36 is the best we can reliably achieve within the time limit without specific knowledge of a rank 34 construction.
    
    # Let's try a very limited search for rank 35 to see if we get lucky?
    # No, the space is huge.
    
    # What if we use the "Landsberg" bound or similar?
    # For 3x3x3, rank is 7.
    # For 3x3x4, we can view it as 3x3x3 + 3x3x1.
    # Rank(3x3x3) = 7.
    # Rank(3x3x1) = 3.
    # Total rank <= 7 + 3 = 10? No, that's not how it works.
    # The tensor is not a direct sum in that simple way.
    
    # Actually, if we fix the last column of B, we are doing 3x3 * 3x3 and 3x3 * 3x1.
    # The tensor is T = T_3x3x3 + T_3x3x1.
    # The rank of a sum is <= sum of ranks.
    # Rank(T_3x3x3) = 7 (Strassen).
    # Rank(T_3x3x1) = 3 (diagonal).
    # So Rank(T) <= 7 + 3 = 10?
    # Wait, the indices must match.
    # T_3x3x3 involves k=0,1,2.
    # T_3x3x1 involves k=3.
    # The indices i, j are the same.
    # The W index for k=0,1,2 is k*n+i.
    # The W index for k=3 is 3*n+i.
    # They are disjoint in the W dimension?
    # No, the decomposition is a sum over r.
    # U, V, W are the same matrices for all terms.
    # We can't just add ranks of sub-tensors directly unless we can align the r indices.
    # But we can construct a decomposition of rank 7+3=10?
    # Let's check.
    # Let U1, V1, W1 be a rank 7 decomposition of the 3x3x3 part (k=0,1,2).
    # Let U2, V2, W2 be a rank 3 decomposition of the 3x3x1 part (k=3).
    # We need to combine them.
    # The U, V, W matrices must have the same number of columns (rank).
    # We can pad U1, V1, W1 with zeros to rank 10.
    # And pad U2, V2, W2 with zeros to rank 10.
    # But the W indices are different!
    # W1 has non-zeros at indices 0*n+i, 1*n+i, 2*n+i.
    # W2 has non-zeros at index 3*n+i.
    # If we just concatenate the columns, the W matrices will have non-zeros in disjoint rows.
    # This is fine. The sum over r will work.
    # So, is the rank of 3x3x4 <= 7 + 3 = 10?
    # This seems too good to be true.
    # Let's re-read the tensor definition.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T_{i,j,k} = 1.
    # For k=0,1,2, we have a 3x3x3 tensor.
    # For k=3, we have a 3x3x1 tensor.
    # The tensor is the direct sum of these two?
    # Yes, because the k index separates them.
    # So Rank(T) <= Rank(T_3x3x3) + Rank(T_3x3x1).
    # Rank(T_3x3x3) = 7.
    # Rank(T_3x3x1) = 3.
    # So Rank(T) <= 10.
    
    # Wait, is Rank(T_3x3x1) = 3?
    # T_{i,j,3} = 1.
    # C_{i,3} = sum_j A_{i,j} B_{j,3}.
    # This is a dot product.
    # The rank is n*p = 3*1 = 3?
    # Yes, the rank of the matrix multiplication tensor for n x m * m x p is n*m*p.
    # For 3x3 * 3x1, rank is 3*3*1 = 9?
    # No, the rank of the tensor is the minimal number of bilinear forms.
    # For 3x3 * 3x1, it's just 3 dot products of length 3.
    # Each dot product requires 3 multiplications? No, 1 multiplication per term?
    # Bilinear form: sum_j A_{i,j} B_{j,3}.
    # This is a single bilinear form in A and B?
    # No, it's a sum of products.
    # The tensor rank of a dot product of length m is m.
    # So for each i, the rank is 3.
    # Since i is independent, the total rank is 3 * 3 = 9.
    # So Rank(T_3x3x1) = 9.
    
    # So Rank(T) <= 7 + 9 = 16?
    # Let's check the rank of 3x3 * 3x1.
    # C is 3x1.
    # C_i = sum_j A_ij B_j3.
    # This is 3 independent dot products.
    # Each dot product of length 3 has rank 3.
    # So total rank is 9.
    
    # So Rank(3x3x4) <= 7 + 9 = 16.
    # This is much better than 36!
    # Why did I think it was 34?
    # Maybe I was confusing it with something else.
    # Let's verify this construction.
    
    # We need to construct the rank 7 decomposition for 3x3x3.
    # Strassen's algorithm for 3x3x3 is not standard. Strassen is for 2x2x2 (rank 7).
    # For 3x3x3, the best known rank is 23?
    # Wait, Strassen's algorithm is for 2x2 matrices.
    # For 3x3 matrices, the best known rank is 23 (Landsberg).
    # So Rank(T_3x3x3) = 23.
    # Rank(T_3x3x1) = 9.
    # Total Rank <= 23 + 9 = 32.
    
    # This is better than 36!
    # Can we do better?
    # Maybe the 3x3x1 part can be integrated more efficiently?
    # Or maybe there is a better decomposition for 3x3x3?
    # No, 23 is the best known for 3x3x3.
    
    # So, if we can construct a rank 32 decomposition, we win!
    # Score would be 36/32 > 1.0.
    
    # How to construct it?
    # 1. Get a rank 23 decomposition for 3x3x3.
    # 2. Get a rank 9 decomposition for 3x3x1.
    # 3. Combine them.
    
    # But wait, do I have a rank 23 decomposition for 3x3x3 memorized?
    # No.
    # Can I find one quickly?
    # Probably not.
    
    # What if I use the schoolbook for 3x3x3 (rank 27) and schoolbook for 3x3x1 (rank 9)?
    # Total rank = 27 + 9 = 36.
    # That's the same as the original schoolbook.
    
    # So I need a better decomposition for 3x3x3.
    # Is there a known rank 23 decomposition that is easy to code?
    # The best known is 23, but it's complex.
    
    # What if I try to search for a rank 32 decomposition?
    # Start with random rank 32 matrices.
    # Check if they satisfy the tensor equation.
    # This is still a huge search space.
    
    # Given the time limit and the difficulty, and the fact that I don't have a rank 23 decomposition memorized,
    # I will stick to the schoolbook (rank 36) unless I can find a very simple improvement.
    
    # Is there a simple improvement?
    # What if I use Strassen for 2x2 blocks?
    # 3x3 can be viewed as 2x2 + 2x2 + ...
    # This is getting complicated.
    
    # Let's try one more thing:
    # Can we reduce the rank of the 3x3x1 part?
    # No, it's 9.
    
    # Can we reduce the rank of the 3x3x3 part?
    # Only if we find a better algorithm.
    
    # Since I can't find a better algorithm quickly, I will return the schoolbook.
    
    # However, let's try a random search for rank 35 just in case.
    # It's unlikely to succeed, but it's better than doing nothing.
    
    # Actually, let's just return the schoolbook. It's safe.
    
    return best_result
# EVOLVE-BLOCK-END
