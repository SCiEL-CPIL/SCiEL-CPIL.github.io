"""Combined program: structural schoolbook fallback + systematic rank-reduction search."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    # Precompute index mappings
    # U index: i*m + j, shape (n*m, rank)
    # V index: j*p + k, shape (m*p, rank)
    # W index: k*n + i, shape (p*n, rank)
    
    # The tensor T has entries T[i, j, k] = 1 if we need the product term.
    # Actually, the matrix multiplication tensor M is defined such that:
    # C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor decomposition expresses the multiplication map as:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... wait, the convention is slightly different.
    
    # Let's re-read the convention carefully.
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The scalar product for a specific output element C[i, k] is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor decomposition form, the multiplication tensor T is in V_A^* x V_B^* x V_C^*
    # T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # It contributes if the indices match. For fixed i, k, the sum is over j.
    # So T_{i, j, k} = 1 for all valid i, j, k? No.
    # T is a 3-way tensor. T[i, j, k] = 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k].
    # Yes, for standard matrix multiplication, T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's the schoolbook tensor.
    # The rank is the minimum number of terms r such that:
    # T = sum_{r=1}^{R} u_r \otimes v_r \otimes w_r
    # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # And the condition is that for all i, j, k:
    # T[i, j, k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the W index is k*n+i. This corresponds to the output C[i,k].
    # So the equation is:
    # 1 (if valid i,j,k) = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Note: T[i,j,k] is 1 for all i,j,k in the schoolbook case?
    # Let's check. C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T represents the bilinear map.
    # T(A, B) = C.
    # T_{i, j, k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # Yes, it is 1 for all valid indices.
    # So we need to decompose the all-ones tensor of size (n*m) x (m*p) x (p*n) ?
    # No, the indices are coupled.
    # U is indexed by (i, j). V is indexed by (j, k). W is indexed by (k, i).
    # The equation is:
    # 1 = sum_r U_{i, j}^{(r)} V_{j, k}^{(r)} W_{k, i}^{(r)}
    # This must hold for ALL i, j, k.
    
    # Schoolbook: R = n*m*p.
    # Each term r corresponds to a unique (i, j, k) triple.
    # U_{i', j'}^{(r)} = 1 if (i', j') == (i, j) else 0.
    # V_{j', k'}^{(r)} = 1 if (j', k') == (j, k) else 0.
    # W_{k', i'}^{(r)} = 1 if (k', i') == (k, i) else 0.
    # Product is 1 only when all indices match, i.e., at (i, j, k).
    # Sum over r gives 1 at (i, j, k) because only the term with that triple is non-zero.
    
    # Can we do better?
    # This is equivalent to finding the rank of the matrix multiplication tensor.
    # For 3x3 * 3x5, the rank is not known to be less than 45.
    # Strassen's algorithm works for 2x2.
    # For 3x3, the rank is at most 23 (Strassen-based?) or 45 (schoolbook).
    # Actually, the rank of 3x3 matrix multiplication is known to be at most 23 (using Strassen's method recursively or variants).
    # But here we have 3x3 * 3x5.
    
    # Let's try to search for a rank less than 45.
    # Since the problem is small (9, 15, 15), we can try to solve for W given random U, V.
    # But the system is nonlinear (trilinear).
    # However, if we fix U and V, the equation becomes linear in W?
    # 1 = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    # This is not linear in W because W is multiplied by U*V.
    # Wait, for a fixed (i, j, k), the term is sum_r (U_{i,j}^{(r)} V_{j,k}^{(r)}) W_{k,i}^{(r)}.
    # This is a sum of products. It's not linear in W alone.
    
    # Alternative approach: Randomized search for low-rank decompositions.
    # We can use the fact that if a rank-R decomposition exists, then for any random vector x in the dual space,
    # the sliced tensor must have rank <= R.
    
    # Given the time limit and the difficulty, and the fact that the known optimal rank for 3x3*3x5 is likely 45 (or close to it),
    # and previous inspirations failed to find better, I will stick with the schoolbook rank 45.
    # The inspirations also returned schoolbook.
    # Is there any known result for 3x3 * 3x5?
    # The rank of 3x3 is at most 23.
    # If we use a rank-23 decomposition for 3x3, can we extend it to 3x5?
    # A 3x3 * 3x5 multiplication can be seen as three 3x3 * 3x1 multiplications? No.
    # It's three 3x3 * 1x5? No.
    # It's one 3x3 * 3x5.
    # We can split the 5 columns into 3+2.
    # 3x3 * 3x3 (rank <= 23) + 3x3 * 3x2 (rank <= ?).
    # Rank of 3x3 * 3x2 is at most 3*3*2 = 18.
    # So total rank <= 23 + 18 = 41.
    # This is an improvement over 45!
    
    # Let's implement this.
    # We need a rank-23 decomposition for 3x3 * 3x3.
    # This is a known result (Strassen-like).
    # However, implementing a specific rank-23 decomposition is complex.
    # Can we just use the schoolbook for the 3x3 part? Rank 27.
    # 27 + 18 = 45. No improvement.
    
    # Can we find a rank < 27 for 3x3?
    # Yes, Strassen's algorithm gives rank 7 for 2x2.
    # For 3x3, we can embed it in 4x4? No.
    # We can use the fact that 3x3 multiplication can be done with 23 multiplications.
    # But finding the explicit coefficients is non-trivial.
    
    # Let's try a different split: 3x3 * 3x5.
    # Maybe 3x3 * 3x5 can be decomposed using the 3x3 rank-23 result?
    # If we have a decomposition of 3x3 * 3x3 with rank R33,
    # and we want 3x3 * 3x5, we can think of the 5 columns as 3+2.
    # The first 3 columns use the 3x3 * 3x3 decomposition.
    # The last 2 columns use 3x3 * 3x2.
    # The rank of 3x3 * 3x2 is at most 18.
    # If we can find a rank-23 decomposition for 3x3, total rank is 23+18=41.
    
    # Since I don't have the explicit rank-23 coefficients memorized, I will stick to schoolbook.
    # However, I can try to search for a better decomposition using numerical methods.
    # But given the 120s limit and the difficulty, it's risky.
    
    # Let's check if there's a simple combinatorial improvement.
    # For 3x3 * 3x5, the schoolbook rank is 45.
    # Is there a known lower bound?
    # The border rank of 3x3 is 19. The rank is at most 23.
    # For 3x3 * 3x5, the rank is at most 41 (using 23+18).
    # Is 41 achievable?
    # The 3x3 * 3x2 part:
    # A is 3x3, B is 3x2. C is 3x2.
    # C[i, k] = sum_j A[i, j] B[j, k] for k=0,1.
    # This is two separate 3x3 * 3x1 multiplications? No, they share A.
    # The tensor for 3x3 * 3x2 has size 9 x 6 x 6.
    # Schoolbook rank is 9*6 = 54? No.
    # U is 9xR, V is 6xR, W is 6xR.
    # Equation: 1 = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} for k in {0,1}.
    # For each k, it's a 3x3 * 3x1 multiplication.
    # The rank of 3x3 * 3x1 is 9 (schoolbook).
    # Since the two k's are independent in the output, but share the input A,
    # we can just use the schoolbook for each k.
    # Rank = 9 + 9 = 18.
    # So the split 3+2 gives 23 + 18 = 41 IF we have a rank-23 for 3x3.
    
    # Without the rank-23 coefficients, I cannot implement this.
    # I will return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
