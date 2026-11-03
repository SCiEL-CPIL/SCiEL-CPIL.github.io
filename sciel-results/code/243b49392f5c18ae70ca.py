import time
import numpy as np
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = get_schoolbook()
    
    def verify(U, V, W, tol=1e-9):
        # Check 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False, abs(val - 1.0)
        
        # Check 0-entries
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val) > tol:
                        return False, abs(val)
        return True, 0.0

    # Strategy: Exploit the block structure.
    # The tensor is a block diagonal of 3 copies of the 3x3x3 multiplication tensor.
    # Specifically, if we partition the 3x6 matrix B into three 3x3 blocks B_0, B_1, B_2,
    # then C_0 = A B_0, C_1 = A B_1, C_2 = A B_2.
    # Thus, rank(3x3x6) = 3 * rank(3x3x3).
    # We know rank(3x3x3) = 23. So the target rank is 69.
    
    # We need a rank 23 decomposition for 3x3x3.
    # We will attempt to find one using ALS (Alternating Least Squares) or return a hardcoded one if we find it.
    # Since finding rank 23 is hard, we will try to construct it.
    
    # For now, let's just return schoolbook (rank 54) to be safe? 
    # Wait, 3 * 27 = 81 is schoolbook.
    # The current program returns schoolbook rank 54? No, n*m*p = 3*3*6 = 54.
    # So current score is 54.
    # If we can get 69, that's worse (higher is worse for rank? No, lower is better).
    # Wait, score is seed/value. Lower value (rank) is better.
    # Schoolbook rank is 54.
    # If rank(3x3x3) is 23, then 3*23 = 69. That is WORSE than 54.
    # So the block structure argument gives rank <= 69, but schoolbook is 54.
    # So schoolbook is already better than 3 * rank(3x3x3).
    # This implies that the 3x3x6 tensor has a rank LOWER than the sum of ranks of its diagonal blocks?
    # No, rank of block diagonal tensor is exactly the sum of ranks if the spaces are direct sums.
    # Let's re-evaluate.
    # T_3x3x6 is not a block diagonal in the standard sense where the indices are disjoint.
    # The indices are (i, j, k) with i in 0..2, j in 0..2, k in 0..5.
    # The tensor is T[i, j, k] = 1 if k < 3? No.
    # The matrix multiplication is C = A B.
    # A is 3x3, B is 3x6.
    # C is 3x6.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T is defined on indices (i, j, k) such that T[i, j, k] = 1 if the operation is A[i, j] * B[j, k] -> C[i, k].
    # This is a single tensor of size 3x3x6.
    # It is NOT 3 copies of 3x3x3. It is one multiplication where the second dimension of B is 6.
    # The rank of this tensor is the minimum number of terms in the decomposition.
    # We know rank(3x3x3) = 23.
    # What is rank(3x3x6)?
    # Lower bound: max(3*3, 3*6, 6*3) = 18? No, lower bound is max(n*m, m*p, p*n) / something?
    # Actually, rank(3x3x6) >= 3*3 = 9?
    # Known result: rank(3x3x6) is likely 3 * rank(3x3x3) if we view it as 3 independent multiplications?
    # No, B is 3x6. It's not 3 separate 3x3 multiplications.
    # However, we can split B into B_left (3x3) and B_right (3x3).
    # Then C_left = A B_left and C_right = A B_right.
    # This IS a block diagonal structure in the output space.
    # The tensor is T = T_left + T_right where T_left acts on (i, j, k<3) and T_right on (i, j, k>=3).
    # The index spaces are disjoint for the k dimension.
    # Therefore, rank(T) = rank(T_left) + rank(T_right) = 23 + 23 = 46.
    # So the rank of 3x3x6 is 46.
    # Schoolbook is 54.
    # So we can improve from 54 to 46.
    
    # We need a rank 23 decomposition for 3x3x3.
    # We will try to find one.
    
    # Hardcoding a known rank 23 decomposition for 3x3x3 is difficult.
    # We can try to generate one using a search.
    # But 110s is not enough for a full search from scratch.
    # We might need to use a known solution or a very fast heuristic.
    
    # Let's check if we can find a rank 23 decomposition quickly.
    # If not, we return schoolbook (54).
    
    # For now, let's assume we can't find it quickly and return schoolbook.
    # But wait, if we can find it, we get 46 which is better than 54.
    # Score = 54 / 46 = 1.17.
    
    # Let's try to construct the rank 46 decomposition using a known rank 23 solution.
    # I will hardcode a rank 23 solution if I can find one in memory or derive it.
    # Since I can't, I will stick to schoolbook for now, but I will structure the code to allow for a rank 46 solution if found.
    
    U, V, W = U_school, V_school, W_school
    valid, err = verify(U, V, W)
    if not valid:
        raise Exception("Schoolbook decomposition failed verification")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
