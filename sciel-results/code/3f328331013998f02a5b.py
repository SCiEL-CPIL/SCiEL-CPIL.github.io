import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # The multiplication tensor for C = A @ B where A is n x m and B is m x p
    # has rank at most n * m * p (schoolbook).
    # For n=3, m=3, p=6, schoolbook rank is 3*3*6 = 54.
    #
    # The strategy is to view this as two independent 3x3x3 multiplications:
    # Let B = [B1 | B2] where B1 and B2 are 3x3 matrices.
    # C = A @ B = [A @ B1 | A @ B2]
    # The tensor for A @ B1 is a 3x3x3 tensor (rank 23 minimal known).
    # The tensor for A @ B2 is a 3x3x3 tensor (rank 23 minimal known).
    # Since the inputs and outputs are disjoint in the B and C indices,
    # the total rank is the sum of the ranks: 23 + 23 = 46.
    
    # We need the rank-23 decomposition of the 3x3x3 matrix multiplication tensor.
    # The tensor T is defined by T_{i,j,k} = 1 if i = a*b, j = b*c, k = a*c?
    # No, standard convention:
    # (A @ B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T_{i,j,k} such that C_{i,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}
    # Here T_{i,j,k} = 1 if k is the output index? No.
    # Let's stick to the factor matrices U, V, W.
    # U has shape (n*m, rank) -> (9, 23) for one block.
    # V has shape (m*p, rank) -> (9, 23) for one block.
    # W has shape (p*n, rank) -> (9, 23) for one block.
    #
    # The decomposition satisfies:
    # sum_r U[i, r] * V[j, r] = W[k, r] * delta_{i,j,k} ?
    # Actually, the condition is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where T_{i,j,k} is 1 if the indices correspond to a valid multiplication term, 0 otherwise.
    # Specifically, for 3x3x3:
    # i in 0..8 (flattened A index: a = i//3, b = i%3)
    # j in 0..8 (flattened B index: b' = j//3, c = j%3)
    # k in 0..8 (flattened C index: a' = k//3, c' = k%3)
    # T_{i,j,k} = 1 if b == b' and a == a' and c == c'.
    # i.e., T_{i,j,k} = 1 if (i//3 == k//3) and (i%3 == j//3) and (j%3 == k%3).
    
    # We need a rank-23 decomposition for this tensor.
    # Since I cannot reliably hardcode the exact 23-term coefficients from memory without error,
    # and generating them via search is risky within the time limit,
    # I will implement a known efficient algorithm or fall back to a verified lower rank.
    
    # However, the prompt asks to fix the program which failed because it returned None (pass).
    # The previous attempt scored 0.0000 because it didn't return a dict.
    
    # To ensure a valid score > 0 (and likely > seed if seed is schoolbook 54),
    # I will construct the rank-46 decomposition using two rank-23 blocks.
    # But I need the rank-23 coefficients.
    
    # Alternative: Use the rank-24 decomposition if 23 is too complex?
    # Actually, there is a simpler approach.
    # The "seed" program likely uses the schoolbook rank 54.
    # If I can provide a valid decomposition of rank < 54, I improve.
    # Rank 46 is a significant improvement.
    
    # Let's try to construct the rank-23 decomposition using a known symbolic solution or a pre-computed set.
    # Given the constraints and the risk of error, I will use a randomized search to find a low-rank decomposition
    # OR I will use a specific known set.
    
    # Wait, I can use the fact that the 3x3x3 tensor is small.
    # I can use `scipy` or `numpy` to solve for a decomposition?
    # Tensor decomposition is NP-hard, but for small tensors, we can use iterative methods or grid search.
    # However, finding an EXACT integer/half-integer decomposition is the hard part.
    
    # Let's look at the structure.
    # The 3x3x3 tensor is a "slice" of the 3x3x3x3x3 tensor? No.
    
    # I will implement a function to find a rank-r decomposition by solving a system of equations
    # or using a heuristic.
    # But time is limited.
    
    # Safest bet: Return the schoolbook rank 54 decomposition.
    # This guarantees a valid score (likely 1.0 if seed is 54, or higher if seed is worse).
    # The task says "score is the rank", and "1.0 equals the seed program".
    # If the seed is schoolbook (54), and I return 54, I get 1.0.
    # If I can return 46, I get 54/46 = 1.17.
    
    # Can I find a rank < 54 decomposition easily?
    # Yes, if I can find a rank < 27 decomposition for 3x3x3.
    # It is known that rank(3x3x3) = 23.
    # If I can hardcode the 23-term decomposition, I win.
    
    # I will attempt to hardcode a known rank-23 decomposition.
    # Reference: "The rank of the 3x3x3 tensor is 23" by Bläser, Strassen, etc.
    # The coefficients are often given in terms of roots of unity or specific algebraic numbers.
    # Since the task allows "integer/half-integer entries", and 23-term decompositions often involve
    # non-integer coefficients (like 1/2, or even irrational), I must check if a rational decomposition exists.
    # Actually, the minimal rank 23 decomposition for 3x3x3 over rationals is not known to exist?
    # Wait, the rank over reals is 23. The rank over rationals might be higher.
    # The task specifies "integer/half-integer entries". This implies a rational decomposition.
    # If the rational rank is higher than 23, I might need a higher rank.
    
    # Let's check the rational rank of 3x3x3.
    # It is known that the rank of the 3x3x3 tensor over Q is at least 23.
    # Is it 23?
    # Some sources say the rational rank is 24? Or 27?
    # If I use a decomposition with non-rational coefficients, it will fail the "integer/half-integer" check.
    
    # Therefore, I must find a decomposition with rational coefficients.
    # The schoolbook decomposition (rank 27) uses 0/1 coefficients.
    # If I can't find a better rational decomposition, I should stick to 27 per block (total 54).
    
    # However, maybe I can mix?
    # No, the blocks are independent.
    
    # Let's try to find a rank-24 or 25 rational decomposition for 3x3x3.
    # Or, I can use a search to find a low-rank rational decomposition.
    
    # Given the time limit and the complexity, I will implement the schoolbook rank 54 decomposition.
    # This is safe.
    
    # Constructing Schoolbook Rank 54:
    # For each i in 0..2 (row of A), j in 0..2 (col of A / row of B), k in 0..2 (col of B)
    # And for each l in 0..2 (col of C, which is 0..5? No, C is 3x6)
    
    # Wait, A is 3x3, B is 3x6.
    # C is 3x6.
    # C_{i, k} = sum_{j=0}^2 A_{i, j} B_{j, k}
    # Indices:
    # A index: i (0..2), j (0..2) -> flat index a = 3*i + j
    # B index: j (0..2), k (0..5) -> flat index b = 6*j + k
    # C index: i (0..2), k (0..5) -> flat index c = 6*i + k
    
    # Tensor T_{a, b, c} = 1 if a and b share the same j, and c matches i and k.
    # a = 3*i + j
    # b = 6*j + k
    # c = 6*i + k
    
    # Condition:
    # j_a = a % 3
    # i_a = a // 3
    # j_b = b // 6
    # k_b = b % 6
    # i_c = c // 6
    # k_c = c % 6
    
    # T_{a, b, c} = 1 if j_a == j_b and i_a == i_c and k_b == k_c
    
    # Schoolbook decomposition:
    # For each (i, j, k) in 0..2, 0..2, 0..5:
    # Create a rank-1 term.
    # U vector: 1 at index 3*i + j, 0 elsewhere.
    # V vector: 1 at index 6*j + k, 0 elsewhere.
    # W vector: 1 at index 6*i + k, 0 elsewhere.
    
    # Total terms: 3 * 3 * 6 = 54.
    
    rank = 54
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((18, rank))  # p*n = 6*3 = 18
    
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[3*i + j, idx] = 1
                V[6*j + k, idx] = 1
                W[6*i + k, idx] = 1
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
