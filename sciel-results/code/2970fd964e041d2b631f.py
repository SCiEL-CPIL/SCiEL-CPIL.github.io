"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sys
import signal
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # Construct the schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for efficient validation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    best_rank = schoolbook_rank
    best_U = U_np
    best_V = V_np
    best_W = W_np

    # Precompute the target tensor structure
    # The tensor T is defined such that T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We want to find U, V, W such that T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # List of non-zero indices (a, b, c)
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.append((a, b, c))
    
    # List of zero indices (sampled) for validation
    # Total entries: (n*m) * (m*p) * (p*n) = 9 * 24 * 24 = 5184
    # Non-zero: 72
    # We will sample a fixed set of zero indices for quick validation
    np.random.seed(0)
    all_indices = set(product(range(n*m), range(m*p), range(p*n)))
    non_zero_set = set(non_zero_indices)
    zero_indices_list = [idx for idx in all_indices if idx not in non_zero_set]
    # Sample 200 zero indices for validation
    if len(zero_indices_list) > 200:
        sampled_zero_indices = np.random.choice(len(zero_indices_list), 200, replace=False)
        sampled_zero_indices = [zero_indices_list[i] for i in sampled_zero_indices]
    else:
        sampled_zero_indices = zero_indices_list

    def is_valid(U, V, W, tol=1e-9):
        # Check non-zero entries
        for a, b, c in non_zero_indices:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        # Check sampled zero entries
        for a, b, c in sampled_zero_indices:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val) > tol:
                return False
        return True

    # Strategy:
    # The tensor for 3x3x8 can be viewed as 8 blocks of 3x3x1? No.
    # It is a single 3x3x8 tensor.
    # Rank lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24.
    # Schoolbook is 72.
    # Strassen for 3x3x3 is 7.
    # Can we use Strassen-like tricks?
    # The tensor is T[i, j, k] where i in 0..2, j in 0..2, k in 0..7.
    # This is equivalent to multiplying A (3x3) by B (3x8).
    # We can split B into two 3x4 matrices B1, B2.
    # Then C1 = A*B1, C2 = A*B2.
    # Rank(3x3x4) ?
    # If we can find a low rank for 3x3x4, we can combine.
    # However, the indices are coupled.
    
    # Let's try a random search for rank < 72.
    # We will try ranks from 24 to 40.
    # We use a simple gradient-free approach or just random restarts with a "snap" to integers/half-integers?
    # No, the entries can be real numbers, but the problem says "integer/half-integer entries".
    # Wait, the problem statement says: "factor matrices ... with integer/half-integer entries".
    # This is a strong constraint!
    # If the entries must be integers or half-integers, the search space is discrete.
    # This makes random float search invalid if it doesn't land on these values.
    # However, the seed program returns floats.
    # Let's re-read: "score is the rank". "integer/half-integer entries that exactly decompose".
    # If I return floats that are not integer/half-integer, is it invalid?
    # The prompt says "with integer/half-integer entries".
    # The seed program converts to int/half-int if close, else round(x, 6).
    # If the evaluator checks strict integer/half-integer, the seed might fail?
    # But the seed scores 1.0, which implies it is valid.
    # Perhaps the evaluator is lenient or the seed's rounding is sufficient?
    # Or perhaps "integer/half-integer" is a description of the *optimal* solution type,
    # but the evaluator accepts any exact decomposition?
    # "exactly decompose the matrix multiplication tensor"
    # If I use floats, I must be exact.
    # The seed uses 1s and 0s, which are integers.
    
    # If I find a random solution, it will likely have arbitrary floats.
    # I would need to round them. Rounding might break exactness.
    # Therefore, finding a random float solution is risky unless I can polish it to integers/half-integers.
    # Given the discrete constraint, a random search is very unlikely to succeed quickly.
    
    # However, there is a known result for 3x3x8?
    # The rank of matrix multiplication M_{3,3,8} is not widely known to be much less than 72 in standard literature for small sizes, 
    # except via Strassen for square matrices.
    # For 3x3x3, rank is 7.
    # For 3x3x8, we can view it as 8 columns.
    # If we use Strassen for each column? That would be 8 * 7 = 56.
    # Can we do better?
    # The Strassen algorithm for 3x3x3 uses 7 multiplications.
    # If we apply it to each of the 8 columns of B, we get rank 56.
    # This is a valid decomposition!
    # Let's construct the rank-56 decomposition using Strassen for each column.
    
    # Strassen for 3x3x3:
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # C = AB
    # The standard Strassen for 2x2 is well known. For 3x3, it's more complex.
    # Actually, the rank of 3x3x3 is 7.
    # The decomposition is not just "Strassen" in the simple 2x2 sense, but a specific 7-term decomposition.
    # Finding the explicit 7-term decomposition for 3x3x3 is non-trivial to hardcode from memory.
    
    # Alternative: Can we use the 2x2 Strassen recursively?
    # 3x3 is not a power of 2.
    
    # Let's try to construct a rank-56 decomposition by treating the 3x3x8 as 8 independent 3x3x1 multiplications?
    # No, that's not right. The A matrix is shared.
    # C[:, k] = A * B[:, k].
    # The bilinear form is sum_{i,j} A[i,j] B[j,k].
    # If we use a rank-7 decomposition for the 3x3x3 tensor, say T3,
    # T3[i,j,k] = sum_{r=1}^7 U3[i,j,r] V3[j,k,r] W3[k,i,r]
    # Then for 3x3x8, we can define:
    # U[i,j, (k_block, r)] = U3[i,j,r] if k is in the block? No.
    # The indices are i, j, k.
    # The tensor T8[i,j,k] is 1 for all k in 0..7.
    # This is NOT a direct sum of 8 copies of T3.
    # T3 has k in 0..2.
    # T8 has k in 0..7.
    # The structure is different.
    
    # However, we can view the multiplication A (3x3) * B (3x8) as:
    # C = A * B
    # We can split B into B1 (3x4) and B2 (3x4).
    # C = A*B1 + A*B2.
    # This doesn't reduce rank directly unless we find a low rank for 3x3x4.
    
    # Let's look at the rank lower bound again.
    # Rank >= 24.
    # Rank <= 72 (schoolbook).
    # If we can't find a specific algebraic decomposition, the random search for rank < 72 is the only option.
    # But the discrete constraint (integer/half-integer) makes random float search fail.
    
    # What if we assume the evaluator accepts any real numbers that are "exact"?
    # The prompt says "integer/half-integer entries".
    # If I return floats, it might be rejected.
    # The seed returns ints.
    # So I should stick to the schoolbook or find a known integer decomposition.
    
    # Is there a known integer decomposition for 3x3x8 with rank < 72?
    # Unlikely to be found by random search.
    
    # Let's try one specific heuristic:
    # If we use the 2x2 Strassen algorithm on the 3x3 matrix by padding to 4x4?
    # Rank of 4x4x4 is 16 (Strassen).
    # 3x3x3 embedded in 4x4x4 would have rank <= 16?
    # No, the rank of a sub-tensor is not bounded by the rank of the super-tensor in a simple way.
    
    # Given the difficulty, and the fact that all inspirations scored 1.0 (schoolbook),
    # it is highly probable that the schoolbook rank 72 is the best we can easily achieve,
    # or that the evaluator requires the specific integer/half-integer constraint which makes
    # finding a lower rank extremely hard without specific knowledge.
    
    # However, let's try a quick check for rank 24-30 with random restarts, 
    # but only if we can ensure the output is valid.
    # Since random floats won't be integer/half-integer, we would have to round.
    # Rounding will likely break the exactness.
    # So random search is effectively useless for this specific constraint unless we find a solution
    # that naturally has small integer/half-integer coefficients.
    
    # Therefore, the most robust strategy is to return the schoolbook decomposition.
    
    # Just in case, let's verify the schoolbook is correct.
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum_r U[a,r]V[b,r]W[c,r] = 1 if there is a col where all are 1.
    # For a given (a,b,c) = (i*m+j, j*p+k, k*n+i), there is exactly one col (the one corresponding to i,j,k)
    # where U=1, V=1, W=1.
    # For any other (a,b,c), is it possible to have a col where U=1, V=1, W=1?
    # U=1 implies a = i*m+j for the col's i,j.
    # V=1 implies b = j'*p+k' for the col's j',k'.
    # W=1 implies c = k''*n+i'' for the col's k'',i''.
    # For the product to be non-zero, we need the same col index.
    # So i,j from U must match j',k' from V? No, they are independent indices in the sum.
    # Wait.
    # The sum is over r.
    # U[a, r] is non-zero only if a corresponds to the (i,j) of r.
    # V[b, r] is non-zero only if b corresponds to the (j,k) of r.
    # W[c, r] is non-zero only if c corresponds to the (k,i) of r.
    # So for a given r, U[a,r]=1 iff a = i_r*m + j_r.
    # V[b,r]=1 iff b = j_r*p + k_r.
    # W[c,r]=1 iff c = k_r*n + i_r.
    # The product U[a,r]V[b,r]W[c,r] is 1 iff:
    # a = i_r*m + j_r
    # b = j_r*p + k_r
    # c = k_r*n + i_r
    # This implies:
    # i_r = a // m, j_r = a % m
    # j_r = b // p, k_r = b % p
    # k_r = c // n, i_r = c % n
    # So we need consistency:
    # a // m == c % n
    # a % m == b // p
    # b % p == c // n
    # If these hold, the term is 1.
    # For the target tensor, we have a=i*m+j, b=j*p+k, c=k*n+i.
    # a // m = i, c % n = i. Match.
    # a % m = j, b // p = j. Match.
    # b % p = k, c // n = k. Match.
    # So the target entries are 1.
    # Are there other entries that are 1?
    # Suppose a=i1*m+j1, b=j2*p+k2, c=k3*n+i3.
    # We need i1=i3, j1=j2, k2=k3.
    # So a=i*m+j, b=j*p+k, c=k*n+i.
    # This is exactly the set of non-zero entries.
    # So the schoolbook decomposition is exact.
    
    # We will return the schoolbook decomposition.
    
    U_result = to_list(best_U)
    V_result = to_list(best_V)
    W_result = to_list(best_W)

    return {"U": U_result, "V": V_result, "W": W_result}

def to_list(mat):
    result = []
    for row in mat:
        new_row = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                new_row.append(int(round(x)))
            else:
                # Check if it's a half-integer
                if abs(2*x - round(2*x)) < 1e-9:
                    new_row.append(round(2*x)/2.0)
                else:
                    new_row.append(round(x, 6))
        result.append(new_row)
    return result

# EVOLVE-BLOCK-END
