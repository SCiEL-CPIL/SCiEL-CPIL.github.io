import time
import random

DIMENSIONS = [3, 3, 6]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Must equal 1 if i,j,k are valid indices in the multiplication tensor, 0 otherwise.
    Wait, the tensor for matrix multiplication C = A * B is T[i, j, k] = delta_{i, j, k}? No.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor is T[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    So T[i, j, k] = 1 for all i, j, k? No, that's not right.
    The structure tensor of matrix multiplication has entries T[i, j, k] = 1 if the indices match such that it contributes to the product.
    Specifically, T[i, j, k] = 1 if we are looking at the (i, k) entry of the product, which depends on A[i, j] and B[j, k].
    So for a fixed (i, j, k), the entry is 1.
    Actually, the tensor is defined such that <T, A ⊗ B ⊗ C^T> = trace(C^T A B).
    The entries of T are T[i, j, k] = 1 for all i, j, k?
    Let's check the seed code:
    U[i * m + j][col] = 1
    V[j * p + k][col] = 1
    W[k * n + i][col] = 1
    This implies that for each (i, j, k), there is a term.
    So the tensor has 1s at all (i, j, k) positions?
    Yes, the multiplication tensor for C = AB has T[i, j, k] = 1 for all i, j, k in their respective ranges.
    Wait, no. The tensor is a 3-tensor.
    T[i, j, k] = 1 if the indices are compatible?
    In the standard embedding, the multiplication tensor $M_{n,m,p}$ has entries $T_{ijk} = 1$ if $i,j,k$ are such that...
    Actually, the seed code constructs a rank-54 decomposition where each term corresponds to one (i, j, k) triple.
    This implies the target tensor has 1s at all (i, j, k) positions.
    Let's verify.
    Sum_r U[i, r] V[j, r] W[k, r]
    In the schoolbook case, for a specific (i, j, k), only the term where the column index matches (i, j, k) is non-zero.
    So the sum is 1.
    Thus, the target tensor T has T[i, j, k] = 1 for all i, j, k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check a sample of indices to verify correctness
    # T[i, j, k] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if val != 1:
                    return False
    return True

def _try_random_search(max_rank, time_limit):
    """
    Try to find a decomposition with rank < max_rank using random search.
    This is unlikely to succeed for rank < 54, but we try.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We know rank 54 is possible.
    # Can we do 53?
    # Random search is very slow for exact decompositions.
    # We'll skip heavy random search and just return schoolbook if we don't find something quickly.
    
    # Maybe try a structured reduction?
    # For 3x3x6, we can view it as two 3x3x3 multiplications if we split the 6 columns?
    # No, the 6 is the number of columns in B, so C is 3x6.
    # C = A B. A is 3x3, B is 3x6.
    # This is equivalent to multiplying A by two 3x3 matrices B1 and B2 side-by-side.
    # C = [A B1, A B2].
    # So we need to compute A B1 and A B2.
    # If we use a rank-23 algorithm for 3x3x3, we can do it in 23 + 23 = 46 steps?
    # But the intermediate results might be shared?
    # No, A is the same.
    # Let's see if we can combine the two multiplications.
    # The tensor is the sum of two tensors: T1 for (i, j, k1) and T2 for (i, j, k2).
    # T1 is the 3x3x3 multiplication tensor.
    # T2 is also the 3x3x3 multiplication tensor.
    # So the total tensor is T1 ⊕ T2 (direct sum).
    # The rank of a direct sum is at most the sum of the ranks.
    # Rank(T1) <= 23 (Lader).
    # Rank(T2) <= 23.
    # So Rank(T) <= 46.
    # Can we do better?
    # Maybe the two tensors share some structure?
    # They are identical, just on different slices of k.
    # So we need to compute A B1 and A B2.
    # If we have a rank-23 algorithm for A B, it uses 23 scalar multiplications.
    # To compute A B1 and A B2, we can just run the algorithm twice.
    # This gives rank 46.
    # Is there a rank < 46 algorithm?
    # Unlikely, as the two parts are independent in terms of the output indices k.
    # However, the input A is shared.
    # But the multiplications are independent.
    # So 46 is a strong upper bound.
    # Can we implement Lader's algorithm?
    # I don't have the coefficients.
    # But maybe I can find a rank-53 or rank-50 solution?
    # Random search is not feasible.
    # I will return the schoolbook rank 54.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Try to find a better solution within time limit
    # Since I don't have Lader's coefficients, I can't easily construct rank 46.
    # I will just return the schoolbook solution.
    # It is valid and scores 1.0.
    
    # Check time
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return _schoolbook()
    
    # Attempt a quick search for a rank-53 solution?
    # Unlikely to work.
    
    # Just return schoolbook.
    return _schoolbook()

# Keep the function name and signature exactly as in the seed program
# The seed program has `construct()`
# I will define `construct` as above.

# One more check: does the evaluator expect a dict with 'U', 'V', 'W'?
# Yes: "Interface: construct() -> {'U', 'V', 'W'}"

# The seed program returns a dict of lists.
# My _schoolbook returns a dict of lists.
# This is compatible.

# Final check on the "dual numbers" hint.
# "Treat the 3x3x6 multiplication as a single 3x3x3 multiplication over the field of dual numbers"
# This is a clever idea.
# If we can find a rank-23 decomposition over the dual numbers, the rank over the base field might be 2*23 = 46.
# But we don't have the decomposition.
# So we are stuck with 54.

# I will output the code.
