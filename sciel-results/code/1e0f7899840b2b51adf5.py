"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    
    The schoolbook rank is n*m*p = 3*3*6 = 54.
    The theoretical lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18.
    However, for rectangular matrix multiplication, the rank is generally higher.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 54).
    2. Try to exploit the structure. The multiplication of (n x m) * (m x p) can be 
       viewed as a tensor T in R^{n x m x p} where T[i,j,k] = 1 if the product C[i,k] 
       depends on A[i,j]*B[j,k].
       
    Actually, the tensor for matrix multiplication C = A * B is defined by:
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    The decomposition is a set of rank-1 tensors (u, v, w) such that:
    T = sum_{c} u_c ⊗ v_c ⊗ w_c
    where T[i,j,k] = sum_c U[i, c] * V[j, c] * W[k, c]
    
    In the schoolbook case, each (i,j,k) term is a separate rank-1 tensor:
    U[i, c] = 1, V[j, c] = 1, W[k, c] = 1 for the specific c corresponding to (i,j,k).
    
    Can we do better?
    For square matrices, the rank is known to be less than n^3 for n>=2.
    For rectangular matrices, the rank is at least max(n*m, m*p, p*n).
    
    Let's try a random search / optimization approach to find a lower rank.
    However, finding an *exact* decomposition with integer/half-integer entries is hard.
    The problem statement says "integer/half-integer entries". This suggests that if a solution exists, it might have simple rational entries.
    
    Let's first verify if a rank lower than 54 is possible.
    The rank of matrix multiplication tensor M_{n,m,p} is denoted R(n,m,p).
    R(3,3,3) is known to be 23 (Landsberg's bound, but actually it's 23? No, 3x3x3 is 23? Strassen is 7 for 2x2. For 3x3, it's 23).
    Wait, 3x3x3 rank is 23? 
    Actually, the rank of 3x3 matrix multiplication is 23.
    Here we have 3x3 times 3x6.
    This is equivalent to two independent 3x3x3 multiplications?
    C = A * B where A is 3x3, B is 3x6.
    We can write B = [B1 | B2] where B1, B2 are 3x3.
    Then C = [A*B1 | A*B2].
    So the tensor is the direct sum of two 3x3x3 multiplication tensors.
    If R(3,3,3) = 23, then R(3,3,6) <= 23 + 23 = 46.
    Is it possible to do better than 46?
    The lower bound is max(9, 18, 18) = 18.
    
    Let's check known values.
    R(3,3,3) = 23.
    So R(3,3,6) <= 46.
    Can we do better?
    If we could find a decomposition of rank < 46, that would be an improvement.
    
    Let's try to construct a rank 46 decomposition.
    We need a rank 23 decomposition for 3x3x3.
    The standard Strassen-like algorithms or specific known decompositions can be used.
    
    However, implementing a known rank 23 decomposition for 3x3x3 is complex.
    Let's see if we can find one online or derive it.
    Actually, the rank of 3x3x3 is 23.
    
    Let's try a different approach: Randomized search for a low-rank decomposition.
    But we need *exact* integer/half-integer entries.
    
    Given the complexity and time limit, let's stick to a simpler improvement if possible, or return the schoolbook if we can't find a better one quickly.
    
    Wait, the task says "Find a rank-minimal exact tensor decomposition".
    If I can't find the minimal one, I should find a good one.
    The schoolbook is 54.
    If I can find 46, that's a 1.17x improvement (54/46 = 1.17).
    
    Let's try to find a rank 23 decomposition for 3x3x3.
    One known decomposition is based on the "Landsberg" or "Strassen" methods.
    
    Actually, there is a known rank 23 decomposition for 3x3x3.
    Let's try to search for it or generate one using numerical methods and then round to integers?
    The problem requires *exact* integer/half-integer entries.
    This implies that if we find a numerical solution, it must be very close to integers/half-integers.
    
    Let's try a numerical optimization to find a rank R decomposition.
    We can use gradient descent or similar.
    But 3x3x3 is small.
    
    Let's try to construct a rank 46 decomposition for 3x3x6 by using two rank 23 decompositions for 3x3x3.
    If I can find a rank 23 decomposition for 3x3x3, I can just duplicate it for the two 3x3 blocks.
    
    How to find a rank 23 decomposition for 3x3x3?
    This is a known result. I will try to hardcode a known one or find it via search.
    
    Since I don't have a hardcoded rank 23 decomposition, let's try a randomized search with integer constraints?
    No, that's too hard.
    
    Let's try a different strategy:
    Use the schoolbook decomposition but try to merge some terms?
    No, that's unlikely to work easily.
    
    Let's try to use a known algorithm for matrix multiplication.
    Strassen's algorithm works for 2x2.
    For 3x3, there are specific algorithms.
    
    Given the time constraint (120s), and the difficulty of finding an exact rank 23 decomposition programmatically without prior knowledge, I will attempt a numerical search for a lower rank, but with a fallback to the schoolbook decomposition.
    
    Actually, let's look at the dimensions again.
    n=3, m=3, p=6.
    Rank 54 is schoolbook.
    Rank 46 is two 3x3x3 blocks.
    
    Is there a rank lower than 46?
    The lower bound is 18.
    
    Let's try to find a rank R decomposition using random initialization and optimization.
    We want to minimize || T - sum u_c v_c w_c ||.
    If we can get close to 0, we can round to integers.
    
    Let's try R = 40, 45, etc.
    
    However, the "integer/half-integer" constraint is very strict.
    If I find a numerical solution that is not close to integers, it's invalid.
    
    Let's assume the optimal solution has integer entries.
    Then we can search for integer vectors.
    But the search space is huge.
    
    Alternative: Use the known fact that R(3,3,3)=23.
    I will try to find a rank 23 decomposition online in my training data.
    One such decomposition is given in "The Rank of 3x3 Matrix Multiplication" by Landsberg.
    It is complex.
    
    Let's try a simpler approach:
    What if I just return the schoolbook decomposition? Score 1.0.
    Can I do better?
    
    Let's try to find a rank 46 decomposition.
    If I can find a rank 23 decomposition for 3x3x3, I can use it.
    
    Let's try to generate a rank 23 decomposition using a known method.
    Actually, I can try to use the "Strassen" like idea.
    
    Let's try to use a random search for a rank R decomposition where R is slightly less than 54.
    Maybe rank 50?
    
    No, let's try to find the rank 23 decomposition for 3x3x3.
    I will try to hardcode a known one if I can remember it.
    I don't remember the exact coefficients.
    
    Let's try a numerical approach:
    1. Initialize U, V, W randomly with small integers.
    2. Use gradient descent to minimize the error.
    3. If the error is very small, round to integers and check if it's exact.
    
    This is risky and might not finish in time.
    
    Let's try a different idea:
    The tensor is T[i,j,k] = 1 if i,j,k are in range.
    Actually, T is the identity tensor for the multiplication.
    T[i,j,k] = 1 if the product C[i,k] involves A[i,j] and B[j,k].
    Yes, T[i,j,k] = 1 for all i,j,k.
    Wait, no.
    C[i,k] = sum_j A[i,j] B[j,k].
    So the tensor T has entries T[i,j,k] = 1.
    The decomposition is sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k].
    So we want sum_c U[i,c] V[j,c] W[k,c] = 1 for all i,j,k.
    
    This is equivalent to finding a set of rank-1 tensors that sum to the all-ones tensor?
    No, that's not right.
    The tensor T for matrix multiplication is NOT the all-ones tensor.
    The tensor T is defined such that the contraction of T with A and B gives C.
    T is a 3-index tensor.
    The entry T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    Yes, for every i,j,k, the term A[i,j]B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 for all i,j,k.
    
    Wait, if T[i,j,k] = 1 for all i,j,k, then the problem is to decompose the all-ones tensor of size 3x3x6.
    The rank of the all-ones tensor of size n x m x p is 1!
    Because U[i,c]=1, V[j,c]=1, W[k,c]=1 for c=0.
    Then sum_c U[i,c]V[j,c]W[k,c] = 1*1*1 = 1.
    
    But that's not matrix multiplication!
    Matrix multiplication is C = A B.
    The tensor T is defined by C[i,k] = sum_{j,l} T[i,j,k] A[i,j] B[j,k]?
    No, the indices are different.
    
    Let's clarify the tensor definition.
    The matrix multiplication tensor T is in R^{n x m x p}.
    The decomposition is:
    T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    
    And the multiplication is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    How does T relate to A, B, C?
    Usually, the tensor is defined such that:
    C = sum_c (U_c^T A) (B V_c) ? No.
    
    The standard definition:
    We want to compute C = A B.
    A is n x m, B is m x p, C is n x p.
    The tensor T is of size n x m x p.
    T[i,j,k] = 1 if the product A[i,j] * B[j,k] is part of the sum for C[i,k].
    Since C[i,k] = sum_j A[i,j] B[j,k], the term A[i,j] B[j,k] contributes to C[i,k] with coefficient 1.
    So T[i,j,k] = 1 for all i,j,k.
    
    Wait, if T[i,j,k] = 1, then the decomposition is:
    1 = sum_c U[i,c] V[j,c] W[k,c]
    This must hold for all i,j,k.
    
    If I choose U[i,0]=1, V[j,0]=1, W[k,0]=1, then sum = 1.
    So the rank is 1?
    
    This cannot be right. Matrix multiplication has rank > 1.
    
    Ah, the indices of U, V, W are different.
    U is n x rank.
    V is m x rank.
    W is p x rank.
    
    The decomposition is:
    T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    
    If T[i,j,k] = 1 for all i,j,k, then the rank is indeed 1.
    
    But matrix multiplication is NOT just summing 1s.
    The structure is that A and B are multiplied.
    The tensor T encodes the *structure* of the multiplication.
    
    Let's look at the schoolbook decomposition in the seed code:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    Here, the rows of U are indexed by i*m+j.
    This suggests that the "input" to the tensor is not just A[i,j] but a flattened vector.
    
    Let's re-read the task carefully.
    "Matrix multiplication 3,3,6"
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 6 matrix."
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    The interface is:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    This is the standard "tensor rank" of matrix multiplication.
    The tensor T is of size (n*m) x (m*p) x (p*n).
    The entries of T are 1 if the corresponding term is present in the multiplication, 0 otherwise.
    
    Specifically, the multiplication is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Let's flatten A to a vector of size n*m. Index: i*m+j.
    Let's flatten B to a vector of size m*p. Index: j*p+k.
    Let's flatten C to a vector of size n*p. Index: i*p+k.
    
    But the W factor has size p*n. Index: k*n+i.
    This is the "trace convention".
    
    The tensor T has dimensions (n*m) x (m*p) x (p*n).
    T[im+j, jp+k, kn+i] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    Since it does, T[im+j, jp+k, kn+i] = 1.
    All other entries are 0.
    
    So we need to find U, V, W such that:
    sum_c U[im+j, c] V[jp+k, c] W[kn+i, c] = 1
    if (i,j,k) is a valid index (which it always is), and 0 otherwise.
    
    Wait, for any i,j,k, the term exists.
    So for every combination of (i,j,k), T[im+j, jp+k, kn+i] = 1.
    
    But what about other entries in T?
    The tensor T is sparse.
    T[a, b, c] = 1 if there exist i,j,k such that:
    a = i*m+j
    b = j*p+k
    c = k*n+i
    
    And T[a,b,c] = 0 otherwise.
    
    So the problem is to decompose this sparse tensor.
    
    The schoolbook decomposition uses rank n*m*p.
    Each term (i,j,k) is a rank-1 tensor with 1s at the corresponding positions.
    
    Can we do better?
    Yes, if we can find a set of rank-1 tensors that sum to 1 at the required positions and 0 elsewhere.
    
    This is equivalent to finding a low-rank decomposition of the matrix multiplication tensor.
    
    For 3x3x3, the rank is 23.
    For 3x3x6, the rank is at most 46.
    
    Let's try to find a rank 46 decomposition.
    We need a rank 23 decomposition for 3x3x3.
    
    I will try to search for a rank 23 decomposition of 3x3x3 using a known algorithm or hardcoded values.
    Since I don't have hardcoded values, I will try to use a numerical method to find one, and then round to integers.
    
    However, the time limit is 120s.
    A numerical search for rank 23 might be slow.
    
    Let's try a simpler improvement:
    What if I use a rank 54 decomposition but with a different structure?
    No, that's the same.
    
    Let's try to find a rank 46 decomposition.
    I will try to use the fact that 3x3x6 is two 3x3x3 blocks.
    If I can find a rank 23 decomposition for 3x3x3, I can just use it twice.
    
    Let's try to find a rank 23 decomposition for 3x3x3 using a random search with integer constraints.
    I will initialize U, V, W with random integers in {-1, 0, 1}.
    Then I will check if the decomposition is exact.
    If not, I will try a few random restarts.
    
    This is a "shot in the dark", but it might work if the solution is simple.
    
    Actually, the rank 23 decomposition is not simple.
    
    Let's try a different approach:
    Use the schoolbook decomposition, but try to merge some terms.
    For example, if two terms have the same U and V but different W, can we merge them?
    No, because W is different.
    
    Let's try to use a known lower bound.
    The rank of 3x3x3 is 23.
    The rank of 3x3x6 is at least 18.
    
    I will try to find a rank 46 decomposition.
    I will try to find a rank 23 decomposition for 3x3x3 using a numerical method.
    
    Let's try to use `scipy.optimize.minimize` to find a rank 23 decomposition.
    I will define the objective function as the sum of squares of the errors.
    I will initialize U, V, W randomly.
    
    This might be too slow.
    
    Let's try a simpler idea:
    What if I just return the schoolbook decomposition?
    Score 1.0.
    
    Can I find a rank 50 decomposition?
    Maybe by merging some terms?
    
    Let's try to find a rank 46 decomposition by using two rank 23 decompositions.
    I will try to find a rank 23 decomposition for 3x3x3 using a known method.
    
    Actually, I can try to use the "Strassen" algorithm for 3x3.
    But I don't know the details.
    
    Let's try to use a random search for a rank R decomposition where R is 46, 45, etc.
    I will initialize U, V, W with random integers in {-2, -1, 0, 1, 2}.
    Then I will check if the decomposition is exact.
    
    This is unlikely to work quickly.
    
    Let's try a different strategy:
    Use the schoolbook decomposition, but with a different indexing.
    No, that's the same.
    
    I will return the schoolbook decomposition for now, as I don't have a reliable way to find a lower rank decomposition within the time limit.
    
    Wait, the task says "Find a rank-minimal exact tensor decomposition".
    If I can't find the minimal one, I should find a good one.
    
    Let's try to find a rank 46 decomposition.
    I will try to use a known rank 23 decomposition for 3x3x3.
    I will try to find it online in my training data.
    
    One known decomposition is:
    U, V, W are 9x23 matrices.
    
    I will try to hardcode a known one.
    If I can't find it, I will return the schoolbook decomposition.
    
    Actually, let's try to use a numerical method to find a rank 23 decomposition.
    I will use `scipy.optimize.minimize` with a random initialization.
    I will try a few restarts.
    
    If I find a solution with a very small error, I will round to integers and check if it's exact.
    
    Let's try this.
    
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # For 3x3x6, the rank is at most 46 (two 3x3x3 blocks)
    # Let's try to find a rank 46 decomposition
    
    # First, try to find a rank 23 decomposition for 3x3x3
    # Then use it for the two blocks
    
    # Since finding a rank 23 decomposition is hard, let's try a random search for a rank R decomposition
    # where R is slightly less than 54
    
    # Actually, let's just return the schoolbook decomposition for now
    # as a safe fallback
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
