"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _get_schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verify that U, V, W form a valid decomposition for 2x3 * 3x7.
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 if (i,j,k) is a valid index triple, else 0.
    Actually, for matrix multiplication C = A * B where A is 2x3 and B is 3x7,
    C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor is T[i,j,k] = delta (where T[i,j,k] = 1 means C[i,k] gets A[i,j]*B[j,k]).
    So T[i,j,k] = 1 for all valid i,j,k, and the decomposition should satisfy:
    For each (i,k): sum_j T[i,j,k] * U[i,j] * V[j,k] ... wait, let me re-read the convention.
    
    The convention is W[k*n+i, r]. So the output index is (k, i) mapped to k*n+i.
    The tensor decomposition requires:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    and the "non-tensor" components should be 0, but actually the tensor only has the 
    multiplication structure. Let me think again.
    
    The matrix multiplication tensor M is defined by M[i, j, k] = 1 for all valid indices,
    and the multiplication is: C[i,k] = sum_j A[i,j] B[j,k].
    A rank-r decomposition gives: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * V[j,k,r] * W[k,i,r]
    Wait, that's not right either.
    
    Standard tensor decomposition for matmul:
    We want: C[i,k] = sum_j A[i,j] B[j,k]
    A rank-r decomposition provides:
    C[i,k] = sum_{r=1}^r U[i, r] * V[k, r] * W[j, r] ... no.
    
    Let me use the standard formulation. The matrix multiplication tensor T has entries:
    T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    Actually, the tensor is: T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    And the multiplication is: C[i,k] = sum_j T_{i,j,k} A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    
    A rank-r decomposition of T is: T_{i,j,k} = sum_{r=1}^r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    Wait, the indices don't match. Let me use the convention from the problem.
    
    The problem says: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    So U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    
    The decomposition should satisfy:
    For the matrix multiplication to work, we need:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * V[j*p+k, r] * W[k*n+i, r]
    
    Hmm, that's not quite right. Let me think of it differently.
    
    Actually, the standard formulation is:
    C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
    
    No, that's not standard either. Let me look at this more carefully.
    
    The matrix multiplication tensor is a 3-tensor T with T[i,j,k] = 1 for all i,j,k.
    The "multiplication" is: given A (n x m) and B (m x p), compute C (n x p) where
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    A rank-r decomposition of the tensor T means we can write:
    T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    But wait, the indices of U, V, W should be consistent. U has indices (i,j), V has (j,k), W has (k,i).
    So the contraction is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    
    But T[i,j,k] = 1 for all i,j,k. So we need:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i in [n], j in [m], k in [p].
    
    That's the condition! Let me verify with schoolbook:
    For schoolbook, each term r corresponds to a unique (i,j,k) triple.
    U[i,j,r] = 1 if r corresponds to (i,j,k), else 0.
    V[j,k,r] = 1 if r corresponds to (i,j,k), else 0.
    W[k,i,r] = 1 if r corresponds to (i,j,k), else 0.
    So for a fixed (i,j,k), only the term r corresponding to (i,j,k) contributes, giving 1*1*1 = 1.
    For any other (i',j',k'), the sum is 0 because no single r matches all three.
    Wait, that's not right. For a fixed (i',j',k'), we sum over r:
    sum_r U[i',j',r] * V[j',k',r] * W[k',i',r]
    For schoolbook, U[i',j',r] is nonzero only for r corresponding to (i',j',k'') for some k''.
    V[j',k',r] is nonzero only for r corresponding to (i'',j',k') for some i''.
    W[k',i',r] is nonzero only for r corresponding to (i',j'',k') for some j''.
    For the product to be nonzero, we need a single r that corresponds to (i',j',k'') AND (i'',j',k') AND (i',j'',k').
    This requires i''=i', j''=j', k''=k', so r corresponds to (i',j',k').
    Then U[i',j',r]=1, V[j',k',r]=1, W[k',i',r]=1, product=1.
    So the sum is 1 for all (i,j,k). Correct!
    
    So the verification is: for all i in [n], j in [m], k in [p]:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_random_search(max_rank, min_rank, time_limit, seed=42):
    """
    Try to find a rank-r decomposition for r from max_rank down to min_rank.
    Uses random initialization and checks if the tensor equation is satisfied.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start_time = time.time()
    
    for rank in range(max_rank, min_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
        # Try multiple random initializations for this rank
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() - start_time < time_limit:
            attempts += 1
            # Random initialization with small integers or half-integers
            U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
            
            # Check if this works
            if _verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W, "rank": rank}
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    First try to find a better-than-schoolbook decomposition, fallback to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Try to find a better decomposition
    # We know schoolbook is 42. Try ranks 41 down to, say, 35.
    # But random search is unlikely to find exact solutions.
    # Let's try a more structured approach.
    
    # Actually, for small problems, we can try to use the fact that n=2 is small.
    # One approach: fix the structure of U, V, W based on the small dimension.
    
    # Let me try a different approach: use the known result for 2x3 * 3x2 which has rank 14.
    # But our problem is 2x3 * 3x7, which is different.
    
    # Given the difficulty, let's just return schoolbook if we can't find better quickly.
    # But let's try a few random searches first.
    
    result = _try_random_search(schoolbook_rank - 1, max(30, schoolbook_rank - 10), 5.0)
    
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fallback to schoolbook
    schoolbook = _get_schoolbook()
    return schoolbook
# EVOLVE-BLOCK-END
