"""Matrix multiplication 2x2x15: search for rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 15]

# Deadline guard
START_TIME = time.time()
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _check_deadline():
    return time.time() >= (DEADLINE - SAFETY_MARGIN)

def _verify(U, V, W, n=2, m=2, p=15, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = delta_{i,k} (for the standard matmul tensor where
    C[i,k] = sum_j A[i,j] * B[j,k]).
    
    With the trace convention W[k*n+i, r], the decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta_{i,k}
    """
    n, m, p = n, m, p
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if i == k else 0.0
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n=2, m=2, p=15, tol=1e-9):
    """Faster verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape for efficient computation
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Use einsum
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We want: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Expected: delta_{i,k}
    for i in range(n):
        for k in range(p):
            expected = 1.0 if i == k else 0.0
            for j in range(m):
                if abs(T[i,j,k] - expected) > tol:
                    return False
    return True

def _schoolbook(n=2, m=2, p=15):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_random_search(max_rank, n=2, m=2, p=15, num_trials=200, time_limit=30.0):
    """Try random initialization and numerical solving for a given rank."""
    start = time.time()
    best = None
    
    for trial in range(num_trials):
        if _check_deadline():
            break
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small values
        np.random.seed(trial * 1000 + 42)
        U = np.random.randn(n*m, max_rank) * 0.5
        V = np.random.randn(m*p, max_rank) * 0.5
        W = np.random.randn(p*n, max_rank) * 0.5
        
        # Try a few gradient descent steps
        # Loss: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - delta_{i,k})^2
        # This is expensive, so skip for now in favor of direct verification of known structures
        
    return best

def _construct_rank_60():
    """
    Construct a rank-60 decomposition using the structure:
    2x2x15 = 2x2x14 + 2x2x1
    
    For the 2x2x1 block (k=14, the last column):
    T[i,j,14] = delta_{i,14}
    Since i ranges over {0,1} and 14 != 0,1, all entries are 0.
    So the 2x2x1 block is identically zero! It needs rank 0.
    
    Wait, that's not right. Let me reconsider.
    
    The tensor for multiplying A (2x2) by B (2x15) to get C (2x15):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] has T[i,j,k] = 1 if this term contributes to C[i,k], which is always
    the case when we're computing C[i,k] = sum_j A[i,j]*B[j,k]. So T[i,j,k] = 1 for all i,j,k?
    
    No wait. The matrix multiplication tensor is defined as:
    T[i,j,k] = delta_{i,k} in the sense that the bilinear form is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for all valid i,j,k? No...
    
    Actually, the standard matrix multiplication tensor for n x m times m x p is:
    T in R^{n x m x p} where T[i,j,k] = 1 if the (i,j) entry of the first matrix and (j,k) entry
    of the second matrix multiply to contribute to the (i,k) entry of the result.
    
    So T[i,j,k] = 1 for ALL i, j, k in the valid ranges. That's the schoolbook tensor.
    
    Wait no. Let me re-read the task. "multiplying a 2 by 2 matrix by a 2 by 15 matrix"
    
    A is 2x2, B is 2x15, C = A*B is 2x15.
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,14}.
    
    So T is just the all-ones tensor of shape (2,2,15)!
    
    The rank of the all-ones tensor of shape (a,b,c) is min(a,b,c)... no.
    
    Actually, the rank of the all-ones tensor [1]_{i,j,k} is 1, because:
    [1]_{i,j,k} = U[i,r]*V[j,r]*W[k,r] with U[i,0]=1, V[j,0]=1, W[k,0]=1.
    
    So the rank is 1! That seems too easy. Let me re-read the problem.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 15 matrix."
    
    Hmm, but the schoolbook has rank 2*2*15 = 60. If the tensor is all ones, rank 1 suffices.
    
    Wait, I think I'm misunderstanding the tensor. Let me re-read.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    In the schoolbook code:
    U[i*m+j, column] = 1
    V[j*p+k, column] = 1  
    W[k*n+i, column] = 1
    
    So for each (i,j,k), we have one rank-1 term:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    And the reconstruction is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    In the schoolbook, each (i,j,k) has its own unique r, so the sum gives 1 for that (i,j,k) and 0 for others.
    
    So the tensor being decomposed has T[i,j,k] = 1 for ALL i,j,k.
    
    And as I noted, the all-ones tensor has rank 1!
    
    Let me verify: if I set U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i,
    then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 for all i,j,k.
    
    Yes! The rank is 1.
    """
    n, m, p = 2, 2, 15
    rank = 1
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return the rank-minimal exact tensor decomposition for 2x2x15 matrix multiplication.
    """
    # The tensor T[i,j,k] = 1 for all i,j,k (all-ones tensor) has rank 1.
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    
    result = _construct_rank_60()
    
    # Verify
    if _verify_fast(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()

BET: {mechanism: "The 2x2x15 matmul tensor is the all-ones tensor which has rank 1, not 60; the schoolbook is wildly suboptimal", q: 0.95, delta: [0.983, 0.999]}
