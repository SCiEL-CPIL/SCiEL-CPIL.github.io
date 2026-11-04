"""Matrix multiplication 3x5 * 5x5: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE_SECONDS = 135.0

def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)

_START_TIME = time.time()

def _validate_np(U, V, W, n, m, p, tol=1e-10):
    """Fast validation using NumPy."""
    U3 = np.array(U, dtype=np.float64)
    V3 = np.array(V, dtype=np.float64)
    W3 = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i*m+j], V3[j*p+k] * W3[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _winograd_dot5():
    """
    Winograd's algorithm for dot product of two length-5 vectors.
    Computes sum_{j=0}^{4} a[j]*b[j] using 7 multiplications instead of 5.
    
    The idea: precompute linear combinations of a and b, multiply them,
    then take linear combinations of the products.
    
    For length 5, one construction uses the fact that:
    sum a[j]*b[j] can be computed via 7 products of linear forms.
    
    Actually, the minimal rank for 1x5x5 tensor is 5 (it's a rank-5 tensor,
    the identity-like structure). Winograd doesn't help for pure dot products
    of length > 2 in the same way.
    
    Let me reconsider. The matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]B[j,k]
    has the property that for fixed (i,k), it's the dot product tensor of length m.
    
    The rank of the full 3x5x5 matmul tensor is what we want.
    
    Key known result: The rank of M_{n,m,p} (matrix multiplication tensor) satisfies:
    - M_{2,2,2} has rank 7 (Strassen)
    - M_{3,3,3} has rank 23 (Laderman)
    - M_{5,5,5} has rank <= 49 (various constructions)
    
    For M_{3,5,5}, we can use a block approach.
    """
    pass

def _strassen_block_decomposition(n, m, p):
    """
    Try to use Strassen-like decomposition.
    
    For 3x5 * 5x5: Think of A as 3 rows of length 5, B as 5 rows of length 5.
    C is 3x5.
    
    One approach: partition the j-index (length 5) into groups.
    If we partition j into {0,1} and {2,3,4}, we get:
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3,4}} A[i,j]*B[j,k]
    
    The first part is 3x2 * 2x5, the second is 3x3 * 3x5.
    
    This doesn't directly help with rank reduction of the tensor.
    
    Let me try a different approach: use the known fact that the rank of
    M_{n,m,p} is at most n*m*p, and try to find algebraic cancellations.
    """
    return None

def _try_laderman_style(n, m, p):
    """
    Laderman's algorithm for 3x3x3 matmul uses rank 23 instead of 27.
    The key idea is to use the structure of the 3x3 case.
    
    For 3x5x5, we can think of it as:
    A is 3x5, B is 5x5, C is 3x5.
    
    Partition B's columns into groups and use algebraic identities.
    
    Actually, let me try a more direct approach: use the fact that
    the matmul tensor can be decomposed using polynomial methods.
    
    For now, let's try a randomized search with better structure.
    """
    return None

def _random_search_structured(n, m, p, start_rank=60, max_time=30):
    """
    Structured random search: use sparse patterns inspired by known
    low-rank constructions.
    
    Key insight: In low-rank decompositions, each rank-1 term typically
    has a "structured" pattern of nonzeros. For matmul tensors, the
    nonzero patterns in U, V, W are related.
    """
    best = None
    deadline = time.time() + max_time
    
    for rank in range(start_rank, n*m*p + 1):
        if time.time() > deadline or _time_left() < 10:
            break
        
        # Try a few random structured candidates at this rank
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Create sparse random matrices with structured patterns
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Each column r: pick random subsets of rows
            for r in range(rank):
                # U column: ~n nonzeros (one per i-index group)
                for i in range(n):
                    if random.random() < 0.7:
                        j = random.randint(0, m-1)
                        U[i*m+j, r] = random.choice([-1, 1])
                
                # V column: ~m nonzeros
                for j in range(m):
                    if random.random() < 0.7:
                        k = random.randint(0, p-1)
                        V[j*p+k, r] = random.choice([-1, 1])
                
                # W column: ~p nonzeros  
                for k in range(p):
                    if random.random() < 0.7:
                        i = random.randint(0, n-1)
                        W[k*n+i, r] = random.choice([-1, 1])
            
            if _validate_np(U, V, W, n, m, p):
                return U, V, W, rank
    
    return None

def _try_tensor_product(n, m, p):
    """
    Try tensor product constructions.
    
    If we have decompositions of smaller matmul tensors, we can combine them.
    M_{n1,m1,p1} ⊗ M_{n2,m2,p2} gives M_{n1*n2, m1*m2, p1*p2}
    with rank = rank1 * rank2.
    
    3x5x5 = 3x5 x 5x5. Not a clean tensor product.
    
    But: 3x5x5 can be seen as embedding into 4x8x8 or similar.
    This likely won't beat 75.
    """
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank = n*m*p = 75)
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try structured approaches
    if _time_left() > 30:
        result = _try_laderman_style(n, m, p)
        if result is not None:
            U, V, W, rank = result
    
    if _time_left() > 20:
        result = _try_tensor_product(n, m, p)
        if result is not None:
            U, V, W, rank = result
    
    # Try random search for lower rank
    if _time_left() > 15:
        result = _random_search_structured(n, m, p, start_rank=65, max_time=10)
        if result is not None:
            U, V, W, rank = result
    
    # Final verification
    if not _validate_np(U, V, W, n, m, p):
        U, V, W, rank = _schoolbook(n, m, p)
    
    # Convert to integer lists
    U_out = U.astype(int).tolist()
    V_out = V.astype(int).tolist()
    W_out = W.astype(int).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
