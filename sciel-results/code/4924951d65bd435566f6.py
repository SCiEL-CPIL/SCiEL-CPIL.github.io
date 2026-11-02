"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    C = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return bool(np.allclose(C, 1.0, atol=tol))

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each product r, define u (2x2), v (2x2), w (2x2) such that:
    # sum_r u_r * v_r * w_r = identity in the tensor sense
    
    products = [
        # Product 1: (A11+A22)(B11+B22) -> contributes to C11 and C22
        (np.array([[1,0],[0,1]], dtype=np.float64), 
         np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[1,0],[0,-1]], dtype=np.float64)),
        # Product 2: (A21+A22)B11
        (np.array([[0,0],[1,1]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64)),
        # Product 3: A11(B21-B22)
        (np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[0,0],[1,-1]], dtype=np.float64), 
         np.array([[0,1],[0,0]], dtype=np.float64)),
        # Product 4: A22(B12-B11)
        (np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[0,1],[-1,0]], dtype=np.float64), 
         np.array([[0,0],[1,0]], dtype=np.float64)),
        # Product 5: (A11+A12)B22
        (np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[-1,0],[0,1]], dtype=np.float64)),
        # Product 6: (A21-A11)(B11+B12)
        (np.array([[-1,0],[1,0]], dtype=np.float64), 
         np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64)),
        # Product 7: (A12-A22)(B21+B22)
        (np.array([[0,1],[0,-1]], dtype=np.float64), 
         np.array([[0,0],[1,1]], dtype=np.float64), 
         np.array([[1,0],[0,1]], dtype=np.float64)),
    ]
    
    for r, (Ur, Vr, Wr) in enumerate(products):
        U[:, r] = Ur.flatten()
        V[:, r] = Vr.flatten()
        W[:, r] = Wr.flatten()
    
    return U, V, W

def _build_1x2x2():
    """
    Build a rank-2 decomposition for 1x2x2 matrix multiplication.
    A is 1x2, B is 2x2, C is 1x2.
    C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    Schoolbook rank is 4, but we can do rank 2 using Strassen-like ideas.
    
    Actually for 1x2x2: C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0,1
    This is just two independent dot products of length 2.
    Minimum rank is 2 (can't do better than the number of outputs when each output
    is a dot product of two terms).
    
    Wait, let me reconsider. The tensor is T[i,j,k] = 1 for i=0, j in {0,1}, k in {0,1}
    We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    For i=0: sum_r U[0,j,r] V[j,k,r] W[k,0,r] = 1 for all j,k
    This means for each (j,k), the product must equal 1.
    
    Rank 2: 
    r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1, U[0,1,0]=0, V[1,*,0]=0, W[*,*,0]=0 (except W[0,0,0])
    r=1: U[0,1,1]=1, V[1,*,1]=?, W[*,0,1]=?
    
    Actually this is getting complicated. Let me just use schoolbook for 1x2x2 (rank 4).
    """
    n, m, p = 1, 2, 2
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _build_3x2x2():
    """
    Build a decomposition for 3x2x2 matrix multiplication.
    A is 3x2, B is 2x2, C is 3x2.
    Schoolbook rank is 12.
    
    We can view this as 3 independent 1x2x2 multiplications that share B.
    But they don't share anything in the tensor decomposition sense directly.
    
    Alternative: Use the fact that for each row i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Let me try a structured approach. For 3x2x2, we can potentially use rank 9 or less.
    For now, use schoolbook (rank 12) and see if we can improve.
    """
    n, m, p = 3, 2, 2
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _build_3x5x2():
    """
    Build a decomposition for 3x5x2 matrix multiplication.
    Schoolbook rank is 30.
    
    Strategy: Split the 5 middle columns into groups.
    3x5x2 = 3x2x2 + 3x2x2 + 3x1x2
    
    For 3x1x2: C[i,k] = A[i,0]*B[0,k] for k=0,1. Rank 2 per i, total rank 6.
    Actually for 3x1x2: T[i,0,k] = 1 for i in {0,1,2}, k in {0,1}
    This is just 6 independent terms, rank 6.
    
    So 3x5x2: rank = 12 + 12 + 6 = 30 (same as schoolbook)
    
    Let me try a different approach. Use Strassen on 2x2x2 blocks.
    """
    n, m, p = 3, 5, 2
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _build_3x5x6():
    """
    Build a decomposition for 3x5x6 matrix multiplication.
    Schoolbook rank is 90.
    
    Strategy: Use block decomposition and Strassen where possible.
    
    3x5x6: We can split this into blocks.
    - Split the 6 output columns into 3 groups of 2: 3x5x2 + 3x5x2 + 3x5x2
    - Each 3x5x2 has schoolbook rank 30, total 90 (no improvement)
    
    Alternative: Split the 5 middle columns.
    3x5x6 = 3x2x6 + 3x2x6 + 3x1x6
    
    For 3x2x6: schoolbook rank 36. Two of these: 72.
    For 3x1x6: schoolbook rank 18.
    Total: 72 + 18 = 90 (no improvement)
    
    Let me try using Strassen on 2x2x2 sub-blocks.
    
    Actually, let me think about this differently. The key insight is that
    matrix multiplication tensor has structure that can be exploited.
    
    For 3x5x6, one approach:
    - Group the 5 middle indices into pairs: (0,1), (2,3), (4)
    - Group the 6 output indices into pairs: (0,1), (2,3), (4,5)
    
    For each pair of middle indices and pair of output indices, we have a 3x2x2 sub-problem.
    There are 3 * 3 = 9 such sub-problems.
    
    If we use Strassen (rank 7) for each 3x2x2... wait, Strassen is for 2x2x2.
    For 3x2x2, we need a different approach.
    
    Let me just use schoolbook for now and focus on verification.
    """
    n, m, p = 3, 5, 6
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _combine_blocks_mid(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two block decompositions along the middle dimension:
    Block 1: n x m1 x p
    Block 2: n x m2 x p
    Total: n x (m1+m2) x p
    
    U1: (n*m1, r1), V1: (m1*p, r1), W1: (p*n, r1)
    U2: (n*m2, r2), V2: (m2*p, r2), W2: (p*n, r2)
    
    Combined U: (n*(m1+m2), r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((n * (m1 + m2), total_rank), dtype=np.float64)
    V = np.zeros(((m1 + m2) * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # For U: U[i*m_total + j, r]
    for i in range(n):
        for j in range(m1):
            U[i * (m1 + m2) + j, :r1] = U1[i * m1 + j, :]
        for j in range(m2):
            U[i * (m1 + m2) + m1 + j, r1:] = U2[i * m2 + j, :]
    
    # For V: V[j*p_total + k, r]
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :r1] = V1[j * p + k, :]
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k, r1:] = V2[j * p + k, :]
    
    # For W: W[k*n + i, r]
    for k in range(p):
        for i in range(n):
            W[k * n + i, :r1] = W1[k * n + i, :]
            W[k * n + i, r1:] = W2[k * n + i, :]
    
    return U, V, W

def _combine_blocks_out(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two block decompositions along the output dimension:
    Block 1: n x m x p1
    Block 2: n x m x p2
    Total: n x m x (p1+p2)
    
    U1: (n*m, r1), V1: (m*p1, r1), W1: (p1*n, r1)
    U2: (n*m, r2), V2: (m*p2, r2), W2: (p2*n, r2)
    
    Combined U: (n*m, r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * (p1 + p2), total_rank), dtype=np.float64)
    W = np.zeros(((p1 + p2) * n, total_rank), dtype=np.float64)
    
    # For U: U[i*m + j, r] - same structure for both blocks
    for i in range(n):
        for j in range(m):
            U[i * m + j, :r1] = U1[i * m + j, :]
            U[i * m + j, r1:] = U2[i * m + j, :]
    
    # For V: V[j*p_total + k, r]
    for j in range(m):
        for k in range(p1):
            V[j * (p1 + p2) + k, :r1] = V1[j * p1 + k, :]
        for k in range(p2):
            V[j * (p1 + p2) + p1 + k, r1:] = V2[j * p2 + k, :]
    
    # For W: W[k*n + i, r]
    for k in range(p1):
        for i in range(n):
            W[k * n + i, :r1] = W1[k * n + i, :]
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i, r1:] = W2[k * n + i, :]
    
    return U, V, W

def _random_search(n, m, p, target_rank, time_limit=10.0):
    """
    Random search for a rank-target_rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
        V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
        W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    overall_deadline = start_time + 130.0
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try block decomposition: 3x5x6 = 3x2x6 + 3x3x6
    if time.time() < overall_deadline - 20.0:
        U1, V1, W1 = _schoolbook_decomposition(3, 2, 6)
        U2, V2, W2 = _schoolbook_decomposition(3, 3, 6)
        U_block, V_block, W_block = _combine_blocks_mid(U1, V1, W1, U2, V2, W2, n, 2, 3, p)
        
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            block_rank = U_block.shape[1]
            if block_rank < best_rank:
                best_rank = block_rank
                best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
    
    # Try random search for lower ranks
    if time.time() < overall_deadline - 10.0:
        for target_rank in range(80, min(best_rank, 90)):
            if time.time() > overall_deadline - 5.0:
                break
            remaining = min(5.0, overall_deadline - time.time())
            result = _random_search(n, m, p, target_rank, time_limit=remaining)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
