"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import itertools
import numpy as np
from numpy.linalg import matrix_rank, solve, lstsq

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal (or best found) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, provide the schoolbook decomposition as a guaranteed valid result
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Strategy 1: Try Winograd-style rectangular transform decomposition
    if time.time() < deadline:
        winograd_result = _try_winograd_rectangular(n, m, p)
        if winograd_result is not None:
            rank = len(winograd_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = winograd_result
    
    # Strategy 2: Try symmetric/block decomposition
    if time.time() < deadline:
        symmetric_result = _try_symmetric_decomposition(n, m, p)
        if symmetric_result is not None:
            rank = len(symmetric_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = symmetric_result
    
    # Strategy 3: Try rank reduction via linear dependency search
    if time.time() < deadline:
        reduced_result = _try_rank_reduction(n, m, p, best_result, best_rank)
        if reduced_result is not None:
            rank = len(reduced_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = reduced_result
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _try_winograd_rectangular(n, m, p):
    """
    Try Winograd-style rectangular transform decomposition.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    we can use transform-based methods.
    
    For 3x5x5:
    - We can decompose the 5x5 multiplication using Winograd transforms
    - Then handle the 3 rows of A
    
    Winograd's algorithm for square matrices uses transforms to reduce
    multiplications. For 5x5, we can try to find an efficient transform.
    """
    # For 5x5 matrix multiplication, Winograd can potentially reduce rank
    # The key idea: use transform matrices M, G, T such that
    # C = T * (G * A) * (M * B)
    # where * denotes element-wise multiplication followed by transform
    
    # For now, let's try a simple block decomposition
    # Split 5 into 2+3 or 3+2
    
    # Try 3+2 split for the middle dimension
    m1, m2 = 3, 2
    p1, p2 = 3, 2  # We can also split p
    
    # This gives us blocks:
    # C[0:3, 0:3] = A[0:3, 0:3] * B[0:3, 0:3] + A[0:3, 3:5] * B[3:5, 0:3]
    # etc.
    
    # For simplicity, let's try a direct approach with the full 5x5
    # and see if we can find a lower-rank decomposition
    
    # Actually, let's try to construct an explicit decomposition
    # using the fact that we can share some computations
    
    # For 3x5x5, one approach:
    # - Use 5 independent 3x1x5 multiplications (one for each column of B)
    # - Each has rank 3*5 = 15
    # - Total: 5 * 15 = 75 (same as schoolbook)
    
    # To do better, we need to find shared structure
    
    # Let's try a different approach: use the structure of the problem
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can group terms:
    # For each i and k, C[i,k] is a dot product of A[i,:] and B[:,k]
    
    # One idea: use the fact that we only need 3*5 = 15 output values
    # and each depends on 5 inputs from A and 5 from B
    
    # Let's try to find a decomposition with rank < 75
    # by exploiting linear dependencies
    
    # For now, return None and let other strategies try
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The matrix multiplication tensor has entries T[i,j,k] = 1 for all valid
    i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition means:
    T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This must equal 1 for all valid i, j, k.
    """
    rank = len(U[0])
    # Build numpy arrays for efficient verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over all rank terms
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_symmetric_decomposition(n, m, p):
    """
    Try a decomposition that exploits symmetry in the problem structure.
    
    For 3x5x5, we can try to find a decomposition that uses the
    structure of the dimensions more effectively.
    """
    # One approach: note that n=3, m=5, p=5
    # We can try to decompose this as a combination of smaller multiplications
    
    # For example, we can write:
    # C = A * B where A is 3x5, B is 5x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try to find a decomposition with rank < 75
    # by looking for linear dependencies
    
    # The tensor T has entries T[i,j,k] = 1 for all valid i,j,k
    # We want to write T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # One approach: use the fact that we can permute indices
    # and look for patterns
    
    # For now, let's try a simple heuristic:
    # randomly generate rank-1 terms and see if we can cover the tensor
    
    rng = np.random.RandomState(42)
    
    # Try to find a decomposition with rank 50
    target_rank = 50
    
    # Generate random rank-1 terms
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(target_rank):
        # Generate random vectors
        u = rng.randn(n * m)
        v = rng.randn(m * p)
        w = rng.randn(p * n)
        
        # Normalize to have some structure
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)
        
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    # Check if this decomposition works (it won't, but let's verify)
    # This is just a placeholder - we need a better approach
    
    # Actually, let's try a more structured approach
    # Use the fact that the tensor is all ones
    
    # For T[i,j,k] = 1, we can write:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # One simple decomposition: use indicator functions
    # But that gives rank 75
    
    # Let's try to find a lower-rank decomposition by exploiting
    # the fact that we have 3 rows of A
    
    # For each row i of A, we need to compute A[i,:] * B
    # This is a 1x5 times 5x5 multiplication
    
    # The rank of 1x5 x 5x5 multiplication is at most 5*5 = 25
    # But we have 3 such multiplications, so naive is 75
    
    # Can we share computations between the 3 rows?
    # If we write A as a sum of rank-1 matrices, we might be able to
    # reuse some of the computations
    
    # For example, if A = sum_{s=1}^3 a_s ⊗ b_s where a_s is 1x3 and b_s is 3x5
    # Then A*B = sum_s (a_s * (b_s * B))
    
    # But this doesn't directly help with rank reduction
    
    # Let's try a different approach: use the structure of 5
    # 5 = 2 + 3, so we can split B into two parts
    
    # B = [B1, B2] where B1 is 5x2 and B2 is 5x3
    # Then A*B = [A*B1, A*B2]
    
    # A*B1 is 3x2, rank at most 3*2*5 = 30
    # A*B2 is 3x3, rank at most 3*3*5 = 45
    # Total: 75 (no improvement)
    
    # Let's try yet another approach: use the fact that we can
    # write the multiplication in terms of smaller multiplications
    
    # For 5x5, we can use the identity:
    # 5x5 multiplication can be done with fewer than 25 multiplications
    # using Winograd's algorithm
    
    # For now, return None
    return None


def _try_rank_reduction(n, m, p, base_result, base_rank):
    """
    Try to reduce the rank by finding linear dependencies in the decomposition.
    
    Strategy: Start with a valid decomposition and try to express some
    rank-1 terms as linear combinations of others.
    """
    U = np.array(base_result['U'], dtype=float)
    V = np.array(base_result['V'], dtype=float)
    W = np.array(base_result['W'], dtype=float)
    
    rank = U.shape[1]
    
    # Try to find if some columns are linearly dependent
    # This is a heuristic approach
    
    # For now, let's try a simpler approach:
    # randomly perturb and see if we can maintain the decomposition with fewer terms
    
    # Actually, let's try to use the structure more directly.
    # The tensor is T[i,j,k] = 1. We can try to find a decomposition
    # with rank < 75.
    
    # One approach: use the fact that we can write the multiplication as
    # C = A * B where A is 3x5 and B is 5x5.
    
    # Let's try to construct a decomposition with rank 50 (just as a test)
    # and see if it works.
    
    # Actually, this is getting complex. Let's return None for now.
    return None


# EVOLVE-BLOCK-END
