"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
from collections import defaultdict

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses sparse linear algebra to find minimal-rank solutions.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First: try to find a rank-30 solution (theoretical lower bound)
    # Then: try rank 31, 32, ... up to 35
    
    best_result = None
    best_rank = None
    
    # Try ranks from 30 to 35
    for target_rank in range(30, 36):
        if time.time() > deadline - 10:
            break
        
        result = _solve_sparse(target_rank, deadline)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W):
                best_result = (U, V, W)
                best_rank = target_rank
                break
    
    if best_result is None:
        # Fallback: use Strassen rank-35
        U, V, W = _construct_strassen_rank35()
        best_rank = 35
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

def _solve_sparse(target_rank, deadline):
    """
    Solve for a rank-target_rank decomposition using sparse Gaussian elimination.
    
    Variables: U[i,j,r] for i in [0,n), j in [0,m), r in [0,rank)
              V[j,k,r] for j in [0,m), k in [0,p), r in [0,rank)
              W[k,i,r] for k in [0,p), i in [0,n), r in [0,rank)
    
    Constraints: For each (i,j,k,i'), sum_r U[i,j,r]*V[j,k,r]*W[k,i',r] = delta_{i,i'}*delta_{j,j'}
    
    This is a system of quadratic equations. We linearize by fixing W and solving for U,V,
    or use a different approach: treat it as a linear system in the "flattened" variables.
    
    Actually, the correct approach: the tensor T[i,j,k] = delta_{i,k} * I_m[j,j']
    We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, let me re-read the convention. W[k*n+i, r] so W index is (k,i).
    U[i_prime*m+j, r] so U index is (i', j) where i' is the "right" matrix row.
    V[j_prime*p+k_prime, r] so V index is (j', k') where j' is left matrix col, k' is right matrix row.
    
    The product: sum_r W[k*n+i, r] * U[i_prime*m+j, r] * V[j_prime*p+k_prime, r]
    should equal delta_{i,i_prime} * delta_{k,k_prime} * delta_{j,j_prime}
    
    So for fixed (i, i_prime, k, k_prime):
    sum_r W[k*n+i, r] * U[i_prime*m+j, r] * V[j_prime*p+k_prime, r] = delta_{i,i_prime} * delta_{k,k_prime} * delta_{j,j_prime}
    
    This is still quadratic. Let me think differently.
    
    The tensor is: T[i, j, k] where the "output" index is (i, j, k) and 
    T[i,j,k] = 1 if i==k and j==j' else 0... 
    
    Actually the multiplication tensor for C = A*B where A is n×m, B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = delta_{i,k} doesn't make sense. Let me reconsider.
    
    For matrix multiplication A (n×m) times B (m×p) = C (n×p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor decomposition form:
    C[i,k] = sum_r sum_j U[i,j,r] * B[j,k] * ... 
    
    Wait, the standard form is:
    C[i,k] = sum_r (sum_j A[i,j] * U[j,r]) * (sum_k' B[k',k] * V[r,k'])
    
    Or more precisely, the rank-r decomposition of the multiplication tensor:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where T[i,j,k] = 1 if this corresponds to the (i,k) entry of A*B using A[i,j] and B[j,k]
    
    For C = A*B: C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) where j is the summation index.
    Actually T[i,j,k] should be such that:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for the correct terms
    
    Let me just use the verification function to understand the structure.
    """
    n, m, p = DIMENSIONS
    
    # The constraint is: for all (i, i_prime, k, k_prime):
    # sum_r W[k*n+i, r] * U[i_prime*m+j, r] * V[j_prime*p+k_prime, r] = delta_{i,i_prime} * delta_{k,k_prime} * delta_{j,j_prime}
    # Wait, that's not right either.
    
    # Let me re-examine _verify_fast:
    # M[j, j_prime] = sum_r W[k*n+i, r] * U[i_prime*m + j, r] * V[j_prime*p + k_prime, r]
    # Should equal I_m if i==i_prime and k==k_prime, else 0
    
    # So: sum_r W[k*n+i, r] * U[i_prime*m + j, r] * V[j_prime*p + k_prime, r] = delta_{i,i_prime} * delta_{k,k_prime} * delta_{j,j_prime}
    
    # This is a system of (n*n*p*p*m*m) equations in (n*m*rank + m*p*rank + p*n*rank) variables
    # But it's quadratic (product of three variables).
    
    # Key insight: Fix W, then it becomes linear in U and V (bilinear).
    # Or: fix two of the three matrices, solve for the third.
    
    # Alternative approach: use the fact that for each fixed r, we have rank-1 tensors.
    # The sum of rank-1 tensors must equal the target tensor.
    
    # Let me try a different strategy: parameterize and use nonlinear optimization
    # with integer constraints, or use a SAT/SMT approach.
    
    # For now, let's try: fix W to be sparse (mostly zeros), then solve linear system for U,V.
    
    # Strategy: 
    # 1. Choose a pattern for W (which (k,i,r) entries are nonzero)
    # 2. For each such pattern, solve the resulting linear system for U and V
    
    # The target tensor: T[i_prime, j, j_prime, k_prime, k, i] = delta_{i,i_prime} * delta_{k,k_prime} * delta_{j,j_prime}
    # Wait, the indices are: U[i_prime*m+j, r], V[j_prime*p+k_prime, r], W[k*n+i, r]
    # The equation is over (i, i_prime, j, j_prime, k, k_prime):
    # sum_r U[i_prime*m+j, r] * V[j_prime*p+k_prime, r] * W[k*n+i, r] = delta_{i,i_prime} * delta_{k,k_prime} * delta_{j,j_prime}
    
    # For fixed (i, i_prime, k, k_prime) with i==i_prime and k==k_prime:
    # sum_r U[i*m+j, r] * V[j_prime*p+i, r] * W[i*n+i, r] = delta_{j,j_prime}
    
    # For fixed (i, i_prime, k, k_prime) with i!=i_prime or k!=k_prime:
    # sum_r U[i_prime*m+j, r] * V[j_prime*p+k_prime, r] * W[k*n+i, r] = 0
    
    # This is still bilinear. Let me try: fix W, then for each (i, i_prime, k, k_prime),
    # we get a linear system in the products U[*,r]*V[*,r].
    
    # Actually, let me try a much simpler approach: use the known rank-35 Strassen solution
    # and try to compress it using linear algebra.
    
    # Convert to numpy and try to find linear dependencies among the rank-1 terms.
    U, V, W = _construct_strassen_rank35()
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T = sum_r outer(U[:,r], V[:,r], W[:,r])
    # If we can find coefficients c_r such that sum_r c_r * T_r = 0,
    # then we can potentially eliminate one term.
    
    # Flatten each rank-1 tensor
    def flatten_tensor(U_col, V_col, W_col):
        """Flatten a rank-1 tensor U_col ⊗ V_col ⊗ W_col into a vector."""
        n, m, p = DIMENSIONS
        # U_col has shape (n*m,), V_col has shape (m*p,), W_col has shape (p*n,)
        # The tensor T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
        # Wait, the indices don't align for a simple outer product.
        
        # From the verification:
        # sum_r W[k*n+i, r] * U[i_prime*m + j, r] * V[j_prime*p + k_prime, r]
        
        # So the "tensor" is over indices (i, i_prime, j, j_prime, k, k_prime)
        # T[i, i_prime, j, j_prime, k, k_prime] = sum_r W[k*n+i, r] * U[i_prime*m+j, r] * V[j_prime*p+k_prime, r]
        
        # For a single rank-1 term r:
        # T_r[i, i_prime, j, j_prime, k, k_prime] = W[k*n+i, r] * U[i_prime*m+j, r] * V[j_prime*p+k_prime, r]
        
        # This can be written as: (W_col ⊗ U_col ⊗ V_col) with appropriate index mapping
        # But the indices are permuted.
        
        # Let's just enumerate all indices
        vec = np.zeros(n * n * m * m * p * p)
        for i in range(n):
            for i_prime in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                idx = ((i * n + i_prime) * m + j) * m * p * p + ((j_prime * p + k) * p + k_prime)
                                # Wait, let me use a simpler index
                                idx = (i * n + i_prime) * (m * m * p * p) + (j * m + j_prime) * (p * p) + (k * p + k_prime)
                                vec[idx] = W_col[k*n+i] * U_col[i_prime*m+j] * V_col[j_prime*p+k_prime]
        return vec
    
    # Build the matrix of rank-1 tensors
    A = np.zeros((n * n * m * m * p * p, rank))
    for r in range(rank):
        A[:, r] = flatten_tensor(U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Check if any column is in the span of the others
    # Use rank detection
    from scipy.linalg import svd
    
    # For each column, check if it can be eliminated
    for drop_idx in range(rank):
        if time.time() > deadline - 5:
            break
        
        other_cols = [i for i in range(rank) if i != drop_idx]
        A_other = A[:, other_cols]
        b = A[:, drop_idx]
        
        # Check if b is in the column space of A_other
        try:
            # Use least squares
            coeffs, residuals, rank_A, s = np.linalg.lstsq(A_other, b, rcond=None)
            residual = np.linalg.norm(A_other @ coeffs - b)
            norm_b = np.linalg.norm(b)
            
            if norm_b > 1e-10 and residual / norm_b < 1e-10:
                # Column drop_idx is redundant
                U_new = np.delete(U_np, drop_idx, axis=1)
                V_new = np.delete(V_np, drop_idx, axis=1)
                W_new = np.delete(W_np, drop_idx, axis=1)
                
                # Convert back to lists with rounding
                U_list = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_new.tolist()]
                V_list = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_new.tolist()]
                W_list = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_new.tolist()]
                
                return (U_list, V_list, W_list)
        except:
            continue
    
    return None

def _construct_strassen_rank35():
    """Constructs rank-35 decomposition using Strassen's algorithm for each 2x2 block."""
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
    
    return U, V, W

def _verify(U, V, W):
    """Verify decomposition using numpy for speed."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for i_prime in range(n):
            for k in range(p):
                for k_prime in range(p):
                    M = np.zeros((m, m))
                    for j in range(m):
                        for j_prime in range(m):
                            M[j, j_prime] = np.sum(
                                W_np[k*n+i] * U_np[i_prime*m + j] * V_np[j_prime*p + k_prime]
                            )
                    
                    if i == i_prime and k == k_prime:
                        if not np.allclose(M, np.eye(m), atol=1e-10):
                            return False
                    else:
                        if not np.allclose(M, np.zeros((m, m)), atol=1e-10):
                            return False
    return True

# EVOLVE-BLOCK-END
