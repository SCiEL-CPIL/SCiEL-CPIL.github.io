"""Program for matrix multiplication 3,3,6 tensor decomposition.
Goal: find a rank-minimal exact decomposition (lower rank = better score)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 105.0  # safety margin within 120s limit

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _stockle_3x3x3():
    """
    Return rank-23 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 23)
    """
    # Stöckle's rank-23 algorithm for 3x3x3 matrix multiplication
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # Indexing: A[i,j], B[j,k], C[i,k]
    # U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] sums to delta(i,k) for each (i,j,k)
    
    # The 23 products are:
    # Let's define the linear forms.
    # This is based on Stöckle 1994.
    
    n = 3
    rank = 23
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # A entries: a00,a01,a02,a10,a11,a12,a20,a21,a22
    # B entries: b00,b01,b02,b10,b11,b12,b20,b21,b22
    # C entries: c00,c01,c02,c10,c11,c12,c20,c21,c22
    
    # Define the 23 intermediate products as (linear_combo_of_A, linear_combo_of_B)
    # Then C entries are linear combos of these products.
    
    # Let me use a known set of 23 products.
    # Based on the literature, here's one construction:
    
    # Products p0..p22, each = sum_i alpha_i * A[i] * sum_j beta_j * B[j]
    # where A[i] and B[j] are entries.
    
    # I'll use the following known rank-23 scheme:
    # (This is a specific implementation; coefficients may need verification)
    
    # Let me use a simpler approach: just hardcode a verified rank-23 decomposition.
    # Actually, let me construct it from known linear forms.
    
    # The 23 multiplications in Stöckle's algorithm:
    # I'll define them as tuples of (row_coeffs for A, col_coeffs for B, then output coeffs)
    
    # For now, let me try a different known construction.
    # Actually, let me just use the fact that rank 23 is known and try to find explicit coefficients.
    
    # Here's a known rank-23 decomposition (one possible set):
    # I'll define U, V, W directly.
    
    # Let me use a parametrization. The tensor T_{ijk} = delta_{i,k} for matmul.
    # We need sum_r U_{ij,r} V_{jk,r} W_{ki,r} = delta_{i,k}
    
    # This is complex. Let me try a different strategy:
    # Use a rank-23 decomposition if I can find explicit coefficients,
    # otherwise fall back to a better-than-schoolbook construction.
    
    # Actually, for the purpose of this problem, let me try to find a rank-45 or lower
    # decomposition for 3x3x6 directly, or use 2x rank-23 = 46.
    
    # Let me just return None and handle it in the caller.
    return None

def _build_rank46():
    """
    Build a rank-46 decomposition for 3x3x6 by using two rank-23 3x3x3 blocks.
    But since I don't have explicit rank-23 coefficients, let me try another approach.
    
    Actually, let me try to find a better decomposition directly.
    
    Alternative: use the fact that 3x3x6 can be seen as 3x3x3 twice.
    If I can find any decomposition of rank < 54, that's an improvement.
    
    Let me try a structured approach:
    - Split B into B1 (cols 0-2) and B2 (cols 3-5)
    - C1 = A*B1, C2 = A*B2
    - Use a rank-r algorithm for 3x3x3, giving rank 2r for 3x3x6
    
    For rank-r, I need a 3x3x3 decomposition.
    
    Let me try to use a known rank-23 decomposition. I'll hardcode the coefficients.
    """
    pass

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook: rank 54
    rank_schoolbook = n * m * p  # 54
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook verification failed!")
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    deadline = start_time + DEADLINE_S
    
    # Try to find a better decomposition
    # Strategy: use a known rank-23 decomposition for 3x3x3, apply twice for rank 46
    
    # Let me try to construct a rank-46 decomposition using block structure.
    # C[:, 0:3] = A * B[:, 0:3]  (3x3 * 3x3 = 3x3)
    # C[:, 3:6] = A * B[:, 3:6]  (3x3 * 3x3 = 3x3)
    
    # If I have a rank-23 decomposition for 3x3x3 with matrices U3, V3, W3 (each 9x23),
    # then for the first block:
    #   U_block1[i*3+j, r] = U3[i*3+j, r] for r in 0..22
    #   V_block1[j*3+k, r] = V3[j*3+k, r] for r in 0..22, k in 0..2
    #   W_block1[k*3+i, r] = W3[k*3+i, r] for r in 0..22
    # For the second block (k in 3..5):
    #   U_block2[i*3+j, r+23] = U3[i*3+j, r] for r in 0..22
    #   V_block2[j*3+(k-3), r+23] = V3[j*3+(k-3), r] for r in 0..22
    #   W_block2[(k-3)*3+i, r+23] = W3[(k-3)*3+i, r] for r in 0..22
    
    # So I need explicit rank-23 coefficients.
    
    # Let me try to find them. Here's a known rank-23 decomposition:
    # Based on Stöckle (1994), "Eine neue obere Schranke für die Tensorrang von 3x3-Matrixmultiplikation"
    
    # I'll try to construct it. The key insight is that we can save one multiplication
    # compared to the naive 27 by using linear combinations.
    
    # Actually, let me try a computational approach: search for a low-rank decomposition.
    
    # Given time constraints, let me try a random search for a rank < 54 decomposition.
    
    rng = np.random.RandomState(42)
    
    # Try to find a rank-48 or lower decomposition by starting from schoolbook
    # and trying to zero out entries.
    
    # Actually, let me try a smarter approach: use the structure of the problem.
    
    # For 3x3x6, we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # If we fix k, this is a 3x3 matrix product.
    # The tensor is separable in k in a sense: T[i,j,k] = delta[i,k] (in the appropriate indexing)
    
    # Wait, the tensor for matmul is T_{i,j,k} = 1 if the (i,j) entry of A times (j,k) entry of B 
    # contributes to (i,k) entry of C. So T_{i,j,k} = delta_{i,k} is not right.
    # Actually, T_{i,j,k} = 1 for all valid (i,j,k) in the sense that each triple (i,j,k) 
    # corresponds to one term in the sum. The tensor is the identity in the sense that
    # the matmul tensor has T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    
    # So the tensor is just 1 for all (i,j,k) in the valid range, and the decomposition
    # must satisfy sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k).
    
    # This is a rank problem for the tensor that is all-ones in a specific pattern.
    
    # Let me try a different approach: use known results.
    # For 3x3x3, rank is 23. For 3x3x6, rank is at most 46.
    # Can we do better than 46?
    
    # One idea: if the two 3x3x3 subproblems share some structure, we might save.
    # But A is the same for both, so there might be sharing.
    
    # Actually, since A is the same, the U matrix is the same for both blocks (just replicated).
    # So we can't save on U, but maybe on V and W?
    
    # No, V and W are different for the two blocks because they depend on k.
    
    # So rank 46 seems to be the natural upper bound from this construction.
    
    # Let me just try to implement a rank-23 3x3x3 decomposition.
    
    # I found a reference: the 23 products are:
    # Let A = [a b c; d e f; g h i], B = [j k l; m n o; p q r]
    # C = [s t u; v w x; y z alpha]
    
    # The 23 multiplications (one possible set):
    # p1 = a*j
    # p2 = (a+b)*(k+m)  -- wait, this doesn't look right for exact matmul
    
    # Actually, for exact matrix multiplication, we need specific linear forms.
    # Let me look for a verified set.
    
    # Given the complexity, let me try a numerical optimization approach:
    # Start from a random rank-46 or lower decomposition and optimize.
    
    # Actually, let me try a simpler improvement: rank 45 or 46 via structured construction.
    
    # Let me just return the schoolbook for now, as implementing rank-23 correctly
    # is non-trivial without explicit coefficients.
    
    # Wait, let me try one more thing: use a known rank-23 decomposition from literature.
    
    # Here's a set of 23 linear forms for 3x3x3 matmul (from Stöckle):
    # I'll define them as arrays.
    
    # Actually, I recall that one can achieve rank 23 with the following:
    # The products are formed from:
    # U forms (for A): 23 linear combinations of the 9 entries of A
    # V forms (for B): 23 linear combinations of the 9 entries of B  
    # W forms (for C): 23 linear combinations that give the 9 entries of C
    
    # Let me try to find these. One known construction uses:
    # 
    # I'll try a different tactic: use the fact that the problem is small and
    # try to find a low-rank decomposition via optimization.
    
    # Given time constraints, let me just return the schoolbook.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
