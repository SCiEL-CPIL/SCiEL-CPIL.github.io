"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to tensors
    U_t = U_arr.reshape(n, m, -1)  # (n, m, rank)
    V_t = V_arr.reshape(m, p, -1)  # (m, p, rank)
    W_t = W_arr.reshape(p, n, -1)  # (p, n, rank)
    
    # The tensor T[i, j, k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Compute the reconstructed tensor
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # The standard matrix multiplication tensor
    # T[i, j, k] = 1 if this is a valid multiplication term, else 0
    # For (n x m) * (m x p) -> (n x p):
    # T[i, j, k] = 1 for the standard bilinear form
    # Actually, the multiplication tensor T[i,j,k] = delta (meaning: A[i,j]*B[j,k] contributes to C[i,k])
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that each (i,j,k) triple has weight 1)
    # Wait, let me think more carefully.
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # because C[i,k] = sum_j A[i,j] * B[j,k], so each (i,j,k) term has coefficient 1.
    
    T_standard = np.ones((n, m, p))
    
    # Check if they match
    diff = np.max(np.abs(T_reconstructed - T_standard))
    return diff < tol, diff

def _construct_tensor_product_fallback():
    """
    Construct a rank-63 decomposition using the tensor product of:
    - Standard 2x2x2 multiplication tensor (rank 7, Strassen)
    - 2x1x9 vector-matrix multiplication tensor (rank 9, schoolbook)
    
    The tensor product of rank-r1 and rank-r2 tensors has rank at most r1*r2.
    Here: 7 * 9 = 63.
    
    Index conventions:
    - U has shape (n*m, rank) = (4, rank), indexed by (i*m + j) where i in {0,1}, j in {0,1}
    - V has shape (m*p, rank) = (18, rank), indexed by (j*p + k) where j in {0,1}, k in {0..8}
    - W has shape (p*n, rank) = (18, rank), indexed by (k*n + i) where k in {0..8}, i in {0,1}
    """
    n, m, p = 2, 2, 9
    rank = 63
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 63
    V = [[0] * rank for _ in range(m * p)]   # 18 x 63
    W = [[0] * rank for _ in range(p * n)]   # 18 x 63
    
    # Strassen's 7 products for 2x2x2
    # U indices: (i*m + j) -> i*2 + j
    # V indices: (j*p + k) -> j*2 + k for k in {0,1}
    # W indices: (k*n + i) -> k*2 + i for k in {0,1}
    
    strassen_products = [
        # (U_entries, V_entries, W_entries)
        # Each entry is (index, coefficient)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),           # M2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),           # M3 = A00(B10-B11)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),           # M4 = A11(B01-B00)
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),           # M5 = (A00+A01)B11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6 = (A10-A00)(B00+B01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7 = (A01-A11)(B10+B11)
    ]
    
    # For each of the 7 Strassen products, and for each of the 9 k-values in the 9-column dimension,
    # we create a rank-1 term.
    
    # The 2x1x9 tensor: multiplying a 2-vector (row) by a 2x9 matrix to get a 1x9 vector.
    # Actually, let me reconsider. The tensor product approach:
    # 
    # We want to compute (2x2) * (2x9) = (2x9)
    # 
    # Think of it as: for each output column k (0..8), we compute a 2x2 * 2x1 multiplication.
    # The 2x2 * 2x1 multiplication tensor has rank 4 (schoolbook) or can be part of Strassen.
    # 
    # Actually, the tensor product approach is:
    # T_{2x2x9} = T_{2x2x2} ⊗ T_{1x1x9/2}... hmm, this isn't quite right.
    # 
    # Let me think differently. The 2x2x9 multiplication tensor can be viewed as:
    # For each pair of "inner" indices, we have a 2x2x2 sub-tensor scaled by the outer index.
    # 
    # Actually, the cleanest tensor product: 
    # The 2x2x9 tensor = sum over k=0..8 of (2x2x2 tensor with only column k active)
    # Each 2x2x2-with-one-column tensor has rank 4 (schoolbook for 2x2*2x1).
    # Total rank: 9 * 4 = 36.
    # 
    # But we can do better by grouping. Let me just use the direct approach:
    # For each k in 0..8, use schoolbook for the 2x2 * 2x1 subproblem.
    # That gives rank 36.
    # 
    # Or use Strassen for pairs of k values: 4 pairs * 7 + 1 single * 4 = 32.
    # That's what the current program does.
    # 
    # For the fallback, let me just use the straightforward rank-36 schoolbook approach.
    
    # Actually, let me implement a clean rank-36 schoolbook fallback:
    # For each k in 0..8, 4 products for the 2x2 * 2x1 multiplication.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 4 x 36
    V = [[0] * rank for _ in range(m * p)]   # 18 x 36
    W = [[0] * rank for _ in range(p * n)]   # 18 x 36
    
    for k in range(p):
        base = k * 4
        # Schoolbook for 2x2 * 2x1:
        # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # Product 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][base] = 1
        V[0 * p + k][base] = 1
        W[k * n + 0][base] = 1
        
        # Product 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][base + 1] = 1
        V[1 * p + k][base + 1] = 1
        W[k * n + 0][base + 1] = 1
        
        # Product 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][base + 2] = 1
        V[0 * p + k][base + 2] = 1
        W[k * n + 1][base + 2] = 1
        
        # Product 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][base + 3] = 1
        V[1 * p + k][base + 3] = 1
        W[k * n + 1][base + 3] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Strategy:
    - Partition the 9 output columns into 4 pairs + 1 single column.
    - For each pair of columns (k1, k2), use Strassen's rank-7 algorithm on the
      2x2 * 2x2 sub-problem.
    - For the single remaining column, use the rank-4 schoolbook algorithm.
    - Total rank: 4*7 + 4 = 32.
    
    Index conventions:
    - U has shape (n*m, rank) = (4, rank), indexed by (i*m + j) where i in {0,1}, j in {0,1}
    - V has shape (m*p, rank) = (18, rank), indexed by (j*p + k) where j in {0,1}, k in {0..8}
    - W has shape (p*n, rank) = (18, rank), indexed by (k*n + i) where k in {0..8}, i in {0,1}
    """
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32
    
    def strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2 starting at column start_col."""
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1
        
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1
        
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        """Add 4 schoolbook products for a single column k."""
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        w_0 = k * n + 0
        w_1 = k * n + 1
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    col = schoolbook_single(8, col)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the rank-32 solution
    try:
        valid, diff = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return result
        else:
            # Fallback to rank-36 schoolbook
            return _construct_tensor_product_fallback()
    except Exception:
        # Fallback to rank-36 schoolbook
        return _construct_tensor_product_fallback()
# EVOLVE-BLOCK-END
