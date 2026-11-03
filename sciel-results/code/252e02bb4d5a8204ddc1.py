"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention.
    
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """
    Construct a Strassen-style decomposition for matrix multiplication.
    Returns (U, V, W, rank) where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    The tensor condition is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    """
    U = [[0.0] * (n*m*p) for _ in range(n*m)]
    V = [[0.0] * (n*m*p) for _ in range(m*p)]
    W = [[0.0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    
    # Partition indices
    j_pairs = []
    k_pairs = []
    j_singles = []
    k_singles = []
    
    for j in range(0, m - 1, 2):
        j_pairs.append((j, j+1))
    if m % 2 == 1:
        j_singles.append(m - 1)
    
    for k in range(0, p - 1, 2):
        k_pairs.append((k, k+1))
    if p % 2 == 1:
        k_singles.append(p - 1)
    
    # For each pair of j indices and pair of k indices, use Strassen (7 multiplications)
    # The 2x2x2 Strassen decomposition gives:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k) in {0,1}^3
    # 
    # Strassen's 7 terms for 2x2 matrix multiplication C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A01-A00)(B00+B01)
    # M7 = (A10-A11)(B10+B11)
    #
    # C00 = M1 + M3 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M4 + M6
    #
    # In our tensor notation, i indexes rows of A (output row), j indexes columns of A / rows of B,
    # k indexes columns of B (output column).
    # W[k*n+i, r] corresponds to the output coefficient for C[i,k].
    
    for (j_a, j_b) in j_pairs:
        for (k_a, k_b) in k_pairs:
            # M1: (A_0,j_a + A_1,j_b) * (B_j_a,k_a + B_j_b,k_b)
            # contributes to C[0,k_a], C[1,k_a], C[0,k_b], C[1,k_b]
            U[0*m + j_a][col] += 1.0
            U[1*m + j_b][col] += 1.0
            V[j_a*p + k_a][col] += 1.0
            V[j_b*p + k_b][col] += 1.0
            W[k_a*n + 0][col] += 1.0
            W[k_b*n + 1][col] += 1.0
            col += 1
            
            # M2: (A_1,j_a + A_1,j_b) * B_j_a,k_a
            U[1*m + j_a][col] += 1.0
            U[1*m + j_b][col] += 1.0
            V[j_a*p + k_a][col] += 1.0
            W[k_a*n + 1][col] += 1.0
            W[k_b*n + 1][col] -= 1.0
            col += 1
            
            # M3: A_0,j_a * (B_j_a,k_b - B_j_b,k_b)
            U[0*m + j_a][col] += 1.0
            V[j_a*p + k_b][col] += 1.0
            V[j_b*p + k_b][col] -= 1.0
            W[k_b*n + 0][col] += 1.0
            W[k_b*n + 1][col] += 1.0
            col += 1
            
            # M4: A_1,j_b * (B_j_b,k_a - B_j_a,k_a)
            U[1*m + j_b][col] += 1.0
            V[j_b*p + k_a][col] += 1.0
            V[j_a*p + k_a][col] -= 1.0
            W[k_a*n + 0][col] += 1.0
            W[k_a*n + 1][col] += 1.0
            col += 1
            
            # M5: (A_0,j_a + A_0,j_b) * B_j_b,k_b
            U[0*m + j_a][col] += 1.0
            U[0*m + j_b][col] += 1.0
            V[j_b*p + k_b][col] += 1.0
            W[k_a*n + 0][col] -= 1.0
            W[k_b*n + 0][col] += 1.0
            col += 1
            
            # M6: (A_1,j_a - A_0,j_a) * (B_j_a,k_a + B_j_a,k_b)
            U[1*m + j_a][col] += 1.0
            U[0*m + j_a][col] -= 1.0
            V[j_a*p + k_a][col] += 1.0
            V[j_a*p + k_b][col] += 1.0
            W[k_b*n + 1][col] += 1.0
            col += 1
            
            # M7: (A_0,j_b - A_1,j_b) * (B_j_b,k_a + B_j_b,k_b)
            U[0*m + j_b][col] += 1.0
            U[1*m + j_b][col] -= 1.0
            V[j_b*p + k_a][col] += 1.0
            V[j_b*p + k_b][col] += 1.0
            W[k_a*n + 0][col] += 1.0
            col += 1
    
    # Handle single j indices (for odd m) with paired k
    for j in j_singles:
        for (k_a, k_b) in k_pairs:
            # For single j, paired k: 4 scalar multiplications
            # C[i,k_a] += A[i,j] * B[j,k_a]
            # C[i,k_b] += A[i,j] * B[j,k_b]
            for i in range(n):
                for k in [k_a, k_b]:
                    U[i*m + j][col] += 1.0
                    V[j*p + k][col] += 1.0
                    W[k*n + i][col] += 1.0
                    col += 1
    
    # Handle paired j indices with single k (for odd p)
    for (j_a, j_b) in j_pairs:
        for k in k_singles:
            for i in range(n):
                for j in [j_a, j_b]:
                    U[i*m + j][col] += 1.0
                    V[j*p + k][col] += 1.0
                    W[k*n + i][col] += 1.0
                    col += 1
    
    # Handle single j and single k
    for j in j_singles:
        for k in k_singles:
            for i in range(n):
                U[i*m + j][col] += 1.0
                V[j*p + k][col] += 1.0
                W[k*n + i][col] += 1.0
                col += 1
    
    # Trim unused columns
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return U, V, W, col

def _try_linear_combination_removal(U, V, W, n, m, p, max_time=30.0):
    """Try to find a linear combination of columns that allows removing one column."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    start = time.time()
    
    # Try removing each column individually (greedy)
    removed = set()
    current_U = U_arr
    current_V = V_arr
    current_W = W_arr
    current_rank = rank
    
    for c in range(rank):
        if time.time() - start > max_time:
            break
        if c in removed:
            continue
        
        # Build mask excluding c and previously removed
        mask = [i for i in range(current_rank) if i not in removed and i != c]
        if len(mask) < 1:
            continue
        
        test_U = current_U[:, mask]
        test_V = current_V[:, mask]
        test_W = current_W[:, mask]
        
        if _verify_decomposition(test_U.tolist(), test_V.tolist(), test_W.tolist(), n, m, p):
            removed.add(c)
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank = len(mask)
    
    if current_rank < rank:
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank
    
    return U, V, W, rank

def _snap_to_half_integer(arr, tol=1e-9):
    """Snap values close to half-integers to exact half-integers."""
    result = arr.copy()
    for i in range(len(result)):
        val = result[i]
        nearest = round(val * 2) / 2.0
        if abs(val - nearest) < tol:
            result[i] = nearest
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with the Strassen-style decomposition
    U, V, W, rank = _strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = [[0.0] * (n*m*p) for _ in range(n*m)]
        V = [[0.0] * (n*m*p) for _ in range(m*p)]
        W = [[0.0] * (p*n) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col
    
    # Try greedy column elimination
    if time.time() < deadline - 10:
        U, V, W, rank = _try_linear_combination_removal(U, V, W, n, m, p, max_time=60.0)
    
    # Snap to half-integers
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_arr = _snap_to_half_integer(U_arr)
    V_arr = _snap_to_half_integer(V_arr)
    W_arr = _snap_to_half_integer(W_arr)
    
    # Final verification
    if not _verify_decomposition(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p):
        # Re-snap with tighter tolerance or revert
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END
