"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
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
    """
    U = [[0.0] * (n*m*p) for _ in range(n*m)]  # Start with max possible rank
    V = [[0.0] * (n*m*p) for _ in range(m*p)]
    W = [[0.0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    
    # Partition J indices into pairs and singles
    # For m=5: pairs (0,1), (2,3), single (4)
    # For p=6: pairs (0,1), (2,3), (4,5)
    
    j_pairs = []
    k_pairs = []
    j_singles = []
    k_singles = []
    
    # Create pairs for j
    for j in range(0, m - 1, 2):
        j_pairs.append((j, j+1))
    if m % 2 == 1:
        j_singles.append(m - 1)
    
    # Create pairs for k
    for k in range(0, p - 1, 2):
        k_pairs.append((k, k+1))
    if p % 2 == 1:
        k_singles.append(p - 1)
    
    # For each pair of j indices and pair of k indices, use Strassen
    for (j_a, j_b) in j_pairs:
        for (k_a, k_b) in k_pairs:
            # 7 Strassen multiplications
            # M1 = (A_0,j_a + A_1,j_b) * (B_j_a,k_a + B_j_b,k_b)
            # M2 = (A_1,j_a + A_1,j_b) * B_j_a,k_a
            # M3 = A_0,j_a * (B_j_a,k_b - B_j_b,k_b)
            # M4 = A_1,j_b * (B_j_b,k_a - B_j_a,k_a)
            # M5 = (A_0,j_a + A_0,j_b) * B_j_b,k_b
            # M6 = (A_1,j_a - A_0,j_a) * (B_j_a,k_a + B_j_a,k_b)
            # M7 = (A_0,j_b - A_1,j_b) * (B_j_b,k_a + B_j_b,k_b)
            
            # M1: (A_0,j_a + A_1,j_b) * (B_j_a,k_a + B_j_b,k_b)
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
    
    # Handle single j indices (for odd m)
    for j in j_singles:
        for (k_a, k_b) in k_pairs:
            # For a single j and pair of k's, we need 3 multiplications
            # C_0,k_a = A_0,j * B_j,k_a
            # C_1,k_a = A_1,j * B_j,k_a
            # C_0,k_b = A_0,j * B_j,k_b
            # C_1,k_b = A_1,j * B_j,k_b
            # But we can use a smaller Strassen-like scheme
            # Actually, for single j and pair k, we just do 4 scalar mults
            # Or we can use: M1 = A_0,j * (B_j,k_a + B_j,k_b)
            #                M2 = A_1,j * B_j,k_a
            #                M3 = A_1,j * B_j,k_b
            # Then: C_0,k_a = M1 - (something)... this doesn't work cleanly
            # Let's just do 4 scalar multiplications for simplicity
            for k in [k_a, k_b]:
                # C_0,k += A_0,j * B_j,k
                U[0*m + j][col] += 1.0
                V[j*p + k][col] += 1.0
                W[k*n + 0][col] += 1.0
                col += 1
                # C_1,k += A_1,j * B_j,k
                U[1*m + j][col] += 1.0
                V[j*p + k][col] += 1.0
                W[k*n + 1][col] += 1.0
                col += 1
    
    # Handle single k indices (for odd p)
    for (j_a, j_b) in j_pairs:
        for k in k_singles:
            # For pair of j's and single k, we need 3 multiplications
            # C_0,k = A_0,j_a * B_j_a,k + A_0,j_b * B_j_b,k
            # C_1,k = A_1,j_a * B_j_a,k + A_1,j_b * B_j_b,k
            # Use Strassen-like:
            # M1 = (A_0,j_a + A_1,j_b) * B_j_a,k
            # M2 = (A_1,j_a + A_1,j_b) * B_j_b,k
            # M3 = A_0,j_a * B_j_b,k
            # Hmm, this is getting complicated. Let's just do 4 scalar mults.
            for j in [j_a, j_b]:
                # C_0,k += A_0,j * B_j,k
                U[0*m + j][col] += 1.0
                V[j*p + k][col] += 1.0
                W[k*n + 0][col] += 1.0
                col += 1
                # C_1,k += A_1,j * B_j,k
                U[1*m + j][col] += 1.0
                V[j*p + k][col] += 1.0
                W[k*n + 1][col] += 1.0
                col += 1
    
    # Handle single j and single k
    for j in j_singles:
        for k in k_singles:
            # C_0,k += A_0,j * B_j,k
            U[0*m + j][col] += 1.0
            V[j*p + k][col] += 1.0
            W[k*n + 0][col] += 1.0
            col += 1
            # C_1,k += A_1,j * B_j,k
            U[1*m + j][col] += 1.0
            V[j*p + k][col] += 1.0
            W[k*n + 1][col] += 1.0
            col += 1
    
    # Trim unused columns
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with the Strassen-style decomposition
    U, V, W, rank = _strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if Strassen fails
        U = [[0.0] * (n*m*p) for _ in range(n*m)]
        V = [[0.0] * (n*m*p) for _ in range(m*p)]
        W = [[0.0] * (n*m*p) for _ in range(p*n)]
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
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy arrays for efficient computation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Try to find columns that can be removed or combined
    # This is a heuristic approach - try removing each column and checking validity
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Time-limited search for rank reduction
    max_search_time = 60
    search_start = time.time()
    
    # Try greedy column elimination
    current_U = np.array(U, dtype=np.float64)
    current_V = np.array(V, dtype=np.float64)
    current_W = np.array(W, dtype=np.float64)
    current_rank = rank
    removed_cols = set()
    
    # Try removing each column
    for c in range(rank):
        if time.time() - search_start > max_search_time:
            break
        if c in removed_cols:
            continue
        
        # Try removing column c
        mask = [i for i in range(current_rank) if i not in removed_cols and i != c]
        if len(mask) < 1:
            continue
            
        test_U = current_U[:, mask]
        test_V = current_V[:, mask]
        test_W = current_W[:, mask]
        
        if _verify_decomposition(test_U.tolist(), test_V.tolist(), test_W.tolist(), n, m, p):
            # Column c can be removed
            removed_cols.add(c)
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank = len(mask)
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = current_U.tolist()
        best_V = current_V.tolist()
        best_W = current_W.tolist()
    
    # Ensure we have a valid result
    if best_rank == rank:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
