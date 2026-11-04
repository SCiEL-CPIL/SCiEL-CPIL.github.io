import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook_2x3x6():
    """
    Schoolbook matrix multiplication tensor decomposition for 2x3x6.
    Rank = n * m * p = 36.
    
    U shape: (n*m, rank) = (6, 36)
    V shape: (m*p, rank) = (18, 36)
    W shape: (p*n, rank) = (12, 36)
    
    Index conventions:
      U[i*m + j, r]
      V[j*p + k, r]
      W[k*n + i, r]
    
    The tensor decomposition must satisfy:
      sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    for all i, k, j, l.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2_block(U_base, V_base, W_base, i0, k0, rank_offset, n, m, p):
    """
    Add a Strassen 2x2x2 block to the decomposition.
    i0: row offset in A (2 rows: i0, i0+1)
    k0: col offset in B (2 cols: k0, k0+1)
    
    Strassen products:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
    Output:
      C11 = M1 + M4 - M5 - M7
      C12 = M3 + M5
      C21 = M2 + M7
      C22 = M1 - M2 + M3 + M4
    """
    # U factors for A submatrix (2x2): rows i0, i0+1; cols 0,1
    # V factors for B submatrix (2x2): rows 0,1; cols k0, k0+1
    # W factors for C submatrix (2x2): rows k0, k0+1; cols i0, i0+1
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11 + A22 -> row i0: 1, row i0+1: 1
    # V: B11 + B22 -> row 0: 1, row 1: 1
    # W: C11 + C22 -> (k0, i0): 1, (k0+1, i0+1): 1
    U_base[i0 * m + 0][rank_offset] = 1
    U_base[(i0 + 1) * m + 1][rank_offset] = 1
    V_base[0 * p + k0][rank_offset] = 1
    V_base[1 * p + (k0 + 1)][rank_offset] = 1
    W_base[k0 * n + i0][rank_offset] = 1
    W_base[(k0 + 1) * n + (i0 + 1)][rank_offset] = 1
    
    # M2 = (A21+A22)B11
    # U: A21 + A22 -> row i0+1: col 1: 1, col 2: 1
    # V: B11 -> row 0, col k0
    # W: C21 -> (k0, i0+1)
    U_base[(i0 + 1) * m + 1][rank_offset + 1] = 1
    U_base[(i0 + 1) * m + 2][rank_offset + 1] = 1
    V_base[0 * p + k0][rank_offset + 1] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 1] = 1
    
    # M3 = A11(B12-B22)
    # U: A11 -> row i0, col 0
    # V: B12 - B22 -> row 0, col k0+1: 1; row 1, col k0+1: -1
    # W: C12 -> (k0+1, i0)
    U_base[i0 * m + 0][rank_offset + 2] = 1
    V_base[0 * p + (k0 + 1)][rank_offset + 2] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 2] = -1
    W_base[(k0 + 1) * n + i0][rank_offset + 2] = 1
    
    # M4 = A22(B21-B11)
    # U: A22 -> row i0+1, col 1
    # V: B21 - B11 -> row 1, col k0: 1; row 0, col k0: -1
    # W: C11 -> (k0, i0)
    U_base[(i0 + 1) * m + 1][rank_offset + 3] = 1
    V_base[1 * p + k0][rank_offset + 3] = 1
    V_base[0 * p + k0][rank_offset + 3] = -1
    W_base[k0 * n + i0][rank_offset + 3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11 + A12 -> row i0: col 0: 1, col 1: 1
    # V: B22 -> row 1, col k0+1
    # W: C11 -> (k0, i0) with -1; C12 -> (k0+1, i0) with 1
    U_base[i0 * m + 0][rank_offset + 4] = 1
    U_base[i0 * m + 1][rank_offset + 4] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 4] = 1
    W_base[k0 * n + i0][rank_offset + 4] = -1
    W_base[(k0 + 1) * n + i0][rank_offset + 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21 - A11 -> row i0+1, col 1: 1; row i0, col 0: -1
    # V: B11 + B12 -> row 0, col k0: 1; row 0, col k0+1: 1
    # W: C21 -> (k0, i0+1)
    U_base[(i0 + 1) * m + 1][rank_offset + 5] = 1
    U_base[i0 * m + 0][rank_offset + 5] = -1
    V_base[0 * p + k0][rank_offset + 5] = 1
    V_base[0 * p + (k0 + 1)][rank_offset + 5] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12 - A22 -> row i0, col 1: 1; row i0+1, col 1: -1
    # V: B21 + B22 -> row 1, col k0: 1; row 1, col k0+1: 1
    # W: C21 -> (k0, i0+1) with 1; C11 -> (k0, i0) with -1
    U_base[i0 * m + 1][rank_offset + 6] = 1
    U_base[(i0 + 1) * m + 1][rank_offset + 6] = -1
    V_base[1 * p + k0][rank_offset + 6] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 6] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 6] = 1
    W_base[k0 * n + i0][rank_offset + 6] = -1


def _assemble_strassen():
    """
    Assemble rank-33 decomposition for 2x3x6:
    Split the middle dimension (3) into 2+1:
    - 2x2x6 part: A[:,0:2] * B[0:2,:] -> split 6 cols into three 2-col blocks, each 2x2x2 Strassen (rank 7) -> 21
    - 2x1x6 part: A[:,2] * B[2,:] -> schoolbook -> 12
    Total rank: 21 + 12 = 33
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * rank for _ in range(p * n)]  # (12, 33)
    
    # Part 1: 2x2x6 using three Strassen 2x2x2 blocks
    # A rows 0-1, cols 0-1; B rows 0-1, cols k0..k0+1; C rows k0..k0+1, cols 0-1
    for block in range(3):
        k0 = block * 2
        ro = block * 7  # rank offset
        
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        U[0 * m + 0][ro] = 1      # A[0,0]
        U[1 * m + 1][ro] = 1      # A[1,1]
        V[0 * p + k0][ro] = 1     # B[0,k0]
        V[1 * p + (k0+1)][ro] = 1 # B[1,k0+1]
        W[k0 * n + 0][ro] = 1     # C[k0,0]
        W[(k0+1) * n + 1][ro] = 1 # C[k0+1,1]
        
        # M2 = (A21+A22)B11 -> C21(+1), C22(-1)
        U[1 * m + 1][ro+1] = 1    # A[1,1]
        U[1 * m + 0][ro+1] = 1    # A[1,0]
        V[0 * p + k0][ro+1] = 1   # B[0,k0]
        W[k0 * n + 1][ro+1] = 1   # C[k0,1]
        W[(k0+1) * n + 1][ro+1] = -1 # C[k0+1,1]
        
        # M3 = A11(B12-B22) -> C12(+1), C22(+1)
        U[0 * m + 0][ro+2] = 1    # A[0,0]
        V[0 * p + (k0+1)][ro+2] = 1  # B[0,k0+1]
        V[1 * p + (k0+1)][ro+2] = -1 # B[1,k0+1]
        W[(k0+1) * n + 0][ro+2] = 1  # C[k0+1,0]
        W[(k0+1) * n + 1][ro+2] = 1  # C[k0+1,1]
        
        # M4 = A22(B21-B11) -> C11
        U[1 * m + 1][ro+3] = 1    # A[1,1]
        V[1 * p + k0][ro+3] = 1   # B[1,k0]
        V[0 * p + k0][ro+3] = -1  # B[0,k0]
        W[k0 * n + 0][ro+3] = 1   # C[k0,0]
        
        # M5 = (A11+A12)B22 -> -C11 + C12
        U[0 * m + 0][ro+4] = 1    # A[0,0]
        U[0 * m + 1][ro+4] = 1    # A[0,1]
        V[1 * p + (k0+1)][ro+4] = 1 # B[1,k0+1]
        W[k0 * n + 0][ro+4] = -1  # C[k0,0]
        W[(k0+1) * n + 0][ro+4] = 1 # C[k0+1,0]
        
        # M6 = (A21-A11)(B11+B12) -> C21
        U[1 * m + 0][ro+5] = 1    # A[1,0]
        U[0 * m + 0][ro+5] = -1   # A[0,0]
        V[0 * p + k0][ro+5] = 1   # B[0,k0]
        V[0 * p + (k0+1)][ro+5] = 1 # B[0,k0+1]
        W[k0 * n + 1][ro+5] = 1   # C[k0,1]
        
        # M7 = (A12-A22)(B21+B22) -> C21 - C11
        U[0 * m + 1][ro+6] = 1    # A[0,1]
        U[1 * m + 1][ro+6] = -1   # A[1,1]
        V[1 * p + k0][ro+6] = 1   # B[1,k0]
        V[1 * p + (k0+1)][ro+6] = 1 # B[1,k0+1]
        W[k0 * n + 1][ro+6] = 1   # C[k0,1]
        W[k0 * n + 0][ro+6] = -1  # C[k0,0]
    
    # Part 2: 2x1x6 schoolbook: A[:,2] * B[2,:]
    # C[i,k] += A[i,2] * B[2,k]
    # U[i*m+2, r] = A[i,2], V[2*p+k, r] = B[2,k], W[k*n+i, r] = C[k,i]
    for i in range(n):
        for k in range(p):
            r = 21 + i * p + k
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_full(U, V, W):
    """
    Verify that the tensor decomposition is correct.
    sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    """
    from fractions import Fraction
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[l * p + k][r]) * Fraction(W[k * n + i][r])
                    expected = Fraction(1) if j == l else Fraction(0)
                    if total != expected:
                        return False
    return True


def _verify_numpy(U, V, W, tol=1e-10):
    """
    Fast verification using numpy. Returns True if the decomposition is correct within tolerance.
    """
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i, k, j, l), check sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] == delta_{j,l}
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    # sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r]
                    prod = U_np[i * m + j] * V_np[l * p + k] * W_np[k * n + i]
                    total = np.sum(prod)
                    expected = 1.0 if j == l else 0.0
                    if abs(total - expected) > tol:
                        return False
    return True


def _try_reduce_rank(U, V, W, target_rank, time_limit=10.0):
    """
    Try to reduce the rank from len(U[0]) to target_rank by finding a linear combination
    of rank terms that can be eliminated. Uses randomized local search.
    """
    n, m, p = DIMENSIONS
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return None
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # We need to find coefficients c_r such that for all r, we can express
    # the r-th term as a combination of others, effectively reducing rank.
    # Strategy: try to find a non-trivial linear dependency among the rank-1 tensors.
    
    # Each rank-1 term is defined by (U[:,r], V[:,r], W[:,r])
    # We want to find c_0, ..., c_{R-1} not all zero such that
    # sum_r c_r * U[:,r] * V[:,r] * W[:,r] = 0 (as a tensor)
    
    # This is equivalent to: for all (i, k, j, l),
    # sum_r c_r * U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = 0
    
    # Let's set up a linear system. For each (i, k, j, l), we get one equation.
    # Number of equations: n * p * m * m = 2 * 6 * 3 * 3 = 108
    # Number of unknowns: current_rank
    
    # Build the matrix A where A[(i,k,j,l), r] = U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r]
    eq_count = n * p * m * m
    A = np.zeros((eq_count, current_rank))
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    for r in range(current_rank):
                        A[idx, r] = U_np[i * m + j, r] * V_np[l * p + k, r] * W_np[k * n + i, r]
                    idx += 1
    
    # Find null space of A. If null space dimension >= (current_rank - target_rank + 1),
    # we can potentially reduce rank.
    # Actually, if there's a non-trivial null space vector c, then
    # sum_r c_r * (term r) = 0, meaning the terms are linearly dependent.
    # We can use this to eliminate one term.
    
    start_time = time.time()
    best_result = None
    
    # Try to find null space vectors
    try:
        # SVD to find null space
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
        # Null space vectors are rows of Vt corresponding to zero singular values
        null_space = Vt[s < 1e-10]
        null_dim = len(null_space)
        
        if null_dim > 0:
            # We have null space vectors. Try to use them to reduce rank.
            # For each null space vector, we can potentially eliminate one term.
            for ns_vec in null_space:
                if time.time() - start_time > time_limit:
                    break
                # Find the index with largest absolute value in ns_vec
                max_idx = np.argmax(np.abs(ns_vec))
                if abs(ns_vec[max_idx]) < 1e-10:
                    continue
                
                # We want to eliminate term max_idx.
                # New coefficients for other terms: c_r / c_{max_idx}
                new_U = np.zeros((n * m, current_rank - 1))
                new_V = np.zeros((m * p, current_rank - 1))
                new_W = np.zeros((p * n, current_rank - 1))
                
                col_idx = 0
                for r in range(current_rank):
                    if r == max_idx:
                        continue
                    # The contribution of term r in the null space is ns_vec[r]
                    # To cancel term max_idx, we add (ns_vec[r] / ns_vec[max_idx]) * term_r
                    # But this changes the decomposition. We need to be more careful.
                    # Actually, the null space relation means:
                    # sum_r ns_vec[r] * (U_r, V_r, W_r) = 0
                    # So term max_idx = -sum_{r != max_idx} (ns_vec[r] / ns_vec[max_idx]) * term_r
                    # We can remove term max_idx and adjust the others.
                    # But the adjustment is not straightforward for the individual factors.
                    
                    # For now, just keep the original values for non-eliminated terms
                    new_U[:, col_idx] = U_np[:, r]
                    new_V[:, col_idx] = V_np[:, r]
                    new_W[:, col_idx] = W_np[:, r]
                    col_idx += 1
                
                # Verify the new decomposition
                new_U_list = new_U.tolist()
                new_V_list = new_V.tolist()
                new_W_list = new_W.tolist()
                
                if _verify_numpy(new_U_list, new_V_list, new_W_list, tol=1e-8):
                    # Check if rank is reduced
                    if new_U.shape[1] == target_rank:
                        # Convert to list format
                        result = {
                            "U": [[float(x) for x in row] for row in new_U],
                            "V": [[float(x) for x in row] for row in new_V],
                            "W": [[float(x) for x in row] for row in new_W]
                        }
                        if best_result is None or result["U"][0] is not None:
                            best_result = result
    except Exception:
        pass
    
    return best_result


def _random_search_lower_rank(U, V, W, max_rank, time_limit=30.0):
    """
    Search for a lower-rank decomposition by random perturbation and rank reduction.
    """
    n, m, p = DIMENSIONS
    current_rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    start_time = time.time()
    best_result = None
    best_rank = current_rank
    
    # Try to find a decomposition of rank max_rank by random initialization and optimization
    for target_rank in range(max_rank, min(current_rank, max_rank + 1)):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for trial in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random values
            np.random.seed(42 + trial * 1000 + target_rank)
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            W_rand = np.random.randn(p * n, target_rank) * 0.5
            
            # Verify
            if _verify_numpy(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), tol=1e-8):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_rand.tolist()
                    }
                break  # Found a valid decomposition for this target_rank
    
    return best_result


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # Start with the known rank-33 Strassen baseline
    try:
        result = _assemble_strassen()
        if _verify_full(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
        else:
            best_result = None
            best_rank = 36
    except Exception:
        best_result = None
        best_rank = 36
    
    if best_result is None:
        # Fallback to schoolbook
        try:
            fb = _schoolbook_2x3x6()
            if _verify_full(fb["U"], fb["V"], fb["W"]):
                best_result = fb
                best_rank = len(fb["U"][0])
        except Exception:
            pass
        if best_result is None:
            return _schoolbook_2x3x6()
    
    # Use remaining time to search for lower-rank decomposition
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try to reduce rank from best_rank
        for target_rank in range(best_rank - 1, 25, -1):
            if time.time() > deadline - 10.0:
                break
            time_left = deadline - time.time()
            if time_left < 2.0:
                break
            
            # Try rank reduction via null space analysis
            try:
                reduced = _try_reduce_rank(best_result["U"], best_result["V"], best_result["W"], 
                                          target_rank, time_limit=min(5.0, time_left * 0.3))
                if reduced is not None:
                    # Verify with exact arithmetic
                    if _verify_full(reduced["U"], reduced["V"], reduced["W"]):
                        best_result = reduced
                        best_rank = target_rank
                        continue
            except Exception:
                pass
            
            # Try random search for this target rank
            try:
                rand_result = _random_search_lower_rank(best_result["U"], best_result["V"], best_result["W"],
                                                       target_rank, time_limit=min(3.0, time_left * 0.2))
                if rand_result is not None:
                    if _verify_full(rand_result["U"], rand_result["V"], rand_result["W"]):
                        best_result = rand_result
                        best_rank = target_rank
                        continue
            except Exception:
                pass
    
    return best_result

# EVOLVE-BLOCK-END
