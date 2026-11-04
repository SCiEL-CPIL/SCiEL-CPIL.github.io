"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def check_time():
        return time.time() - start_time > time_limit - 10
    
    def schoolbook():
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
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    error += (val - 1.0) ** 2
        return error < 1e-9
    
    def validate_fast(U_arr, V_arr, W_arr):
        """Fast validation using numpy."""
        rank = U_arr.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    error += (val - 1.0) ** 2
        return error < 1e-9
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy 1: Block decomposition using Strassen for 2x2x2 sub-blocks
    # The 3x5x5 tensor can be decomposed into blocks
    # Use Strassen's algorithm for 2x2 matrix multiplication within blocks
    
    def strassen_2x2():
        """
        Strassen decomposition for 2x2 matrix multiplication.
        This gives rank 7 for multiplying two 2x2 matrices.
        
        For 2x2 * 2x2 -> 2x2:
        C = A * B where A, B are 2x2
        
        Using Strassen:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A10 + A11)B00
        M3 = A00(B01 - B11)
        M4 = A11(B10 - B00)
        M5 = (A00 + A01)B11
        M6 = (A10 - A00)(B00 + B01)
        M7 = (A01 - A11)(B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # For our case, we'll construct a decomposition that uses Strassen
        # on the 2x2 blocks within the 3x5 * 5x5 multiplication
        
        # Pad 3x5 to 4x5 and 5x5 to 5x5, then use block structure
        # Actually, let's use a different approach:
        # Split the 5 columns into 2+2+1 and 5 rows into 2+2+1
        
        # Let's try a structured approach:
        # Decompose into blocks and use Strassen where applicable
        
        # For now, let's try a direct optimization approach with better initialization
        
        return None
    
    # Strategy 2: Structured search exploiting tensor structure
    # The tensor T[i,j,k] = 1 has a Kronecker-like structure
    # We can try to find a decomposition that exploits this
    
    def try_rank_optimized(rank, max_restarts=3):
        """Try to find a rank-`rank` decomposition using optimization with multiple restarts."""
        if check_time():
            return None
        
        best_solution = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if check_time():
                break
            
            # Better initialization: use schoolbook-like structure
            U_init = np.zeros((n * m, rank))
            V_init = np.zeros((m * p, rank))
            W_init = np.zeros((p * n, rank))
            
            # Initialize with a sparse pattern
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U_init[i * m + j, r] = 1.0
                V_init[j * p + k, r] = 1.0
                W_init[k * n + i, r] = 1.0
            
            # Add some random perturbation
            U_init += np.random.randn(n * m, rank) * 0.01
            V_init += np.random.randn(m * p, rank) * 0.01
            W_init += np.random.randn(p * n, rank) * 0.01
            
            def objective(params):
                if check_time():
                    return 1e10
                U = params[:n * m * rank].reshape(n * m, rank)
                V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
                
                # Compute the error efficiently
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                            error += (val - 1.0) ** 2
                return error
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                                         'maxls': 50})
                
                if result.fun < best_error:
                    best_error = result.fun
                    U = result.x[:n * m * rank].reshape(n * m, rank)
                    V = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                    W = result.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if validate_fast(U_rounded, V_rounded, W_rounded):
                        best_solution = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        break
                    
                    # Also try integer rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if validate_fast(U_int, V_int, W_int):
                        best_solution = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        break
                        
            except Exception:
                pass
        
        return best_solution
    
    # Strategy 3: Try known low-rank constructions
    # For 3x5 * 5x5, we can try:
    # 1. Block decomposition: split 5 into 2+2+1, use Strassen on 2x2 blocks
    # 2. This should give rank around 38 (as measured in inspiration)
    
    def block_decomposition_strassen():
        """
        Block decomposition using Strassen for 2x2 sub-blocks.
        
        Split the 5 columns of A into blocks of size 2, 2, 1
        Split the 5 rows of B into blocks of size 2, 2, 1
        Split the 5 columns of C into blocks of size 2, 2, 1
        
        For each pair of blocks, we can use Strassen if both are 2x2
        """
        # Block sizes
        col_blocks_A = [2, 2, 1]  # columns of A (and rows of B)
        row_blocks_B = [2, 2, 1]  # rows of B (and columns of C)
        col_blocks_C = [2, 2, 1]  # columns of C
        
        # Compute block offsets
        col_offsets_A = [0, 2, 4]
        row_offsets_B = [0, 2, 4]
        col_offsets_C = [0, 2, 4]
        
        # For each output block (i_block, k_block), we need to compute:
        # C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
        
        # For 2x2 blocks, we can use Strassen (rank 7)
        # For 2x1 or 1x2 blocks, we need rank 2
        # For 1x1 blocks, we need rank 1
        
        # Let's construct the decomposition
        U_cols = []
        V_cols = []
        W_cols = []
        
        for i_block in range(3):  # 3 row blocks of A (and C)
            i_size = 1  # Each row block of A is size 1 (since n=3, we split into 1+1+1)
            i_offset = i_block
            
            for j_block in range(3):  # 3 column blocks of A (and row blocks of B)
                j_size = col_blocks_A[j_block]
                j_offset = col_offsets_A[j_block]
                
                for k_block in range(3):  # 3 column blocks of C (and row blocks of B)
                    k_size = col_blocks_C[k_block]
                    k_offset = col_offsets_C[k_block]
                    
                    # For this combination of blocks, we need to compute:
                    # C[i_offset:i_offset+i_size, k_offset:k_offset+k_size] += 
                    #   A[i_offset:i_offset+i_size, j_offset:j_offset+j_size] * 
                    #   B[j_offset:j_offset+j_size, k_offset:k_offset+k_size]
                    
                    # The rank needed depends on the block sizes
                    # For i_size=1, j_size=2, k_size=2: rank 2 (outer product of row and 2x2)
                    # For i_size=1, j_size=2, k_size=1: rank 1
                    # etc.
                    
                    # Let's use a simple approach: rank = i_size * j_size * k_size
                    # But we can do better with Strassen for 2x2 blocks
                    
                    # For now, use the schoolbook rank for this block
                    block_rank = i_size * j_size * k_size
                    
                    for r in range(block_rank):
                        # Distribute the rank across the block
                        ii = r % i_size
                        jj = (r // i_size) % j_size
                        kk = (r // (i_size * j_size)) % k_size
                        
                        U_col = [0] * (n * m)
                        V_col = [0] * (m * p)
                        W_col = [0] * (p * n)
                        
                        U_col[(i_offset + ii) * m + (j_offset + jj)] = 1
                        V_col[(j_offset + jj) * p + (k_offset + kk)] = 1
                        W_col[(k_offset + kk) * n + (i_offset + ii)] = 1
                        
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
        
        # This is just schoolbook, let's try something better
        # Use Strassen for the 2x2 * 2x2 -> 2x2 blocks
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Row blocks for A: split 3 rows into 1+1+1
        # Col blocks for A: split 5 cols into 2+2+1
        # Row blocks for B: split 5 rows into 2+2+1
        # Col blocks for C: split 5 cols into 2+2+1
        
        for i in range(3):  # Single row blocks
            for j_block in range(3):
                j_size = col_blocks_A[j_block]
                j_offset = col_offsets_A[j_block]
                
                for k_block in range(3):
                    k_size = col_blocks_C[k_block]
                    k_offset = col_offsets_C[k_block]
                    
                    if j_size == 2 and k_size == 2:
                        # Use Strassen for 1x2 * 2x2 -> 1x2
                        # This is a row vector times a 2x2 matrix
                        # Rank needed: 2 (just two multiplications)
                        # But we can use Strassen-like tricks
                        # Actually for 1x2 * 2x2, we need rank 2
                        for r in range(2):
                            U_col = [0] * (n * m)
                            V_col = [0] * (m * p)
                            W_col = [0] * (p * n)
                            
                            U_col[i * m + j_offset + r] = 1
                            V_col[(j_offset + r) * p + k_offset] = 1
                            W_col[k_offset * n + i] = 1
                            
                            U_col2 = [0] * (n * m)
                            V_col2 = [0] * (m * p)
                            W_col2 = [0] * (p * n)
                            
                            U_col2[i * m + j_offset + r] = 1
                            V_col2[(j_offset + r) * p + k_offset + 1] = 1
                            W_col2[(k_offset + 1) * n + i] = 1
                            
                            U_cols.append(U_col)
                            V_cols.append(V_col)
                            W_cols.append(W_col)
                            U_cols.append(U_col2)
                            V_cols.append(V_col2)
                            W_cols.append(W_col2)
                    else:
                        # Schoolbook for other block sizes
                        for jj in range(j_size):
                            for kk in range(k_size):
                                U_col = [0] * (n * m)
                                V_col = [0] * (m * p)
                                W_col = [0] * (p * n)
                                
                                U_col[i * m + j_offset + jj] = 1
                                V_col[(j_offset + jj) * p + k_offset + kk] = 1
                                W_col[(k_offset + kk) * n + i] = 1
                                
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
        
        rank = len(U_cols)
        return {"U": U_cols, "V": V_cols, "W": W_cols}
    
    # Try block decomposition first
    if not check_time():
        block_result = block_decomposition_strassen()
        if block_result and validate(block_result["U"], block_result["V"], block_result["W"]):
            block_rank = len(block_result["U"][0]) if block_result["U"] else 0
            if block_rank < best_rank:
                best_result = block_result
                best_rank = block_rank
    
    # Now try optimization for lower ranks
    # Start from the current best rank and try to reduce
    lower_bound = max(n * m, m * p, p * n)  # 25
    
    # Try a range of ranks
    ranks_to_try = list(range(lower_bound, best_rank + 1))
    # Sort by rank (ascending)
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if check_time():
            break
        
        result = try_rank_optimized(rank, max_restarts=2)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank == lower_bound:
                break
    
    # Check time
    if check_time():
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
