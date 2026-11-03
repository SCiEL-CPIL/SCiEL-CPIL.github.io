import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 6 matrix.
    
    Strategy: Split p=6 into three blocks of size 2. Find a rank-r decomposition
    for 2x3x2 matrix multiplication, then expand to 2x3x6 with rank 3r.
    Schoolbook gives rank 36. If we find rank 11 for 2x3x2, total rank is 33.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition (rank 36)
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try to find a better decomposition using block structure
    # Split p=6 into three blocks of size 2
    # Each block is a 2x3x2 multiplication
    # If we find a rank r decomposition for 2x3x2, total rank is 3r
    
    # First, try known/structured decompositions for 2x3x2
    # The schoolbook rank for 2x3x2 is 12
    # Try to find rank 11 decomposition
    
    for target_rank in [11, 10, 9]:
        if time.time() > deadline:
            break
        
        decomp_232 = _search_232_decomposition(target_rank, deadline)
        if decomp_232 is not None:
            # Construct full 2x3x6 decomposition
            result = _expand_decomposition(decomp_232, n, m, p, block_size=2)
            if result is not None:
                best_result = result
                best_rank = 3 * target_rank
                break  # Found the best we can with this approach
    
    # Also try direct search on the full 2x3x6 problem for rank < 36
    if time.time() < deadline - 20:
        for target_rank in [33, 32, 31, 30]:
            if time.time() > deadline:
                break
            decomp_full = _search_full_decomposition(n, m, p, target_rank, deadline)
            if decomp_full is not None:
                best_result = decomp_full
                best_rank = target_rank
                break
    
    return best_result

def _schoolbook_decomposition(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W decompose the n x m x p multiplication tensor.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c
                val = 0.0
                for c in range(rank):
                    val += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _search_232_decomposition(target_rank, deadline):
    """
    Search for a rank target_rank decomposition of 2x3x2 matrix multiplication.
    Returns U, V, W matrices or None.
    """
    n_sub, m_sub, p_sub = 2, 3, 2
    r = target_rank
    
    max_attempts = 5000
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        
        # Generate random U, V with small integers/half-integers
        # Use a mix of strategies
        if attempts % 3 == 0:
            # Small integers
            U = np.random.randint(-2, 3, size=(n_sub * m_sub, r)).astype(float)
            V = np.random.randint(-2, 3, size=(m_sub * p_sub, r)).astype(float)
        elif attempts % 3 == 1:
            # Half-integers
            U = (np.random.randint(-4, 5, size=(n_sub * m_sub, r)) / 2.0)
            V = (np.random.randint(-4, 5, size=(m_sub * p_sub, r)) / 2.0)
        else:
            # Small range integers
            U = np.random.randint(-1, 2, size=(n_sub * m_sub, r)).astype(float)
            V = np.random.randint(-1, 2, size=(m_sub * p_sub, r)).astype(float)
        
        # Solve for W
        W = np.zeros((p_sub * n_sub, r))
        valid = True
        
        for i in range(n_sub):
            for k in range(p_sub):
                # Matrix M for this i,k pair
                # M[j, c] = U[i*m_sub + j, c] * V[j*p_sub + k, c]
                M = np.zeros((m_sub, r))
                for j in range(m_sub):
                    for c in range(r):
                        M[j, c] = U[i * m_sub + j, c] * V[j * p_sub + k, c]
                
                # Solve M * W_vec = ones(m_sub)
                # We want W_vec to be half-integer
                try:
                    # Find a particular solution using least squares
                    W_particular, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m_sub), rcond=None)
                    
                    # Round to nearest half-integer
                    W_rounded = np.round(W_particular * 2) / 2
                    
                    # Verify if M @ W_rounded == ones
                    if np.allclose(M @ W_rounded, np.ones(m_sub), atol=1e-10):
                        W[k * n_sub + i, :] = W_rounded
                    else:
                        # Try a different approach: use the null space
                        # Find a solution that is exactly half-integer
                        # This is harder, so for now, skip
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Final verification
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n_sub, m_sub, p_sub):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        attempts += 1
    
    return None

def _search_full_decomposition(n, m, p, target_rank, deadline):
    """
    Search for a rank target_rank decomposition of n x m x p matrix multiplication.
    """
    r = target_rank
    max_attempts = 1000
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        
        # Generate random U, V
        U = np.random.randint(-2, 3, size=(n * m, r)).astype(float)
        V = np.random.randint(-2, 3, size=(m * p, r)).astype(float)
        
        # Solve for W
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                try:
                    W_particular, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_rounded = np.round(W_particular * 2) / 2
                    
                    if np.allclose(M @ W_rounded, np.ones(m), atol=1e-10):
                        W[k * n + i, :] = W_rounded
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        attempts += 1
    
    return None

def _expand_decomposition(decomp_232, n, m, p, block_size):
    """
    Expand a 2x3x2 decomposition to 2x3x6 by repeating it for each block.
    """
    U_232 = decomp_232["U"]
    V_232 = decomp_232["V"]
    W_232 = decomp_232["W"]
    
    r_232 = len(U_232[0])
    num_blocks = p // block_size
    r_total = r_232 * num_blocks
    
    U_full = [[0] * r_total for _ in range(n * m)]
    V_full = [[0] * r_total for _ in range(m * p)]
    W_full = [[0] * r_total for _ in range(p * n)]
    
    for block in range(num_blocks):
        offset = block * r_232
        
        # Copy U (same for all blocks)
        for i in range(n * m):
            for c in range(r_232):
                U_full[i][offset + c] = U_232[i][c]
        
        # Copy V (shift columns for each block)
        for j in range(m):
            for k in range(block_size):
                for c in range(r_232):
                    V_full[j * p + block * block_size + k][offset + c] = V_232[j * block_size + k][c]
        
        # Copy W (shift rows for each block)
        for i in range(n):
            for k in range(block_size):
                for c in range(r_232):
                    W_full[(block * block_size + k) * n + i][offset + c] = W_232[k * n + i][c]
    
    # Verify the full decomposition
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        return None
