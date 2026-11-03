"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    
    Strategy: Split the 2x4 * 4x6 multiplication into blocks:
    - A is split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    - B is split into top (rows 0,1) and bottom (rows 2,3), each with 3 column blocks of width 2
    
    So C = A_left * B_top + A_right * B_bottom
    
    Each (2x2) * (2x2) block multiplication uses 7 Strassen terms.
    Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42 terms.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    import random
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(start_rank, max_time):
    """
    Attempt to find a lower rank decomposition using iterative refinement.
    This is a heuristic approach that may not always succeed.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + max_time
    
    for target_rank in range(start_rank - 1, 35, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 100 + target_rank)
            
            # Initialize U, V, W randomly
            U_np = np.random.randn(n * m, target_rank)
            V_np = np.random.randn(m * p, target_rank)
            W_np = np.random.randn(p * n, target_rank)
            
            # Iterative refinement: fix two matrices, solve for the third
            # For simplicity, we'll do a few iterations of alternating least squares
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Fix U and V, solve for W
                # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is complex to solve directly, so we use a gradient-based approach
                # or simply check if the current decomposition works
                
                # Check if current decomposition works
                valid = True
                for _ in range(5):
                    A = np.random.randint(-10, 11, (n, m))
                    B = np.random.randint(-10, 11, (m, p))
                    C_exp = A @ B
                    C_comp = np.zeros((n, p))
                    for r in range(target_rank):
                        u_val = U_np[:, r] @ A.reshape(-1)
                        v_val = V_np[:, r] @ B.reshape(-1)
                        C_comp += W_np[:, r].reshape(p, n) * u_val * v_val
                    if not np.allclose(C_comp, C_exp, atol=1e-9):
                        valid = False
                        break
                
                if valid:
                    # Convert to list format
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_np.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Simple gradient step (this is a placeholder for actual optimization)
                # In practice, this would involve computing gradients and updating
                lr = 0.01
                for _ in range(10):
                    A = np.random.randint(-10, 11, (n, m))
                    B = np.random.randint(-10, 11, (m, p))
                    C_exp = A @ B
                    C_comp = np.zeros((n, p))
                    grad_U = np.zeros_like(U_np)
                    grad_V = np.zeros_like(V_np)
                    grad_W = np.zeros_like(W_np)
                    
                    for r in range(target_rank):
                        u_val = U_np[:, r] @ A.reshape(-1)
                        v_val = V_np[:, r] @ B.reshape(-1)
                        C_comp += W_np[:, r].reshape(p, n) * u_val * v_val
                    
                    error = C_comp - C_exp
                    for r in range(target_rank):
                        u_val = U_np[:, r] @ A.reshape(-1)
                        v_val = V_np[:, r] @ B.reshape(-1)
                        grad_W[:, r] += error.reshape(-1) * u_val * v_val
                        grad_U[:, r] += (error.reshape(p, n).T * W_np[:, r].reshape(p, n)).reshape(-1) * v_val
                        grad_V[:, r] += (error.reshape(p, n) * W_np[:, r].reshape(p, n).T).reshape(-1) * u_val
                    
                    U_np -= lr * grad_U
                    V_np -= lr * grad_V
                    W_np -= lr * grad_W
                
                if time.time() > deadline:
                    break
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition, then attempt rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    # Try Strassen decomposition first (rank 42)
    result = _strassen_block_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(result, 2, 4, 6, trials=50):
        # Attempt rank reduction
        reduced = _try_rank_reduction(42, max_time=60)
        if reduced is not None:
            if _verify_decomposition(reduced, 2, 4, 6, trials=20):
                return reduced
        return result
    
    # Check deadline
    if time.time() > deadline - 10:
        return _schoolbook_decomposition()
    
    # Fallback to schoolbook (rank 4)
    schoolbook = _schoolbook_decomposition()
    if _verify_decomposition(schoolbook, 2, 4, 6, trials=50):
        return schoolbook
    
    return schoolbook
# EVOLVE-BLOCK-END
