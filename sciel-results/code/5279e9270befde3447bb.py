"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is a dict with keys 'U', 'V', 'W' mapping (i,j) -> coeff.
    For A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A@B:
    C[0,0] = ae + bf
    C[0,1] = af + bg
    C[1,0] = ce + df
    C[1,1] = cf + dg
    
    Strassen's 7 multiplications:
    m1 = (a+d)(e+h)
    m2 = (c+d)(e)
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)(h)
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    And:
    C[0,0] = m1 - m4 + m5 + m6
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m3 + m7
    """
    terms = []
    
    # Term 1: m1 = (a+d)(e+h)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # Term 2: m2 = (c+d)(e)
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # Term 3: m3 = a(f-h)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1}
    })
    
    # Term 4: m4 = d(g-e)
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # Term 5: m5 = (a+b)(h)
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): 1, (0,1): 1}
    })
    
    # Term 6: m6 = (c-a)(e+f)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1}
    })
    
    # Term 7: m7 = (b-d)(g+h)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(1,1): 1}
    })
    
    return terms

def _verify_strassen_2x2():
    """Verify that the Strassen terms correctly compute 2x2 matrix multiplication."""
    terms = _strassen_2x2_terms()
    n = m = p = 2
    
    # Test with random matrices
    for _ in range(100):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for term in terms:
            alpha = 0
            for (i, j), coeff in term['U'].items():
                alpha += coeff * A[i, j]
            
            beta = 0
            for (j, k), coeff in term['V'].items():
                beta += coeff * B[j, k]
            
            for (i, k), coeff in term['W'].items():
                C_computed[i, k] += alpha * beta * coeff
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _construct_block_decomposition(n, m, p, deadline):
    """
    Construct a decomposition using Strassen for 2x2 blocks.
    
    For n=2, m=4, p=6:
    Split A (2x4) into two 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    Split B (4x6) into six 2x2 blocks:
      B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
      B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
    
    C = A @ B
    C[0:2, 0:2] = A0 @ B[0:2,0:2] + A1 @ B[2:4,0:2]
    C[0:2, 2:4] = A0 @ B[0:2,2:4] + A1 @ B[2:4,2:4]
    C[0:2, 4:6] = A0 @ B[0:2,4:6] + A1 @ B[2:4,4:6]
    
    For each 2x2 @ 2x2 multiplication, use Strassen (rank 7)
    Total: 6 * 7 = 42
    """
    n, m, p = 2, 4, 6
    
    # A is 2x4, B is 4x6, C is 2x6
    strassen_terms = _strassen_2x2_terms()
    
    # Build U, V, W matrices
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each of the 6 block multiplications, add 7 Strassen terms
    for a_block in range(2):
        for b_block in range(3):
            for t_idx, term in enumerate(strassen_terms):
                r = a_block * 21 + b_block * 7 + t_idx
                
                # U: which entries of A are used
                for (i, j_local), coeff in term['U'].items():
                    j_global = a_block * 2 + j_local
                    U[i * m + j_global][r] += coeff
                
                # V: which entries of B are used
                for (j_local, k_local), coeff in term['V'].items():
                    j_global = a_block * 2 + j_local
                    k_global = b_block * 2 + k_local
                    V[j_global * p + k_global][r] += coeff
                
                # W: which entry of C
                for (i, k_local), coeff in term['W'].items():
                    k_global = b_block * 2 + k_local
                    W[k_global * n + i][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, max_samples=100):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for _ in range(max_samples):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += U[i * m + j][r] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += V[j * p + k][r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += alpha[i] * beta[k] * W[k * n + i][r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies
    among the columns of the factor matrices.
    """
    # Convert to numpy arrays for easier manipulation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    
    # Try to find if we can remove some columns
    # This is a simplified approach - in practice, this would require
    # more sophisticated tensor rank reduction algorithms
    
    # For now, just verify the current decomposition is valid
    if _verify_decomposition(U, V, W, n, m, p, max_samples=50):
        return U, V, W
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try the block Strassen decomposition (rank 42)
    result = _construct_block_decomposition(n, m, p, deadline)
    
    # Verify it
    if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, max_samples=50):
        # Try rank reduction
        U, V, W = _try_rank_reduction(result['U'], result['V'], result['W'], n, m, p, deadline)
        result = {"U": U, "V": V, "W": W}
        return result
    
    # If verification fails, fall back to schoolbook (rank 48)
    result = _schoolbook_decomposition(n, m, p)
    return result

# EVOLVE-BLOCK-END
