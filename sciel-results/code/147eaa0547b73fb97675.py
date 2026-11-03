"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import sys

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term r has:
    - U[r]: vector of 4 coefficients for A_flat = [a, b, c, d] (row-major)
    - V[r]: vector of 4 coefficients for B_flat = [e, f, g, h] (row-major)
    - W[r]: 2x2 matrix (flat list of 4) of output weights
    
    The decomposition: C[i][j] = sum_r W[r][i*2+j] * (U[r] dot A_flat) * (V[r] dot B_flat)
    """
    # A_flat indices: 0=a, 1=b, 2=c, 3=d
    # B_flat indices: 0=e, 1=f, 2=g, 3=h
    # W flat indices: 0=C[0][0], 1=C[0][1], 2=C[1][0], 3=C[1][1]
    
    terms = []
    
    # P1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1]
    # Should contribute +1 to C[0][0] and +1 to C[1][1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # P2 = (c+d)(e): U=[0,0,1,1], V=[1,0,0,0]
    # Should contribute +1 to C[1][0]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # P3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1]
    # Should contribute +1 to C[0][1]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    
    # P4 = d(g-e): U=[0,0,0,1], V=[0,0,1,-1]
    # Should contribute -1 to C[0][0] and +1 to C[1][1]
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]))
    
    # P5 = (a+b)(h): U=[1,1,0,0], V=[0,0,0,1]
    # Should contribute +1 to C[0][0] and +1 to C[0][1]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    
    # P6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0]
    # Should contribute +1 to C[0][0]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]))
    
    # P7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1]
    # Should contribute +1 to C[0][1] and +1 to C[1][1]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]))
    
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition numerically."""
    terms = _strassen_2x2x2_terms()
    for trial in range(100):
        a, b, c, d = [random.randint(-10, 10) for _ in range(4)]
        e, f, g, h = [random.randint(-10, 10) for _ in range(4)]
        A_flat = [a, b, c, d]
        B_flat = [e, f, g, h]
        
        # Expected C
        C00 = a*e + b*f
        C01 = a*g + b*h
        C10 = c*e + d*f
        C11 = c*g + d*h
        expected = [C00, C01, C10, C11]
        
        # Computed C
        computed = [0, 0, 0, 0]
        for U, V, W in terms:
            u_val = sum(U[i] * A_flat[i] for i in range(4))
            v_val = sum(V[i] * B_flat[i] for i in range(4))
            for j in range(4):
                computed[j] += W[j] * u_val * v_val
        
        if computed != expected:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition for 2x4x6 matrix multiplication using
    2 blocks of 3 Strassen 2x2x2 (rank 7 each) = 42 total.
    
    A is 2x4, B is 4x6, C is 2x6.
    Split middle dimension 4 into two 2's:
    A = [A1 | A2], A1 and A2 are 2x2
    B = [B1; B2], B1 and B2 are 2x6
    C = A1*B1 + A2*B2
    
    For each 2x2x6 product, split the output 6 into 3 groups of 2:
    B1 = [B1_0 | B1_1 | B1_2], each 2x2
    A1*B1 = A1*B1_0 + A1*B1_1 + A1*B1_2, each 2x2
    Use Strassen for each 2x2x2: rank 7
    Total for A1*B1: 3*7 = 21
    Total for A2*B2: 3*7 = 21
    Grand total: 42
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    strassen_terms = _strassen_2x2x2_terms()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    for a_block in range(2):  # A1, A2
        for b_block in range(3):  # three 2x2 sub-blocks of B
            for t in range(7):
                col = a_block * 21 + b_block * 7 + t
                U_t, V_t, W_t = strassen_terms[t]
                
                # A_flat for this subproblem: 2x2 matrix from A
                # A is 2x4, flat: [A[0][0], A[0][1], A[0][2], A[0][3], A[1][0], A[1][1], A[1][2], A[1][3]]
                # A1 = A[:, 0:2] = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
                # A2 = A[:, 2:4] = [[A[0][2], A[0][3]], [A[1][2], A[1][3]]]
                # A_flat for A1: [A[0][0], A[0][1], A[1][0], A[1][1]]
                # A_flat for A2: [A[0][2], A[0][3], A[1][2], A[1][3]]
                
                a_indices = [0, 1, 4, 5] if a_block == 0 else [2, 3, 6, 7]
                for i in range(4):
                    U[a_indices[i]][col] = U_t[i]
                
                # B is 4x6, flat: [B[0][0], B[0][1], ..., B[0][5], B[1][0], ..., B[3][5]]
                # B1 = B[0:2, :] = rows 0,1 of B (2x6)
                # B2 = B[2:4, :] = rows 2,3 of B (2x6)
                # B1_k = B1[:, 2*k:2*k+2] = B[0:2, 2*k:2*k+2]
                # B2_k = B2[:, 2*k:2*k+2] = B[2:4, 2*k:2*k+2]
                
                if a_block == 0:
                    b_row_start = 0  # B1 rows 0,1
                else:
                    b_row_start = 2  # B2 rows 2,3
                
                b_col_start = b_block * 2
                # V_flat indices for the 2x2 B block:
                # e = B[b_row_start][b_col_start] -> V_idx = b_row_start * 6 + b_col_start
                # f = B[b_row_start][b_col_start+1] -> V_idx = b_row_start * 6 + b_col_start + 1
                # g = B[b_row_start+1][b_col_start] -> V_idx = (b_row_start+1) * 6 + b_col_start
                # h = B[b_row_start+1][b_col_start+1] -> V_idx = (b_row_start+1) * 6 + b_col_start + 1
                
                b_indices = [
                    b_row_start * 6 + b_col_start,
                    b_row_start * 6 + b_col_start + 1,
                    (b_row_start + 1) * 6 + b_col_start,
                    (b_row_start + 1) * 6 + b_col_start + 1
                ]
                for i in range(4):
                    V[b_indices[i]][col] = V_t[i]
                
                # W: output C is 2x6, flat as [C[0][0], C[0][1], ..., C[0][5], C[1][0], ..., C[1][5]]
                # W_flat indices: 0=C[0][0], 1=C[0][1], ..., 5=C[0][5], 6=C[1][0], ..., 11=C[1][5]
                # For sub-block b_block, the output columns are 2*b_block and 2*b_block+1
                # W_t has 4 entries: [C[0][0], C[0][1], C[1][0], C[1][1]] of the 2x2 result
                # Map to: C[0][2*b_block], C[0][2*b_block+1], C[1][2*b_block], C[1][2*b_block+1]
                # W_flat indices: 2*b_block, 2*b_block+1, 6+2*b_block, 6+2*b_block+1
                
                w_indices = [
                    2 * b_block,
                    2 * b_block + 1,
                    6 + 2 * b_block,
                    6 + 2 * b_block + 1
                ]
                for i in range(4):
                    W[w_indices[i]][col] = W_t[i]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Expected C
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Computed C
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

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy:
    1. Try the Strassen block decomposition (rank 42)
    2. If verification fails, fall back to schoolbook (rank 48)
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Deadline guard: keep a safety margin before 120s
    deadline = start_time + 110
    
    # First, verify Strassen 2x2x2 base case
    if not _verify_strassen():
        # Strassen base case is broken; fall back to schoolbook
        return _schoolbook_decomposition()
    
    # Try the Strassen block decomposition (rank 42)
    if time.time() < deadline:
        result = _strassen_block_decomposition()
        
        # Verify the full decomposition
        if _verify_decomposition(result, n, m, p, trials=50):
            return result
    
    # Fall back to schoolbook decomposition (rank 48)
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
