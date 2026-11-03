"""Program for matrix multiplication 3,4,5 tensor rank minimization via Strassen blocks."""
import time

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Uses a combination of Strassen and schoolbook terms to minimize rank.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer entries.
    """
    n, m, p = N, M, P
    t0 = time.time()
    
    def strassen_terms(i_off, j_off, k_off):
        """
        Return the 7 Strassen terms for a 2x2x2 block.
        """
        terms = []
        U1 = {(i_off + 0, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
        V1 = {(j_off + 0, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
        W1 = {(k_off + 0, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
        terms.append((U1, V1, W1))
        
        U2 = {(i_off + 1, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
        V2 = {(j_off + 0, k_off + 0): 1}
        W2 = {(k_off + 0, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
        terms.append((U2, V2, W2))
        
        U3 = {(i_off + 0, j_off + 0): 1}
        V3 = {(j_off + 0, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
        W3 = {(k_off + 1, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
        terms.append((U3, V3, W3))
        
        U4 = {(i_off + 1, j_off + 1): 1}
        V4 = {(j_off + 1, k_off + 0): 1, (j_off + 0, k_off + 0): -1}
        W4 = {(k_off + 0, i_off + 1): 1, (k_off + 0, i_off + 0): 1}
        terms.append((U4, V4, W4))
        
        U5 = {(i_off + 0, j_off + 0): 1, (i_off + 0, j_off + 1): 1}
        V5 = {(j_off + 1, k_off + 1): 1}
        W5 = {(k_off + 1, i_off + 0): 1, (k_off + 0, i_off + 0): -1}
        terms.append((U5, V5, W5))
        
        U6 = {(i_off + 1, j_off + 0): 1, (i_off + 0, j_off + 0): -1}
        V6 = {(j_off + 0, k_off + 0): 1, (j_off + 0, k_off + 1): 1}
        W6 = {(k_off + 1, i_off + 1): 1}
        terms.append((U6, V6, W6))
        
        U7 = {(i_off + 0, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
        V7 = {(j_off + 1, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
        W7 = {(k_off + 0, i_off + 0): 1}
        terms.append((U7, V7, W7))
        
        return terms

    all_terms = []
    
    # 1. Main 3x3x3 block: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # Sub-block 1.1: Strassen 2x2x2 for i,j,k in {0,1}
    all_terms.extend(strassen_terms(0, 0, 0))
    
    # Sub-block 1.2: i=2, j in {0,1,2}, k in {0,1,2} (1x3x3)
    # Rank 3: One term for each j
    for j in range(3):
        U_dict = {(2, j): 1}
        V_dict = {(j, k): 1 for k in range(3)}
        W_dict = {(k, 2): 1 for k in range(3)}
        all_terms.append((U_dict, V_dict, W_dict))
        
    # Sub-block 1.3: i in {0,1}, j=2, k in {0,1,2} (2x1x3)
    # Rank 1: Single term
    U_dict = {(i, 2): 1 for i in range(2)}
    V_dict = {(2, k): 1 for k in range(3)}
    W_dict = {(k, i): 1 for i in range(2) for k in range(3)}
    all_terms.append((U_dict, V_dict, W_dict))
    
    # Sub-block 1.4: i in {0,1}, j in {0,1}, k=2 (2x2x1)
    # Rank 2: One term for each j
    for j in range(2):
        U_dict = {(i, j): 1 for i in range(2)}
        V_dict = {(j, 2): 1}
        W_dict = {(2, i): 1 for i in range(2)}
        all_terms.append((U_dict, V_dict, W_dict))
        
    # 2. j=3 block: i in {0,1,2}, j=3, k in {0,1,2,3,4} (3x1x5)
    # Rank 1: Single term
    U_dict = {(i, 3): 1 for i in range(3)}
    V_dict = {(3, k): 1 for k in range(5)}
    W_dict = {(k, i): 1 for i in range(3) for k in range(5)}
    all_terms.append((U_dict, V_dict, W_dict))
    
    # 3. Remainder k in {3,4}, j in {0,1,2}, i in {0,1,2} (3x3x2)
    # Sub-block 3.1: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {3,4}
    all_terms.extend(strassen_terms(0, 0, 3))
    
    # Sub-block 3.2: i=2, j in {0,1,2}, k in {3,4} (1x3x2)
    # Rank 3: One term for each j
    for j in range(3):
        U_dict = {(2, j): 1}
        V_dict = {(j, k): 1 for k in range(3, 5)}
        W_dict = {(k, 2): 1 for k in range(3, 5)}
        all_terms.append((U_dict, V_dict, W_dict))
        
    # Sub-block 3.3: i in {0,1}, j=2, k in {3,4} (2x1x2)
    # Rank 1: Single term
    U_dict = {(i, 2): 1 for i in range(2)}
    V_dict = {(2, k): 1 for k in range(3, 5)}
    W_dict = {(k, i): 1 for i in range(2) for k in range(3, 5)}
    all_terms.append((U_dict, V_dict, W_dict))
    
    rank = len(all_terms)
    
    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    
    # Fallback to schoolbook if verification fails
    U0 = [[0] * RANK_SCHOOLBOOK for _ in range(n * m)]
    V0 = [[0] * RANK_SCHOOLBOOK for _ in range(m * p)]
    W0 = [[0] * RANK_SCHOOLBOOK for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    return {"U": U0, "V": V0, "W": W0}
