import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Returns list of (u_coeff, v_coeff, w_coeff) for 2x2 Strassen.
    u_coeff: dict {(i,j): coeff} for A
    v_coeff: dict {(j,k): coeff} for B
    w_coeff: dict {(k,i): coeff} for C
    """
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (1,1): 1}
    w = {(0,0): 1, (1,1): 1} # C11 += M1, C22 += M1
    terms.append((u, v, w))
    
    # M2 = (A21+A22)B11
    u = {(1,0): 1, (1,1): 1}
    v = {(0,0): 1}
    w = {(0,1): 1, (1,1): -1} # C21 += M2, C22 -= M2
    terms.append((u, v, w))
    
    # M3 = A11(B12-B22)
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(1,0): 1, (1,1): 1} # C12 += M3, C22 += M3
    terms.append((u, v, w))
    
    # M4 = A22(B21-B11)
    u = {(1,1): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(0,0): 1, (0,1): 1} # C11 += M4, C21 += M4
    terms.append((u, v, w))
    
    # M5 = (A11+A12)B22
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(0,0): -1, (1,0): 1} # C11 -= M5, C12 += M5
    terms.append((u, v, w))
    
    # M6 = (A21-A11)(B11+B12)
    u = {(1,0): 1, (0,0): -1}
    v = {(0,0): 1, (0,1): 1}
    w = {(1,1): 1} # C22 += M6
    terms.append((u, v, w))
    
    # M7 = (A12-A22)(B21+B22)
    u = {(0,1): 1, (1,1): -1}
    v = {(1,0): 1, (1,1): 1}
    w = {(0,0): 1} # C11 += M7
    terms.append((u, v, w))
    
    return terms

def _laderman_3x3_terms():
    """
    Returns list of (u_coeff, v_coeff, w_coeff) for 3x3 Laderman.
    Laderman's algorithm for 3x3 matrix multiplication has rank 15.
    """
    terms = []
    
    # Indices are 0-based.
    # M1 = A[0,0]*(B[0,1]-B[1,1])
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(1,0): 1, (2,1): 1} # C12 += M1, C32 += M1
    terms.append((u, v, w))
    
    # M2 = (A[0,0]+A[0,1])*B[1,1]
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(1,0): 1} # C12 += M2
    terms.append((u, v, w))
    
    # M3 = A[2,2]*(B[1,0]-B[0,0])
    u = {(2,2): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(2,0): 1} # C31 += M3
    terms.append((u, v, w))
    
    # M4 = (A[2,0]+A[2,2])*B[0,0]
    u = {(2,0): 1, (2,2): 1}
    v = {(0,0): 1}
    w = {(2,0): 1} # C31 += M4
    terms.append((u, v, w))
    
    # M5 = A[0,2]*(B[2,0]-B[1,0])
    u = {(0,2): 1}
    v = {(2,0): 1, (1,0): -1}
    w = {(1,0): 1} # C21 += M5
    terms.append((u, v, w))
    
    # M6 = (A[0,0]+A[0,2])*B[2,0]
    u = {(0,0): 1, (0,2): 1}
    v = {(2,0): 1}
    w = {(1,0): 1} # C21 += M6
    terms.append((u, v, w))
    
    # M7 = A[1,2]*(B[2,1]-B[0,1])
    u = {(1,2): 1}
    v = {(2,1): 1, (0,1): -1}
    w = {(2,1): 1} # C23 += M7
    terms.append((u, v, w))
    
    # M8 = (A[1,1]+A[1,2])*B[0,1]
    u = {(1,1): 1, (1,2): 1}
    v = {(0,1): 1}
    w = {(2,1): 1} # C23 += M8
    terms.append((u, v, w))
    
    # M9 = A[2,1]*(B[0,2]-B[2,2])
    u = {(2,1): 1}
    v = {(0,2): 1, (2,2): -1}
    w = {(0,2): 1} # C13 += M9
    terms.append((u, v, w))
    
    # M10 = (A[2,1]+A[2,2])*B[2,2]
    u = {(2,1): 1, (2,2): 1}
    v = {(2,2): 1}
    w = {(0,2): 1} # C13 += M10
    terms.append((u, v, w))
    
    # M11 = (A[0,1]-A[2,2])*(B[2,2]+B[0,2])
    u = {(0,1): 1, (2,2): -1}
    v = {(2,2): 1, (0,2): 1}
    w = {(1,1): 1, (2,2): 1} # C22 += M11, C33 += M11
    terms.append((u, v, w))
    
    # M12 = (A[0,2]-A[1,1])*(B[1,1]+B[0,1])
    u = {(0,2): 1, (1,1): -1}
    v = {(1,1): 1, (0,1): 1}
    w = {(0,0): 1, (1,1): -1} # C11 += M12, C22 -= M12
    terms.append((u, v, w))
    
    # M13 = (A[1,2]-A[0,0])*(B[0,0]+B[2,0])
    u = {(1,2): 1, (0,0): -1}
    v = {(0,0): 1, (2,0): 1}
    w = {(0,0): -1, (1,1): 1} # C11 -= M13, C22 += M13
    terms.append((u, v, w))
    
    # M14 = (A[2,0]-A[1,1])*(B[1,0]+B[2,0])
    u = {(2,0): 1, (1,1): -1}
    v = {(1,0): 1, (2,0): 1}
    w = {(0,0): 1, (2,1): -1, (2,2): -1} # C11 += M14, C32 -= M14, C33 -= M14
    terms.append((u, v, w))
    
    # M15 = (A[2,1]-A[0,0])*(B[0,1]+B[1,1])
    u = {(2,1): 1, (0,0): -1}
    v = {(0,1): 1, (1,1): 1}
    w = {(2,1): 1, (2,2): 1} # C32 += M15, C33 += M15
    terms.append((u, v, w))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reshape
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    # Compute tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    
    # Target T is all ones.
    target = np.ones((n, m, p))
    
    if not np.allclose(T, target, atol=1e-9):
        return False
    return True

def _build_block_decomposition(n, m, p, block_size, block_type, offset_i, offset_j, offset_k):
    """
    Builds a block decomposition for a sub-matrix.
    offset_i, offset_j, offset_k are the starting indices in the full matrix.
    """
    terms = []
    
    if block_type == 'strassen':
        if block_size != 2:
            return []
        base_terms = _strassen_2x2_terms()
        for u, v, w in base_terms:
            u_shifted = {(i + offset_i, j + offset_j): c for (i, j), c in u.items()}
            v_shifted = {(j + offset_j, k + offset_k): c for (j, k), c in v.items()}
            w_shifted = {(k + offset_k, i + offset_i): c for (k, i), c in w.items()}
            terms.append((u_shifted, v_shifted, w_shifted))
            
    elif block_type == 'laderman':
        if block_size != 3:
            return []
        base_terms = _laderman_3x3_terms()
        for u, v, w in base_terms:
            u_shifted = {(i + offset_i, j + offset_j): c for (i, j), c in u.items()}
            v_shifted = {(j + offset_j, k + offset_k): c for (j, k), c in v.items()}
            w_shifted = {(k + offset_k, i + offset_i): c for (k, i), c in w.items()}
            terms.append((u_shifted, v_shifted, w_shifted))
            
    elif block_type == 'schoolbook':
        # Schoolbook for a block of size (rows_a, cols_a) x (rows_b, cols_b)
        # Here we assume square blocks for simplicity in this helper, 
        # but the current program uses rectangular blocks.
        # We'll stick to the specific loops in construct() for rectangular blocks
        # to avoid refactoring too much, but we can implement a generic one if needed.
        # For now, return empty and let construct() handle rectangulars explicitly 
        # or we can make this generic. Let's make it generic.
        # A block A_sub is (block_size x block_size) ? No, the current program 
        # uses 2x3 and 3x2.
        # Let's just implement the logic directly in construct for clarity, 
        # or pass dimensions.
        pass
        
    return terms

def construct():
    n, m, p = DIMENSIONS
    
    # Start with empty matrices
    # We don't know the rank ahead of time, so we'll build lists of terms and then flatten.
    
    terms = []
    
    # 1. Top-Left 2x2 (Strassen)
    # A: rows 0-1, cols 0-1. B: rows 0-1, cols 0-1. C: rows 0-1, cols 0-1.
    strassen_terms = _strassen_2x2_terms()
    for u, v, w in strassen_terms:
        # Shift indices (offset 0,0)
        u_shifted = {(i, j): c for (i, j), c in u.items()}
        v_shifted = {(j, k): c for (j, k), c in v.items()}
        w_shifted = {(k, i): c for (k, i), c in w.items()}
        terms.append((u_shifted, v_shifted, w_shifted))
    
    # 2. Bottom-Right 3x3 (Laderman)
    # A: rows 2-4, cols 2-4. B: rows 2-4, cols 2-4. C: rows 2-4, cols 2-4.
    laderman_terms = _laderman_3x3_terms()
    for u, v, w in laderman_terms:
        u_shifted = {(i+2, j+2): c for (i, j), c in u.items()}
        v_shifted = {(j+2, k+2): c for (j, k), c in v.items()}
        w_shifted = {(k+2, i+2): c for (k, i), c in w.items()}
        terms.append((u_shifted, v_shifted, w_shifted))
    
    # 3. Off-diagonal blocks (Schoolbook)
    
    # Block 1: A[0:2, 2:5] (2x3) * B[2:5, 0:2] (3x2) -> C[0:2, 0:2] (2x2)
    # i in 0-1, j in 2-4, k in 0-1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # Block 2: A[2:5, 0:2] (3x2) * B[0:2, 2:5] (2x3) -> C[2:5, 2:5] (3x3)
    # i in 2-4, j in 0-1, k in 2-4
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # Block 3: A[0:2, 0:2] (2x2) * B[0:2, 2:5] (2x3) -> C[0:2, 2:5] (2x3)
    # i in 0-1, j in 0-1, k in 2-4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # Block 4: A[0:2, 2:5] (2x3) * B[2:5, 2:5] (3x3) -> C[0:2, 2:5] (2x3)
    # i in 0-1, j in 2-4, k in 2-4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # Block 5: A[2:5, 0:2] (3x2) * B[0:2, 0:2] (2x2) -> C[2:5, 0:2] (3x2)
    # i in 2-4, j in 0-1, k in 0-1
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # Block 6: A[2:5, 2:5] (3x3) * B[2:5, 0:2] (3x2) -> C[2:5, 0:2] (3x2)
    # i in 2-4, j in 2-4, k in 0-1
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), c in u.items():
            U[i * m + j][r] = c
        for (j, k), c in v.items():
            V[j * p + k][r] = c
        for (k, i), c in w.items():
            W[k * n + i][r] = c
            
    # Convert to numpy arrays for verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    is_valid = _verify_decomposition(U_arr, V_arr, W_arr, n, m, p)
    
    if not is_valid:
        # If invalid, fall back to schoolbook for everything (Rank 125)
        # This is a safety net.
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        is_valid = _verify_decomposition(U_arr, V_arr, W_arr, n, m, p)
        if not is_valid:
            # This should never happen
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
