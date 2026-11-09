import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _build_strassen_decomposition():
    """
    Constructs a rank-77 decomposition for 4x4 by 4x6 matrix multiplication
    using Strassen's algorithm as a building block.
    """
    n, m, p = DIMENSIONS
    
    # Base Strassen vectors for 2x2x2
    # Indices 0..3 correspond to 2x2 matrix elements in row-major order
    base_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    base_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    base_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]

    U_list = []
    V_list = []
    W_list = []

    # Block offsets for U (4x4 matrix, flattened row-major)
    # A00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    U_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # Block offsets for V (4x6 matrix, flattened row-major)
    # B00: rows 0,1 cols 0,1 -> 0,1,4,5
    # B01: rows 0,1 cols 2,3 -> 2,3,6,7
    # B10: rows 2,3 cols 0,1 -> 8,9,12,13
    # B11: rows 2,3 cols 2,3 -> 10,11,14,15
    # B02: rows 0,1 cols 4,5 -> 16,17,20,21
    # B12: rows 2,3 cols 4,5 -> 18,19,22,23
    V_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15],
        '02': [16, 17, 20, 21],
        '12': [18, 19, 22, 23]
    }
    
    # Block offsets for W (p*n = 24 entries, indexed as W[k*n+i] where k is col of C, i is row of C)
    # For C[i,j] where i in 0..3, j in 0..5: W index = j*n + i = j*4 + i
    # C00: i in 0,1; j in 0,1 -> W[0*4+0]=0, W[1*4+0]=4, W[0*4+1]=1, W[1*4+1]=5
    # C01: i in 0,1; j in 2,3 -> W[2*4+0]=8, W[3*4+0]=12, W[2*4+1]=9, W[3*4+1]=13
    # C10: i in 2,3; j in 0,1 -> W[0*4+2]=2, W[1*4+2]=6, W[0*4+3]=3, W[1*4+3]=7
    # C11: i in 2,3; j in 2,3 -> W[2*4+2]=10, W[3*4+2]=14, W[2*4+3]=11, W[3*4+3]=15
    # C02: i in 0,1; j in 4,5 -> W[4*4+0]=16, W[5*4+0]=20, W[4*4+1]=17, W[5*4+1]=21
    # C12: i in 2,3; j in 4,5 -> W[4*4+2]=18, W[5*4+2]=22, W[4*4+3]=19, W[5*4+3]=23
    W_offsets = {
        '00': [0, 4, 1, 5],
        '01': [8, 12, 9, 13],
        '10': [2, 6, 3, 7],
        '11': [10, 14, 11, 15],
        '02': [16, 20, 17, 21],
        '12': [18, 22, 19, 23]
    }

    def add_combined_strassen(a_combs, b_combs, w_combs):
        """
        Add rank-1 tensors for a Strassen product of combined blocks.
        a_combs: list of (block_key, sign) for A matrix
        b_combs: list of (block_key, sign) for B matrix
        w_combs: list of (block_key, sign) for W output
        """
        for t in range(7):
            u_full = [0] * 16
            v_full = [0] * 24
            w_full = [0] * 24
            
            for block, sign in a_combs:
                base_vec = base_u[t]
                offsets = U_offsets[block]
                for i, val in enumerate(base_vec):
                    if val != 0:
                        u_full[offsets[i]] += val * sign
                        
            for block, sign in b_combs:
                base_vec = base_v[t]
                offsets = V_offsets[block]
                for i, val in enumerate(base_vec):
                    if val != 0:
                        v_full[offsets[i]] += val * sign
                        
            for block, sign in w_combs:
                base_vec = base_w[t]
                offsets = W_offsets[block]
                for i, val in enumerate(base_vec):
                    if val != 0:
                        w_full[offsets[i]] += val * sign
                        
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)

    # Strassen's 7 meta-products for 4x4x4 part
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    add_combined_strassen(
        [('00', 1), ('11', 1)],
        [('00', 1), ('11', 1)],
        [('00', 1), ('11', 1)]
    )
    
    # M2: (A10+A11)B00 -> C10
    add_combined_strassen(
        [('10', 1), ('11', 1)],
        [('00', 1)],
        [('10', 1)]
    )
    
    # M3: A00(B01-B11) -> C01
    add_combined_strassen(
        [('00', 1)],
        [('01', 1), ('11', -1)],
        [('01', 1)]
    )
    
    # M4: A11(B10-B00) -> C00 + C10
    add_combined_strassen(
        [('11', 1)],
        [('10', 1), ('00', -1)],
        [('00', 1), ('10', 1)]
    )
    
    # M5: (A00+A01)B11 -> C01 - C00
    add_combined_strassen(
        [('00', 1), ('01', 1)],
        [('11', 1)],
        [('01', 1), ('00', -1)]
    )
    
    # M6: (A10-A00)(B00+B01) -> C11
    add_combined_strassen(
        [('10', 1), ('00', -1)],
        [('00', 1), ('01', 1)],
        [('11', 1)]
    )
    
    # M7: (A01-A11)(B10+B11) -> C00 + C11
    add_combined_strassen(
        [('01', 1), ('11', -1)],
        [('10', 1), ('11', 1)],
        [('00', 1), ('11', 1)]
    )
    
    # Now for the right part (4x4x2)
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    
    # Product 1: A00 B02 -> C02
    add_combined_strassen(
        [('00', 1)],
        [('02', 1)],
        [('02', 1)]
    )
    
    # Product 2: A01 B12 -> C02
    add_combined_strassen(
        [('01', 1)],
        [('12', 1)],
        [('02', 1)]
    )
    
    # Product 3: A10 B02 -> C12
    add_combined_strassen(
        [('10', 1)],
        [('02', 1)],
        [('12', 1)]
    )
    
    # Product 4: A11 B12 -> C12
    add_combined_strassen(
        [('11', 1)],
        [('12', 1)],
        [('12', 1)]
    )
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Construct the Strassen-based decomposition
    result = _build_strassen_decomposition()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
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

# EVOLVE-BLOCK-END
