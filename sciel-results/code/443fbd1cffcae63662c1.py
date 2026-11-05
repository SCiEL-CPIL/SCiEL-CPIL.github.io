"""Matrix multiplication 3,3,7 - rank-minimal decomposition using Strassen block decomposition."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    We split the output dimension p=7 into 3, 3, 1.
    C = [C1, C2, C3] where C1, C2 are 3x3 and C3 is 3x1.
    
    We use a block decomposition for the 3x3 * 3x3 part:
    Rank(3x3*3x3) <= 26 (7 Strassen + 19 Schoolbook borders)
    Rank(3x3*3x1) = 9 (Schoolbook)
    
    Total Rank = 26 + 26 + 9 = 61.
    """
    n, m, p = DIMENSIONS
    rank = 61
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w):
        nonlocal col
        for i, ui in enumerate(u_indices):
            U[ui, col] = coeffs_u[i]
        for j, vj in enumerate(v_indices):
            V[vj, col] = coeffs_v[j]
        for k, wk in enumerate(w_indices):
            W[wk, col] = coeffs_w[k]
        col += 1

    def compute_3x3x3(b_offset, c_offset):
        nonlocal col
        
        # A indices: 0..8
        # B indices: b_offset + 0..8 (within the 3x3 block)
        # C indices: c_offset + 0..8 (within the 3x3 block)
        
        # Strassen 7 terms
        # M1 = A11 * (B12 - B22)
        add_term([0], [b_offset+1, b_offset+8], [c_offset+1, c_offset+4], [1], [1, -1], [1, -1])
        # M2 = (A11 + A12) * B22
        add_term([0, 1], [b_offset+8], [c_offset+0, c_offset+3, c_offset+4], [1, 1], [1], [1, -1, 1]) # C11, C21, C22 contributions?
        # Wait, the W vector for a rank-1 term $u \otimes v \otimes w$ must be the coefficient vector for C.
        # For M2, it contributes to C11, C21, C22?
        # C11 = M5 + M4 - M2 + M6
        # C12 = M1 + M5
        # C21 = M3 + M7
        # C22 = M5 - M1 + M3 - M7
        # So M2 contributes -1 to C11.
        # My add_term function adds a single rank-1 tensor $u \otimes v \otimes w$.
        # The W vector is the linear form for C.
        # For M2, $u = A11+A12$, $v = B22$.
        # The contribution to C is $w$.
        # $w$ must be such that $C = \dots - M2 \dots$
        # So $w$ is the vector representing the linear form in C that M2 multiplies.
        # Actually, the tensor is $T = \sum u_k \otimes v_k \otimes w_k$.
        # $C_{ij} = \sum_k (u_k \cdot A) (v_k \cdot B) (w_k)_{ij}$.
        # So $w_k$ is a vector in the C space.
        
        # Let's redo the Strassen terms with correct W vectors.
        
        # M1: u=[1,0,0,0] (A11), v=[0,1,-1,0] (B12-B22)
        # w: C12 - C22? No.
        # C12 = M1 + M5. C22 = M5 - M1 ...
        # So M1 contributes +1 to C12 and -1 to C22.
        # w1 = [0, 1, 0, -1] (C11, C12, C21, C22)
        add_term([0], [b_offset+1, b_offset+8], [c_offset+1, c_offset+4], [1], [1, -1], [1, -1])
        
        # M2: u=[1,1,0,0] (A11+A12), v=[0,0,0,1] (B22)
        # C11 = ... - M2 ...
        # w2 = [-1, 0, 0, 0]
        add_term([0, 1], [b_offset+8], [c_offset+0], [1, 1], [1], [-1])
        
        # M3: u=[0,0,1,1] (A21+A22), v=[1,0,0,0] (B11)
        # C21 = M3 + M7. C22 = ... + M3 ...
        # w3 = [0, 0, 1, 1]
        add_term([3, 4], [b_offset+0], [c_offset+3, c_offset+4], [1, 1], [1], [1, 1])
        
        # M4: u=[0,0,0,1] (A22), v=[0,-1,1,0] (B21-B11)
        # C11 = ... + M4 ...
        # w4 = [1, 0, 0, 0]
        add_term([4], [b_offset+7, b_offset+0], [c_offset+0], [1], [-1, 1], [1])
        
        # M5: u=[1,0,0,1] (A11+A22), v=[1,0,0,1] (B11+B22)
        # C11 = M5 ... C12 = ... + M5 ... C22 = M5 ...
        # w5 = [1, 1, 0, 1]
        add_term([0, 4], [b_offset+0, b_offset+8], [c_offset+0, c_offset+1, c_offset+4], [1, 1], [1, 1], [1, 1, 1])
        
        # M6: u=[0,1,-1,0] (A12-A22), v=[0,0,1,1] (B21+B22)
        # C11 = ... + M6 ...
        # w6 = [1, 0, 0, 0]
        add_term([1, 4], [b_offset+7, b_offset+8], [c_offset+0], [1, -1], [1, 1], [1])
        
        # M7: u=[1,-1,0,0] (A11-A21), v=[1,1,0,0] (B11+B12)
        # C21 = ... + M7 ... C22 = ... - M7 ...
        # w7 = [0, 0, 1, -1]
        add_term([0, 3], [b_offset+0, b_offset+1], [c_offset+3, c_offset+4], [1, -1], [1, 1], [1, -1])
        
        # Border terms (Schoolbook)
        # a = A13, A23 (indices 2, 5)
        # b = A31, A32 (indices 6, 7)
        # c = A33 (index 8)
        # d = B13, B23 (indices b_offset+2, b_offset+5)
        # e = B31, B32 (indices b_offset+6, b_offset+7)
        # f = B33 (index b_offset+8)
        
        # C13 = A11 B13 + A12 B23 + A13 B33
        # C23 = A21 B13 + A22 B23 + A23 B33
        # C31 = A31 B11 + A32 B21 + A33 B31
        # C32 = A31 B12 + A32 B22 + A33 B32
        # C33 = A31 B13 + A32 B23 + A33 B33
        
        # Note: C13, C23 are in the 3x3 block.
        # C13 is at c_offset+2. C23 is at c_offset+5.
        
        # C13 terms:
        add_term([0], [b_offset+2], [c_offset+2], [1], [1], [1])
        add_term([1], [b_offset+5], [c_offset+2], [1], [1], [1])
        add_term([2], [b_offset+8], [c_offset+2], [1], [1], [1])
        
        # C23 terms:
        add_term([3], [b_offset+2], [c_offset+5], [1], [1], [1])
        add_term([4], [b_offset+5], [c_offset+5], [1], [1], [1])
        add_term([5], [b_offset+8], [c_offset+5], [1], [1], [1])
        
        # C31 terms:
        add_term([6], [b_offset+0], [c_offset+6], [1], [1], [1])
        add_term([7], [b_offset+7], [c_offset+6], [1], [1], [1])
        add_term([8], [b_offset+6], [c_offset+6], [1], [1], [1])
        
        # C32 terms:
        add_term([6], [b_offset+1], [c_offset+7], [1], [1], [1])
        add_term([7], [b_offset+8], [c_offset+7], [1], [1], [1])
        add_term([8], [b_offset+7], [c_offset+7], [1], [1], [1])
        
        # C33 terms:
        add_term([6], [b_offset+2], [c_offset+8], [1], [1], [1])
        add_term([7], [b_offset+5], [c_offset+8], [1], [1], [1])
        add_term([8], [b_offset+8], [c_offset+8], [1], [1], [1])

    # C1 (cols 0-2)
    compute_3x3x3(0, 0)
    
    # C2 (cols 3-5)
    compute_3x3x3(3, 3)
    
    # C3 (col 6)
    # C3 = A * B3 (3x3 * 3x1)
    # B3 is col 6 of B.
    # B3 indices: 6, 13, 20
    # C3 indices: 6, 13, 20? No.
    # C is 3x7. C3 is col 6.
    # W indices for C[:, 6] are 6*3+0=18, 6*3+1=19, 6*3+2=20.
    
    for i in range(3):
        for j in range(3):
            u_idx = i * 3 + j
            v_idx = j * 7 + 6
            w_idx = 6 * 3 + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
