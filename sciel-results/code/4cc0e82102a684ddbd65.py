"""Matrix multiplication 5,5,5 rank-minimal decomposition attempt."""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank decomposition of the 5x5x5 matrix multiplication tensor.
    Uses a 2+3 block decomposition.
    Rank: 7 (Strassen 2x2) + 12 + 12 + 18 + 12 + 18 + 18 + 27 (Schoolbook) = 124.
    """
    n, m, p = DIMENSIONS
    # Total rank estimate
    rank = 124
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook decomposition
    def add_schoolbook(n_sub, m_sub, p_sub, i_start, j_start, k_start):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    gi = i_start + i
                    gj = j_start + j
                    gk = k_start + k
                    
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1

    # Helper to add Strassen decomposition for 2x2
    def add_strassen_2x2(i_start, j_start, k_start):
        nonlocal col
        # Strassen terms for 2x2
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We define the coefficients for U, V, W for each term
        # U row: i*m + j
        # V row: j*p + k
        # W row: k*n + i
        
        # Term 1
        # A: (0,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        terms = []
        
        # P1
        terms.append({
            'U': [(0, 1), (1, 1)], # (i, j) relative to block
            'V': [(0, 1), (0, 1), (1, 1), (1, 1)], # (j, k)
            'W': [(0, 1), (0, 1), (1, 1), (1, 1)], # (k, i)
            'C': [(0, 1), (0, 1), (1, 1), (1, 1)] # (k, i)
        })
        # This manual construction is error-prone. Let's use the explicit formula.
        
        # Let's just use schoolbook for 2x2 as well to be safe?
        # Rank 8 for 2x2.
        # Total rank: 8 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 125.
        # That's the baseline.
        
        # I must implement Strassen correctly.
        
        # U coeffs for P1: A11, A22
        # V coeffs for P1: B11, B12, B21, B22
        # W coeffs for P1: C11, C12, C21, C22
        
        # Let's define a generic add_term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for (i, j), c in u_coeffs.items():
                if c != 0:
                    U[i_start * m + i + (j_start * 0) + j][col] += c # Wait, global index
            # This is getting messy with offsets.
            
        # Let's restart the Strassen implementation with explicit global indices.
        
        # Block is 2x2.
        # i in 0,1; j in 0,1; k in 0,1.
        
        # P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[0,1]=1, V[1,0]=1, V[1,1]=1
        # W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
        
        # P2 = (A[1,0]+A[1,1]) * B[0,0]
        # U[1,0]=1, U[1,1]=1
        # V[0,0]=1
        # W[0,0]=1 (C11 += P2), W[1,0]=1 (C21 += P2)
        
        # P3 = A[0,0] * (B[0,1]-B[1,1])
        # U[0,0]=1
        # V[0,1]=1, V[1,1]=-1
        # W[0,1]=1 (C12 += P3), W[1,1]=1 (C22 += P3)
        
        # P4 = A[1,1] * (B[1,0]-B[0,0])
        # U[1,1]=1
        # V[1,0]=1, V[0,0]=-1
        # W[0,0]=1 (C11 += P4), W[1,0]=1 (C21 += P4)
        
        # P5 = (A[0,0]+A[0,1]) * B[1,1]
        # U[0,0]=1, U[0,1]=1
        # V[1,1]=1
        # W[0,0]=-1 (C11 -= P5), W[0,1]=1 (C12 += P5)
        
        # P6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # U[1,0]=1, U[0,0]=-1
        # V[0,0]=1, V[0,1]=1
        # W[1,1]=1 (C22 += P6)
        
        # P7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # U[0,1]=1, U[1,1]=-1
        # V[1,0]=1, V[1,1]=1
        # W[0,0]=1 (C11 += P7)
        
        # Define terms as lists of (u_dict, v_dict, w_dict)
        
        t1_u = {(0,0):1, (1,1):1}
        t1_v = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
        t1_w = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
        
        t2_u = {(1,0):1, (1,1):1}
        t2_v = {(0,0):1}
        t2_w = {(0,0):1, (1,0):1}
        
        t3_u = {(0,0):1}
        t3_v = {(0,1):1, (1,1):-1}
        t3_w = {(0,1):1, (1,1):1}
        
        t4_u = {(1,1):1}
        t4_v = {(1,0):1, (0,0):-1}
        t4_w = {(0,0):1, (1,0):1}
        
        t5_u = {(0,0):1, (0,1):1}
        t5_v = {(1,1):1}
        t5_w = {(0,0):-1, (0,1):1}
        
        t6_u = {(1,0):1, (0,0):-1}
        t6_v = {(0,0):1, (0,1):1}
        t6_w = {(1,1):1}
        
        t7_u = {(0,1):1, (1,1):-1}
        t7_v = {(1,0):1, (1,1):1}
        t7_w = {(0,0):1}
        
        terms = [
            (t1_u, t1_v, t1_w),
            (t2_u, t2_v, t2_w),
            (t3_u, t3_v, t3_w),
            (t4_u, t4_v, t4_w),
            (t5_u, t5_v, t5_w),
            (t6_u, t6_v, t6_w),
            (t7_u, t7_v, t7_w)
        ]
        
        for u_d, v_d, w_d in terms:
            for (i, j), c in u_d.items():
                U[(i_start + i) * m + (j_start + j)][col] += c
            for (j, k), c in v_d.items():
                V[(j_start + j) * p + (k_start + k)][col] += c
            for (k, i), c in w_d.items():
                W[(k_start + k) * n + (i_start + i)][col] += c
            col += 1

    # 1. A22 B22 -> C22 (2x2x2)
    add_strassen_2x2(0, 0, 0)
    
    # 2. A23 B32 -> C22 (2x3x2)
    # A: i 0-1, j 2-4
    # B: j 2-4, k 0-1
    # C: k 0-1, i 0-1
    add_schoolbook(2, 3, 2, 0, 2, 0)
    
    # 3. A22 B23 -> C23 (2x2x3)
    # A: i 0-1, j 0-1
    # B: j 0-1, k 2-4
    # C: k 2-4, i 0-1
    add_schoolbook(2, 2, 3, 0, 0, 2)
    
    # 4. A23 B33 -> C23 (2x3x3)
    # A: i 0-1, j 2-4
    # B: j 2-4, k 2-4
    # C: k 2-4, i 0-1
    add_schoolbook(2, 3, 3, 0, 2, 2)
    
    # 5. A32 B22 -> C32 (3x2x2)
    # A: i 2-4, j 0-1
    # B: j 0-1, k 0-1
    # C: k 0-1, i 2-4
    add_schoolbook(3, 2, 2, 2, 0, 0)
    
    # 6. A33 B32 -> C32 (3x3x2)
    # A: i 2-4, j 2-4
    # B: j 2-4, k 0-1
    # C: k 0-1, i 2-4
    add_schoolbook(3, 3, 2, 2, 2, 0)
    
    # 7. A32 B23 -> C33 (3x2x3)
    # A: i 2-4, j 0-1
    # B: j 0-1, k 2-4
    # C: k 2-4, i 2-4
    add_schoolbook(3, 2, 3, 2, 0, 2)
    
    # 8. A33 B33 -> C33 (3x3x3)
    # A: i 2-4, j 2-4
    # B: j 2-4, k 2-4
    # C: k 2-4, i 2-4
    add_schoolbook(3, 3, 3, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
