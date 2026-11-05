"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140.0
    safety_margin = 5.0

    # Rank 22 decomposition for 2x3 x 3x4
    # Split B into B1 (3x2) and B2 (3x2)
    # Each 2x3 x 3x2 multiplication has rank 11 (Strassen 7 + Outer 4)
    # Total rank 22
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs, rank_idx):
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][rank_idx] = coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][rank_idx] = coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][rank_idx] = coeff

    # Helper to generate terms for 2x3 x 3x2 block
    # A is 2x3, B_block is 3x2, C_block is 2x2
    # A indices: i in {0,1}, j in {0,1,2} -> i*3+j
    # B indices: j in {0,1,2}, k in {k0, k0+1} -> j*4 + (k0 + local_k)
    # C indices: k in {k0, k0+1}, i in {0,1} -> (k0 + local_k)*2 + i
    
    def add_block(k0, rank_offset):
        # k0 is 0 for first block, 2 for second block
        
        # Strassen terms for A_L (2x2) * B_L (2x2)
        # A_L uses j=0,1. B_L uses j=0,1, k=k0, k0+1
        # C_L uses k=k0, k0+1
        
        # M1: (a11+a22)(b11+b22)
        # U: a11(0), a22(4). V: b11(j0,k0), b22(j1,k0+1). W: c11(1), c22(-1)
        add_term([0, 4], [1, 1], [k0, 4 + k0 + 1], [1, 1], [k0 * 2, k0 * 2 + 3], [1, -1], rank_offset + 0)
        
        # M2: (a21+a22)b11
        # U: a21(3), a22(4). V: b11. W: c21(1)
        add_term([3, 4], [1, 1], [k0], [1], [k0 * 2 + 1], [1], rank_offset + 1)
        
        # M3: a11(b12-b22)
        # U: a11(0). V: b12(j0,k0+1), b22(j1,k0+1). W: c12(1), c22(1)
        add_term([0], [1], [k0 + 1, 4 + k0 + 1], [1, -1], [k0 * 2 + 2, k0 * 2 + 3], [1, 1], rank_offset + 2)
        
        # M4: a22(b21-b11)
        # U: a22(4). V: b21(j1,k0), b11(j0,k0). W: c11(1), c21(1)
        add_term([4], [1], [4 + k0, k0], [1, -1], [k0 * 2, k0 * 2 + 1], [1, 1], rank_offset + 3)
        
        # M5: (a11+a12)b22
        # U: a11(0), a12(1). V: b22. W: c11(-1), c12(1)
        add_term([0, 1], [1, 1], [4 + k0 + 1], [1], [k0 * 2, k0 * 2 + 2], [-1, 1], rank_offset + 4)
        
        # M6: (a21-a11)(b11+b12)
        # U: a21(3), a11(0). V: b11, b12. W: c22(1)
        add_term([3, 0], [1, -1], [k0, k0 + 1], [1, 1], [k0 * 2 + 3], [1], rank_offset + 5)
        
        # M7: (a12-a22)(b21+b22)
        # U: a12(1), a22(4). V: b21, b22. W: c11(1)
        add_term([1, 4], [1, -1], [4 + k0, 4 + k0 + 1], [1, 1], [k0 * 2], [1], rank_offset + 6)
        
        # Outer product terms for A_R (2x1) * B_R (1x2)
        # A_R uses j=2. B_R uses j=2, k=k0, k0+1
        # C_R uses k=k0, k0+1
        
        # T1: a13*b31
        # U: a13(2). V: b31(j2,k0). W: c11(1)
        add_term([2], [1], [8 + k0], [1], [k0 * 2], [1], rank_offset + 7)
        
        # T2: a13*b32
        # U: a13(2). V: b32(j2,k0+1). W: c12(1)
        add_term([2], [1], [8 + k0 + 1], [1], [k0 * 2 + 2], [1], rank_offset + 8)
        
        # T3: a23*b31
        # U: a23(5). V: b31. W: c21(1)
        add_term([5], [1], [8 + k0], [1], [k0 * 2 + 1], [1], rank_offset + 9)
        
        # T4: a23*b32
        # U: a23(5). V: b32. W: c22(1)
        add_term([5], [1], [8 + k0 + 1], [1], [k0 * 2 + 3], [1], rank_offset + 10)

    # Add block 1 (cols 0,1 of B)
    add_block(0, 0)
    
    # Add block 2 (cols 2,3 of B)
    add_block(2, 11)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
