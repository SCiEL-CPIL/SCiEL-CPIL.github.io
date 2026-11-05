"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
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
        
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        num_blocks = 2
        dim = n
        
        rank_inner = block_size ** 3 # 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]
        
        col = 0
        
        steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        for step in steps:
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']
            
            for u in range(block_size):
                for v in range(block_size):
                    for w in range(block_size):
                        for (I, J), ca in A_coeffs.items():
                            if ca != 0:
                                i_global = I * block_size + u
                                j_global = J * block_size + v
                                idx_u = i_global * dim + j_global
                                U[idx_u][col] += ca
                        for (J, K), cb in B_coeffs.items():
                            if cb != 0:
                                j_global = J * block_size + v
                                k_global = K * block_size + w
                                idx_v = j_global * dim + k_global
                                V[idx_v][col] += cb
                        for (I, K), cc in C_coeffs.items():
                            if cc != 0:
                                i_global = I * block_size + u
                                k_global = K * block_size + w
                                idx_w = k_global * dim + i_global
                                W[idx_w][col] += cc
                        col += 1
        
        return {"U": U, "V": V, "W": W}
        
    # Hybrid: 3x3 Laderman (Outer) x 2x2 Schoolbook (Inner) (Rank 189)
    # Laderman's algorithm for 3x3 matrix multiplication uses 15 multiplications.
    # If we divide the 6x6 matrix into four 3x3 blocks, and use schoolbook (9 mults) for each 3x3 block product,
    # the total rank is 15 * 9 = 135.
    # Wait, the outer dimension is 3x3 blocks. There are 4 blocks in A, 4 in B, 4 in C.
    # A = [[A11, A12], [A21, A22]] where each Aij is 3x3.
    # B = [[B11, B12], [B21, B22]] where each Bkj is 3x3.
    # C = A*B = [[C11, C12], [C21, C22]] where each Cij is 3x3.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Laderman's 15 products:
    # M1 = A11 * B11
    # M2 = A12 * B21
    # M3 = A21 * B12
    # M4 = A22 * B22
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 + A22) * (B21 + B11)
    # M7 = (A11 + A12) * (B12 - B22)
    # M8 = (A21 - A11) * (B11 + B12)
    # M9 = (A22 - A12) * (B21 + B22)
    # M10 = (A11 + A21) * (B12 - B11)  -- Wait, let's get the correct Laderman formulas.
    
    # Correct Laderman 3x3 formulas:
    # P1 = A11 * B11
    # P2 = A12 * B21
    # P3 = A21 * B12
    # P4 = A22 * B22
    # P5 = (A11 + A22) * (B11 + B22)
    # P6 = (A12 + A22) * (B21 + B11)
    # P7 = (A11 + A12) * (B12 - B22)
    # P8 = (A21 - A11) * (B11 + B12)
    # P9 = (A22 - A12) * (B21 + B22)
    # P10 = (A11 + A21) * (B12 - B11)
    # P11 = (A12 + A21) * (B11 - B22)
    # P12 = (A11 + A12 + A21 + A22) * (B11 - B22) -- No, this is not right.
    
    # Let's use the standard Laderman decomposition:
    # M1 = A11 B11
    # M2 = A12 B21
    # M3 = A21 B12
    # M4 = A22 B22
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 + A22)(B21 + B11)
    # M7 = (A11 + A12)(B12 - B22)
    # M8 = (A21 - A11)(B11 + B12)
    # M9 = (A22 - A12)(B21 + B22)
    # M10 = (A11 + A21)(B12 - B11)
    # M11 = (A12 + A21)(B11 - B22)
    # M12 = (A11 + A12 + A21 + A22)(B11 - B22) -- This is getting complicated.
    
    # Actually, let's just use the 2x2 Strassen x 3x3 Schoolbook (189) which is already implemented.
    # Is there a better hybrid?
    # 4x4 matrix multiplication rank is 49 (Laderman).
    # If we use 4x4 blocks, we can't fit them in 6x6.
    
    # What about 3x3 Laderman (15) x 2x2 Schoolbook (4)?
    # Rank = 15 * 4 = 60? No, the inner dimension is 2x2.
    # A is 6x6. Divide into 3x3 blocks? No, 6 is divisible by 3.
    # So we have 2x2 blocks of 3x3 matrices.
    # A = [[A11, A12], [A21, A22]] where each is 3x3.
    # B = [[B11, B12], [B21, B22]] where each is 3x3.
    # C = A*B.
    # Each Cij is a 3x3 matrix product of two 3x3 matrices.
    # We can use Laderman's algorithm for the 2x2 block structure, which requires 15 "multiplications" of 3x3 matrices.
    # Each 3x3 matrix multiplication can be done with 27 scalar multiplications (schoolbook).
    # So total rank = 15 * 27 = 405. This is worse than 189.
    
    # What if we use Strassen for the 2x2 block structure (7 mults) and Laderman for the 3x3 inner?
    # Rank = 7 * 15 = 105.
    # This is much better!
    # Let's implement this.
    
    def _hybrid_strassen_laderman():
        block_size = 3
        dim = n
        
        # Strassen 2x2 block structure
        # Laderman 3x3 inner structure
        
        # Total rank = 7 * 15 = 105
        total_rank = 105
        
        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]
        
        # Strassen steps for 2x2 blocks
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        # Laderman 3x3 inner steps
        # We need the 15 products for 3x3 matrix multiplication.
        # Let's define them.
        laderman_steps = [
            {'A': {(0,0): 1}, 'B': {(0,0): 1}, 'C': {(0,0): 1}}, # M1 = A11 B11
            {'A': {(0,1): 1}, 'B': {(1,0): 1}, 'C': {(0,0): 1}}, # M2 = A12 B21
            {'A': {(1,0): 1}, 'B': {(0,1): 1}, 'C': {(1,0): 1}}, # M3 = A21 B12
            {'A': {(1,1): 1}, 'B': {(1,1): 1}, 'C': {(1,1): 1}}, # M4 = A22 B22
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}}, # M5
            {'A': {(0,1): 1, (1,1): 1}, 'B': {(1,0): 1, (0,0): 1}, 'C': {(0,0): 1, (1,0): 1}}, # M6
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}}, # M7
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}}, # M8
            {'A': {(1,1): 1, (0,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}}, # M9
            {'A': {(0,0): 1, (1,0): 1}, 'B': {(0,1): 1, (0,0): -1}, 'C': {(0,1): 1, (1,1): 1}}, # M10
            {'A': {(0,1): 1, (1,0): 1}, 'B': {(0,0): 1, (1,1): -1}, 'C': {(0,1): 1, (1,0): 1}}, # M11
            {'A': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): -1}, 'C': {(0,0): 1, (1,1): 1}}, # M12
            {'A': {(0,0): 1, (1,1): -1}, 'B': {(0,1): 1, (1,0): 1}, 'C': {(0,0): 1, (1,1): 1}}, # M13
            {'A': {(0,1): 1, (1,0): -1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,1): 1, (1,0): 1}}, # M14
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,0): 1, (1,1): -1}, 'C': {(1,1): 1}} # M15
        ]
        
        # This is getting very complex to implement correctly.
        # Let's stick with the 189 rank solution for now, as it is simpler and already works.
        # The 105 rank solution would be a significant improvement if correct.
        
        # For now, return the 189 rank solution.
        return _hybrid_strassen_schoolbook()
        
    # Try Hybrid (189)
    try:
        res = _hybrid_strassen_schoolbook()
        # Verify
        U_np = np.array(res["U"], dtype=float)
        V_np = np.array(res["V"], dtype=float)
        W_np = np.array(res["W"], dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            return res
    except Exception:
        pass
        
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
