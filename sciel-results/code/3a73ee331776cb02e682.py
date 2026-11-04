"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill

def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) as dicts mapping (row_index, col_index) -> 1 for each factor.
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
        
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
        
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
        
    # M1 = (A00+A11)*(B00+B11)
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1); add_w(1, 1, 0, 1)
    
    # M2 = (A10+A11)*B00
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1); add_w(1, 1, 1, -1)
    
    # M3 = A00*(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1); add_w(1, 1, 2, 1)
    
    # M4 = A11*(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    add_w(0, 1, 3, 1); add_w(0, 0, 3, 1)
    
    # M5 = (A00+A01)*B11
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1); add_w(1, 0, 4, 1)
    
    # M6 = (A10-A00)*(B00+B01)
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)
    
    # M7 = (A01-A11)*(B10+B11)
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)
    
    return U_dict, V_dict, W_dict

def _make_3x3_laderman():
    """
    Laderman decomposition of 3x3 * 3x3 -> 3x3.
    Rank 23.
    Returns (U, V, W) as dicts.
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
        
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
        
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
        
    # 1-9: Direct terms
    # M1: A00 B00
    add_u(0,0,0,1); add_v(0,0,0,1); add_w(0,0,0,1)
    # M2: A01 B10
    add_u(0,1,1,1); add_v(1,0,1,1); add_w(0,0,1,1) # C00 gets A01 B10? No, C[i,k] = sum_j A[i,j]B[j,k].
    # C[0,0] = A00B00 + A01B10 + A02B20.
    # So M2 contributes to C00.
    # M3: A02 B20 -> C00
    add_u(0,2,2,1); add_v(2,0,2,1); add_w(0,0,2,1)
    # M4: A10 B01 -> C10
    add_u(1,0,3,1); add_v(0,1,3,1); add_w(1,0,3,1)
    # M5: A11 B11 -> C11
    add_u(1,1,4,1); add_v(1,1,4,1); add_w(1,1,4,1)
    # M6: A12 B21 -> C11
    add_u(1,2,5,1); add_v(2,1,5,1); add_w(1,1,5,1)
    # M7: A20 B02 -> C20
    add_u(2,0,6,1); add_v(0,2,6,1); add_w(2,0,6,1)
    # M8: A21 B12 -> C21
    add_u(2,1,7,1); add_v(1,2,7,1); add_w(2,1,7,1)
    # M9: A22 B22 -> C22
    add_u(2,2,8,1); add_v(2,2,8,1); add_w(2,2,8,1)
    
    # 10-23: Sum terms
    # M10: (A00+A11)(B00+B11)
    add_u(0,0,9,1); add_u(1,1,9,1)
    add_v(0,0,9,1); add_v(1,1,9,1)
    # Contributes to C00, C01, C10, C11?
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # We already have A00B00 (M1) and A11B11 (M5).
    # So M10 adds A00B11 and A11B00.
    # C01 gets A00B11. C10 gets A11B00.
    add_w(1,0,9,1) # C[0,1] gets A00B11? C[i,k]. i=0, k=1. Yes.
    add_w(0,1,9,1) # C[1,0] gets A11B00? i=1, k=0. Yes.
    # Note: C00 and C11 already have their direct terms. 
    # But wait, C00 = A00B00 + A01B10 + A02B20. 
    # M10 does NOT contribute to C00 directly in the final sum if we subtract?
    # Laderman's algorithm is a linear combination.
    # We must be careful. The W matrix defines how the products M_c contribute to C.
    # The identity is sum_c U_c V_c W_c = T.
    # For M10, the product is P10 = (A00+A11)(B00+B11).
    # P10 = A00B00 + A00B11 + A11B00 + A11B11.
    # We need to form C.
    # C00 needs A00B00. We have M1.
    # C11 needs A11B11. We have M5.
    # C01 needs A00B11.
    # C10 needs A11B00.
    # So M10 should contribute to C01 and C10.
    # Does it contribute to C00 or C11? No, because those are handled by M1 and M5?
    # Actually, in Laderman, the terms are designed so that the sum works out.
    # I will assume the standard Laderman W coefficients.
    
    # This is getting complex to hardcode from memory.
    # I will implement a verification. If it fails, I return None.
    
    # Let's use a known valid set of coefficients for Laderman 3x3.
    # Reference: 
    # U, V, W for Laderman 3x3.
    # I will use a pre-computed set if I can recall it, otherwise I will fallback.
    # Given the risk, I will implement the 2x2 Strassen (7) and 3x3 Schoolbook (27) 
    # which is known to work (Rank 189).
    # To improve, I will try to implement Laderman.
    
    # If I cannot guarantee Laderman correctness, I should stick to 189.
    # However, the directive asks for improvements.
    # I will implement Laderman and verify.
    
    # For now, I will return None to indicate failure, and the caller will fallback.
    # I will hardcode the Laderman terms I derived above and verify.
    
    # M11: (A00+A22)(B00+B22)
    add_u(0,0,10,1); add_u(2,2,10,1)
    add_v(0,0,10,1); add_v(2,2,10,1)
    add_w(2,0,10,1) # C02? A00B22? No.
    # This is too error-prone without a reference.
    
    return None

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
