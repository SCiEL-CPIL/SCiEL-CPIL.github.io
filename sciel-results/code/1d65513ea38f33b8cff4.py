import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen
    s_terms = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        )
    ]
    
    c_comb = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)]
    ]
    
    # 4x4 Block Strassen
    b_terms = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        )
    ]
    
    bc_comb = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)]
    ]
    
    col = 0
    for t in range(7):
        x_block_coeffs, y_block_coeffs = b_terms[t]
        for s in range(7):
            x_scalar_coeffs, y_scalar_coeffs = s_terms[s]
            
            a_coeffs = {}
            for (u,v), cx in x_scalar_coeffs.items():
                for (br,bc), cb in x_block_coeffs.items():
                    idx = (br*2+u)*4 + (bc*2+v)
                    if idx not in a_coeffs:
                        a_coeffs[idx] = 0
                    a_coeffs[idx] += cx * cb
            
            b_coeffs = {}
            for (u,v), cx in y_scalar_coeffs.items():
                for (br,bc), cb in y_block_coeffs.items():
                    idx = (br*2+u)*4 + (bc*2+v)
                    if idx not in b_coeffs:
                        b_coeffs[idx] = 0
                    b_coeffs[idx] += cx * cb
            
            # Accumulate W for this column
            w_coeffs = {}
            
            for (u_prime, v_prime), signs_s in enumerate(c_comb):
                sign_s = 0
                for (s_idx, sgn) in signs_s:
                    if s_idx == s:
                        sign_s = sgn
                        break
                if sign_s == 0:
                    continue
                
                for (br, bc), signs_t in enumerate(bc_comb):
                    sign_t = 0
                    for (t_idx, sgn) in signs_t:
                        if t_idx == t:
                            sign_t = sgn
                            break
                    if sign_t == 0:
                        continue
                    
                    total_sign = sign_t * sign_s
                    w_idx = (bc*2+v_prime)*4 + (br*2+u_prime)
                    
                    if w_idx not in w_coeffs:
                        w_coeffs[w_idx] = 0
                    w_coeffs[w_idx] += total_sign
            
            # Add to U, V, W
            for a_idx, coeff_a in a_coeffs.items():
                U[a_idx][col] = coeff_a
            for b_idx, coeff_b in b_coeffs.items():
                V[b_idx][col] = coeff_b
            for w_idx, coeff_w in w_coeffs.items():
                W[w_idx][col] = coeff_w
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
