import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen product definitions
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Products:
    # P1: (x00+x11)(y00+y11) -> Z00: 1, Z11: 1
    # P2: (x10+x11)y00       -> Z10: 1, Z11: -1
    # P3: x00(y10-y11)       -> Z01: 1, Z11: 1
    # P4: x11(y01-y00)       -> Z00: 1, Z10: 1
    # P5: (x00+x01)y11       -> Z00: -1, Z01: 1
    # P6: (x10-x00)(y00+y10) -> Z11: 1
    # P7: (x01-x11)(y01+y11) -> Z00: 1
    
    # Mapping of product index to (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {A_index: coeff}
    # V_coeffs: dict {B_index: coeff}
    # W_coeffs: dict {C_index: coeff}
    
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # X is A1 (j=0,1), Y is B1 (j=0,1)
    # X' is A2 (j=2,3), Y' is B2 (j=2,3)
    
    # We iterate over k_block (0..3)
    for kb in range(4):
        k0 = 2 * kb
        k1 = 2 * kb + 1
        
        # Define the 14 products for this kb
        # 0-6 for Z (X*Y), 7-13 for Z' (X'*Y')
        
        # Helper to set a product
        def add_product(u_dict, v_dict, w_dict):
            nonlocal col
            for idx, coeff in u_dict.items():
                U[idx][col] += coeff
            for idx, coeff in v_dict.items():
                V[idx][col] += coeff
            for idx, coeff in w_dict.items():
                W[idx][col] += coeff
            col += 1
            
        # Products for Z = X Y
        # X entries: A[i, j] for i in 0,1; j in 0,1
        # Y entries: B[j, k] for j in 0,1; k in k0, k1
        
        # P1: (x00+x11)(y00+y11)
        u1 = {0: 1, 3: 1} # A[0,0]=0, A[1,1]=3
        v1 = {k0: 1, p + k1: 1} # B[0,k0]=k0, B[1,k1]=p+k1
        w1 = {k0*n: 1, (k1+1)*n + 1: 1} # C[0,k0]=k0, C[1,k1]=k1*2+1? 
        # C index is k*n + i.
        # Z00 is C[0, k0]. Index k0*2 + 0 = 2*k0.
        # Z11 is C[1, k1]. Index k1*2 + 1 = 2*k1 + 1.
        # Wait, k1 = 2*kb+1.
        # Z00: i=0, k=k0. W[2*k0]
        # Z11: i=1, k=k1. W[2*k1 + 1]
        w1 = {2*k0: 1, 2*k1 + 1: 1}
        add_product(u1, v1, w1)
        
        # P2: (x10+x11)y00
        u2 = {2: 1, 3: 1} # A[1,0]=2, A[1,1]=3
        v2 = {k0: 1} # B[0,k0]
        # Z10: i=1, k=k0. W[2*k0 + 1]
        # Z11: i=1, k=k1. W[2*k1 + 1]
        w2 = {2*k0 + 1: 1, 2*k1 + 1: -1}
        add_product(u2, v2, w2)
        
        # P3: x00(y10-y11)
        u3 = {0: 1} # A[0,0]
        v3 = {p + k0: 1, p + k1: -1} # B[1,k0], B[1,k1]
        # Z01: i=0, k=k1. W[2*k1]
        # Z11: i=1, k=k1. W[2*k1 + 1]
        w3 = {2*k1: 1, 2*k1 + 1: 1}
        add_product(u3, v3, w3)
        
        # P4: x11(y01-y00)
        u4 = {3: 1} # A[1,1]
        v4 = {k1: 1, k0: -1} # B[0,k1], B[0,k0]
        # Z00: i=0, k=k0. W[2*k0]
        # Z10: i=1, k=k0. W[2*k0 + 1]
        w4 = {2*k0: 1, 2*k0 + 1: 1}
        add_product(u4, v4, w4)
        
        # P5: (x00+x01)y11
        u5 = {0: 1, 1: 1} # A[0,0], A[0,1]
        v5 = {p + k1: 1} # B[1,k1]
        # Z00: -1, Z01: 1
        w5 = {2*k0: -1, 2*k1: 1}
        add_product(u5, v5, w5)
        
        # P6: (x10-x00)(y00+y10)
        u6 = {2: 1, 0: -1} # A[1,0], A[0,0]
        v6 = {k0: 1, p + k0: 1} # B[0,k0], B[1,k0]
        # Z11: 1
        w6 = {2*k1 + 1: 1}
        add_product(u6, v6, w6)
        
        # P7: (x01-x11)(y01+y11)
        u7 = {1: 1, 3: -1} # A[0,1], A[1,1]
        v7 = {k1: 1, p + k1: 1} # B[0,k1], B[1,k1]
        # Z00: 1
        w7 = {2*k0: 1}
        add_product(u7, v7, w7)
        
        # Products for Z' = X' Y'
        # X' entries: A[i, j] for i in 0,1; j in 2,3
        # Y' entries: B[j, k] for j in 2,3; k in k0, k1
        
        # P1': (x'00+x'11)(y'00+y'11)
        u1p = {4: 1, 7: 1} # A[0,2]=4, A[1,3]=7
        v1p = {2*p + k0: 1, 3*p + k1: 1} # B[2,k0], B[3,k1]
        w1p = {2*k0: 1, 2*k1 + 1: 1}
        add_product(u1p, v1p, w1p)
        
        # P2': (x'10+x'11)y'00
        u2p = {6: 1, 7: 1} # A[1,2]=6, A[1,3]=7
        v2p = {2*p + k0: 1} # B[2,k0]
        w2p = {2*k0 + 1: 1, 2*k1 + 1: -1}
        add_product(u2p, v2p, w2p)
        
        # P3': x'00(y'10-y'11)
        u3p = {4: 1} # A[0,2]
        v3p = {3*p + k0: 1, 3*p + k1: -1} # B[3,k0], B[3,k1]
        w3p = {2*k1: 1, 2*k1 + 1: 1}
        add_product(u3p, v3p, w3p)
        
        # P4': x'11(y'01-y'00)
        u4p = {7: 1} # A[1,3]
        v4p = {2*p + k1: 1, 2*p + k0: -1} # B[2,k1], B[2,k0]
        w4p = {2*k0: 1, 2*k0 + 1: 1}
        add_product(u4p, v4p, w4p)
        
        # P5': (x'00+x'01)y'11
        u5p = {4: 1, 5: 1} # A[0,2], A[0,3]
        v5p = {3*p + k1: 1} # B[3,k1]
        w5p = {2*k0: -1, 2*k1: 1}
        add_product(u5p, v5p, w5p)
        
        # P6': (x'10-x'00)(y'00+y'10)
        u6p = {6: 1, 4: -1} # A[1,2], A[0,2]
        v6p = {2*p + k0: 1, 3*p + k0: 1} # B[2,k0], B[3,k0]
        w6p = {2*k1 + 1: 1}
        add_product(u6p, v6p, w6p)
        
        # P7': (x'01-x'11)(y'01+y'11)
        u7p = {5: 1, 7: -1} # A[0,3], A[1,3]
        v7p = {2*p + k1: 1, 3*p + k1: 1} # B[2,k1], B[3,k1]
        w7p = {2*k0: 1}
        add_product(u7p, v7p, w7p)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
