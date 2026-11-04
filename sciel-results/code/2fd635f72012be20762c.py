"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 51 construction
    # 3 blocks of 2x3x3
    # Each 2x3x3 has rank 17 (11 for 2x2x3 + 6 for 2x1x3)
    # 2x2x3 has rank 11 (7 for 2x2x2 + 4 for 2x2x1)
    
    U = [[0] * 51 for _ in range(n * m)]
    V = [[0] * 51 for _ in range(m * p)]
    W = [[0] * 51 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_sp, v_sp, w_sp):
        nonlocal col
        for idx, val in u_sp.items():
            U[idx][col] = val
        for idx, val in v_sp.items():
            V[idx][col] = val
        for idx, val in w_sp.items():
            W[idx][col] = val
        col += 1

    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i

    # We have 3 blocks of columns in B (k in 0..8)
    for block in range(3):
        k_base = block * 3
        
        # Sub-block 1: 2x2x3 (A cols 0,1; B rows 0,1; B cols k_base..k_base+2)
        # Rank 11: 7 (Strassen 2x2x2) + 4 (2x2x1)
        
        # Strassen 2x2x2 for B cols k_base, k_base+1
        # A: i in 0,1; j in 0,1
        # B: j in 0,1; k in k_base, k_base+1
        # C: i in 0,1; k in k_base, k_base+1
        
        # M1 = (a00+a11)(b00+b11)
        add_term({0:1, 3:1}, {0*k_base:1, 2*k_base:1}, {k_base*n:1, (k_base+1)*n+1:1})
        
        # M2 = (a10+a11)b00
        add_term({2:1, 3:1}, {0*k_base:1}, {(k_base+1)*n:1}) # c10 -> k=k_base+1, i=0 -> (k_base+1)*2+0
        
        # M3 = a00(b01-b11)
        add_term({0:1}, {1*k_base:1, 3*k_base:-1}, {k_base*n+1:1}) # c01 -> k=k_base, i=1 -> k_base*2+1
        
        # M4 = a11(b10-b00)
        add_term({3:1}, {2*k_base:1, 0*k_base:-1}, {k_base*n:1, (k_base+1)*n:1}) # c00, c10
        
        # M5 = (a00+a01)b11
        add_term({0:1, 1:1}, {3*k_base:1}, {k_base*n:-1, k_base*n+1:1}) # c00(-), c01(+)
        
        # M6 = (a10-a00)(b00+b01)
        add_term({2:1, 0:-1}, {0*k_base:1, 1*k_base:1}, {(k_base+1)*n+1:1}) # c11 -> k=k_base+1, i=1
        
        # M7 = (a01-a11)(b10+b11)
        add_term({1:1, 3:-1}, {2*k_base:1, 3*k_base:1}, {k_base*n:1, k_base*n+1:1, (k_base+1)*n:1, (k_base+1)*n+1:1})
        # Wait, M7 contributes to C00 and C11?
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M7
        # So M7 is +1 in C00 and +1 in C11.
        # C00 index: k_base*n + 0
        # C11 index: (k_base+1)*n + 1
        
        # 2x2x1 for B col k_base+2
        # A: i in 0,1; j in 0,1
        # B: j in 0,1; k in k_base+2
        # C: i in 0,1; k in k_base+2
        # Rank 4
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        # Note: B col is single, so b00 is B[0, k_base+2], b10 is B[1, k_base+2].
        
        # Term 1: a00*b00 -> c00
        add_term({0:1}, {0*(k_base+2):1}, {(k_base+2)*n:1})
        # Term 2: a01*b10 -> c00
        add_term({1:1}, {1*(k_base+2):1}, {(k_base+2)*n:1})
        # Term 3: a10*b00 -> c10
        add_term({2:1}, {0*(k_base+2):1}, {(k_base+2)*n+1:1})
        # Term 4: a11*b10 -> c10
        add_term({3:1}, {1*(k_base+2):1}, {(k_base+2)*n+1:1})
        
        # Sub-block 2: 2x1x3 (A col 2; B row 2; B cols k_base..k_base+2)
        # Rank 6
        # a20 * b20 -> c00
        # a20 * b21 -> c01
        # a20 * b22 -> c02
        # a21 * b20 -> c10
        # a21 * b21 -> c11
        # a21 * b22 -> c12
        
        for kk in range(3):
            k_idx = k_base + kk
            # a20 * b2k -> c0k
            add_term({4:1}, {2*k_idx:1}, {k_idx*n:1})
            # a21 * b2k -> c1k
            add_term({5:1}, {2*k_idx:1}, {k_idx*n+1:1})
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
