"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using 3 blocks of M_{2,4,2}
    # M_{2,4,2} is decomposed into two M_{2,2,2} (Strassen)
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-z)
    # M4 = d(z-v) -> d(z-w)
    # M5 = (a+b)w
    # M6 = (c-b)(x+w)
    # M7 = (d-c)(y+z)
    # C = [[M1-M4+M7, M3+M4], [M2+M4, M1-M3+M5]]
    
    def add_strassen_block(A_idx_map, B_idx_map, C_idx_map, col_offset):
        # A_idx_map: dict from 'a','b','c','d' to U row index
        # B_idx_map: dict from 'x','y','z','w' to V row index
        # C_idx_map: dict from (r,c) to W row index
        
        # M1
        U[A_idx_map['a']][col_offset] += 1
        U[A_idx_map['d']][col_offset] += 1
        V[B_idx_map['x']][col_offset] += 1
        V[B_idx_map['w']][col_offset] += 1
        W[C_idx_map[(0,0)]][col_offset] += 1
        W[C_idx_map[(1,1)]][col_offset] += 1
        
        # M2
        U[A_idx_map['c']][col_offset+1] += 1
        U[A_idx_map['d']][col_offset+1] += 1
        V[B_idx_map['x']][col_offset+1] += 1
        W[C_idx_map[(1,0)]][col_offset+1] += 1
        
        # M3
        U[A_idx_map['a']][col_offset+2] += 1
        V[B_idx_map['y']][col_offset+2] += 1
        V[B_idx_map['z']][col_offset+2] -= 1
        W[C_idx_map[(0,1)]][col_offset+2] += 1
        
        # M4
        U[A_idx_map['d']][col_offset+3] += 1
        V[B_idx_map['z']][col_offset+3] += 1
        V[B_idx_map['w']][col_offset+3] -= 1
        W[C_idx_map[(0,0)]][col_offset+3] -= 1
        W[C_idx_map[(0,1)]][col_offset+3] += 1
        W[C_idx_map[(1,0)]][col_offset+3] += 1
        
        # M5
        U[A_idx_map['a']][col_offset+4] += 1
        U[A_idx_map['b']][col_offset+4] += 1
        V[B_idx_map['w']][col_offset+4] += 1
        W[C_idx_map[(1,1)]][col_offset+4] += 1
        
        # M6
        U[A_idx_map['c']][col_offset+5] += 1
        U[A_idx_map['b']][col_offset+5] -= 1
        V[B_idx_map['x']][col_offset+5] += 1
        V[B_idx_map['w']][col_offset+5] += 1
        W[C_idx_map[(0,0)]][col_offset+5] += 1
        
        # M7
        U[A_idx_map['d']][col_offset+6] += 1
        U[A_idx_map['c']][col_offset+6] -= 1
        V[B_idx_map['y']][col_offset+6] += 1
        V[B_idx_map['z']][col_offset+6] += 1
        W[C_idx_map[(0,0)]][col_offset+6] += 1
        W[C_idx_map[(1,1)]][col_offset+6] -= 1

    for i in range(3):
        # Block i corresponds to B columns 2i and 2i+1
        col_offset = 14 * i
        
        # A_L is A[:, 0:2]
        A_L_map = {
            'a': 0, # A[0,0]
            'b': 1, # A[0,1]
            'c': 4, # A[1,0]
            'd': 5  # A[1,1]
        }
        
        # A_R is A[:, 2:4]
        A_R_map = {
            'a': 2, # A[0,2]
            'b': 3, # A[0,3]
            'c': 6, # A[1,2]
            'd': 7  # A[1,3]
        }
        
        # B_L is B[0:2, 2i:2i+2]
        B_L_map = {
            'x': 0 * 6 + 2*i,   # B[0, 2i]
            'y': 0 * 6 + 2*i+1, # B[0, 2i+1]
            'z': 1 * 6 + 2*i,   # B[1, 2i]
            'w': 1 * 6 + 2*i+1  # B[1, 2i+1]
        }
        
        # B_R is B[2:4, 2i:2i+2]
        B_R_map = {
            'x': 2 * 6 + 2*i,   # B[2, 2i]
            'y': 2 * 6 + 2*i+1, # B[2, 2i+1]
            'z': 3 * 6 + 2*i,   # B[3, 2i]
            'w': 3 * 6 + 2*i+1  # B[3, 2i+1]
        }
        
        # C is C[:, 2i:2i+2]
        # W index k*n + i_row. n=2.
        # C[0, 2i] -> 2i*2 + 0 = 4i
        # C[1, 2i] -> 2i*2 + 1 = 4i+1
        # C[0, 2i+1] -> (2i+1)*2 + 0 = 4i+2
        # C[1, 2i+1] -> (2i+1)*2 + 1 = 4i+3
        C_map = {
            (0,0): 4*i,
            (1,0): 4*i+1,
            (0,1): 4*i+2,
            (1,1): 4*i+3
        }
        
        # P1 = A_L * B_L
        add_strassen_block(A_L_map, B_L_map, C_map, col_offset)
        
        # P2 = A_R * B_R
        add_strassen_block(A_R_map, B_R_map, C_map, col_offset + 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
