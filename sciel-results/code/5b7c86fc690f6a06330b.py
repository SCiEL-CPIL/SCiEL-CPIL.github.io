import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(u_map, v_map, w_map):
        nonlocal col
        # M1: (X00+X11)(Y00+Y11) -> Z00, Z11
        U[u_map[(0,0)], col] += 1
        U[u_map[(1,1)], col] += 1
        V[v_map[(0,0)], col] += 1
        V[v_map[(1,1)], col] += 1
        W[w_map[(0,0)], col] += 1
        W[w_map[(1,1)], col] += 1
        col += 1
        
        # M2: (X10+X11)Y00 -> Z10, -Z11
        U[u_map[(1,0)], col] += 1
        U[u_map[(1,1)], col] += 1
        V[v_map[(0,0)], col] += 1
        W[w_map[(1,0)], col] += 1
        W[w_map[(1,1)], col] -= 1
        col += 1
        
        # M3: X00(Y01-Y11) -> Z01, Z11
        U[u_map[(0,0)], col] += 1
        V[v_map[(0,1)], col] += 1
        V[v_map[(1,1)], col] -= 1
        W[w_map[(0,1)], col] += 1
        W[w_map[(1,1)], col] += 1
        col += 1
        
        # M4: X11(Y10-Y00) -> Z00, Z10
        U[u_map[(1,1)], col] += 1
        V[v_map[(1,0)], col] += 1
        V[v_map[(0,0)], col] -= 1
        W[w_map[(0,0)], col] += 1
        W[w_map[(1,0)], col] += 1
        col += 1
        
        # M5: (X00+X01)Y11 -> -Z00, Z01
        U[u_map[(0,0)], col] += 1
        U[u_map[(0,1)], col] += 1
        V[v_map[(1,1)], col] += 1
        W[w_map[(0,0)], col] -= 1
        W[w_map[(0,1)], col] += 1
        col += 1
        
        # M6: (X10-X00)(Y00+Y01) -> Z11
        U[u_map[(1,0)], col] += 1
        U[u_map[(0,0)], col] -= 1
        V[v_map[(0,0)], col] += 1
        V[v_map[(0,1)], col] += 1
        W[w_map[(1,1)], col] += 1
        col += 1
        
        # M7: (X01-X11)(Y10+Y11) -> Z00
        U[u_map[(0,1)], col] += 1
        U[u_map[(1,1)], col] -= 1
        V[v_map[(1,0)], col] += 1
        V[v_map[(1,1)], col] += 1
        W[w_map[(0,0)], col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(u_map, v_map, w_map):
        nonlocal col
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # Term 1: A00 B0 -> C0
        U[u_map[(0,0)], col] += 1
        V[v_map[(0,0)], col] += 1
        W[w_map[(0,0)], col] += 1
        col += 1
        
        # Term 2: A01 B1 -> C0
        U[u_map[(0,1)], col] += 1
        V[v_map[(1,0)], col] += 1
        W[w_map[(0,0)], col] += 1
        col += 1
        
        # Term 3: A10 B0 -> C1
        U[u_map[(1,0)], col] += 1
        V[v_map[(0,0)], col] += 1
        W[w_map[(1,0)], col] += 1
        col += 1
        
        # Term 4: A11 B1 -> C1
        U[u_map[(1,1)], col] += 1
        V[v_map[(1,0)], col] += 1
        W[w_map[(1,0)], col] += 1
        col += 1

    # Define Index Maps
    # A is 2x4. A[i, j] -> i*4 + j
    # B is 4x5. B[j, k] -> j*5 + k
    # C is 2x5. C[i, k] -> k*2 + i (Trace convention W[k*n+i])
    
    # Left Block A_L (cols 0,1)
    u_L = {
        (0,0): 0*4+0, (0,1): 0*4+1,
        (1,0): 1*4+0, (1,1): 1*4+1
    }
    
    # Right Block A_R (cols 2,3)
    u_R = {
        (0,0): 0*4+2, (0,1): 0*4+3,
        (1,0): 1*4+2, (1,1): 1*4+3
    }
    
    # B_L1 (rows 0,1, cols 0,1)
    v_L1 = {
        (0,0): 0*5+0, (0,1): 0*5+1,
        (1,0): 1*5+0, (1,1): 1*5+1
    }
    
    # B_L2 (rows 0,1, cols 2,3)
    v_L2 = {
        (0,0): 0*5+2, (0,1): 0*5+3,
        (1,0): 1*5+2, (1,1): 1*5+3
    }
    
    # B_L3 (rows 0,1, col 4)
    v_L3 = {
        (0,0): 0*5+4,
        (1,0): 1*5+4
    }
    
    # B_R1 (rows 2,3, cols 0,1)
    v_R1 = {
        (0,0): 2*5+0, (0,1): 2*5+1,
        (1,0): 3*5+0, (1,1): 3*5+1
    }
    
    # B_R2 (rows 2,3, cols 2,3)
    v_R2 = {
        (0,0): 2*5+2, (0,1): 2*5+3,
        (1,0): 3*5+2, (1,1): 3*5+3
    }
    
    # B_R3 (rows 2,3, col 4)
    v_R3 = {
        (0,0): 2*5+4,
        (1,0): 3*5+4
    }
    
    # W maps are the same for L and R blocks as they add to the same C
    # C[i, k] -> k*2 + i
    w_map = {
        (0,0): 0*2+0, (0,1): 1*2+0,
        (1,0): 0*2+1, (1,1): 1*2+1
    }
    
    # For 2x2x1, the output is 2x1.
    # C[i, k] where k is fixed.
    # For L3, k=4. C[0,4]->8, C[1,4]->9.
    w_L3 = {
        (0,0): 4*2+0, # C[0,4]
        (1,0): 4*2+1  # C[1,4]
    }
    
    # For R3, k=4. C[0,4]->8, C[1,4]->9.
    w_R3 = {
        (0,0): 4*2+0, # C[0,4]
        (1,0): 4*2+1  # C[1,4]
    }

    # Block 1: A_L B_L1
    add_strassen(u_L, v_L1, w_map)
    # Block 2: A_L B_L2
    add_strassen(u_L, v_L2, w_map)
    # Block 3: A_L B_L3
    add_schoolbook_2x2x1(u_L, v_L3, w_L3)
    
    # Block 4: A_R B_R1
    add_strassen(u_R, v_R1, w_map)
    # Block 5: A_R B_R2
    add_strassen(u_R, v_R2, w_map)
    # Block 6: A_R B_R3
    add_schoolbook_2x2x1(u_R, v_R3, w_R3)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
