"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # (8, 20)
    V = [[0] * rank for _ in range(m * p)]  # (20, 20)
    W = [[0] * rank for _ in range(p * n)]  # (10, 20)
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_222(y, z, w):
        """
        Decomposes 2x2 * 2x2 -> 2x2 using Strassen (Rank 7).
        y: list of (i, j) for A
        z: list of (j, k) for B
        w: list of (k, i) for C
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        
        # M1 = (y00+y11)(z00+z11) -> X00 += M1, X11 += M1
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00 -> X10 += M2, X11 -= M2
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w10)][col] += 1
        W[w_idx(*w11)][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11) -> X01 += M3, X11 += M3
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M4 = y11(z10-z00) -> X00 += M4, X10 += M4
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w10)][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11 -> X00 -= M5, X01 += M5
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] -= 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01) -> X11 += M6
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11) -> X00 += M7
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w00)][col] += 1
        col += 1

    def trivial_221(y, z, w):
        """
        Decomposes 2x2 * 2x1 -> 2x1 using Trivial (Rank 4).
        y: list of (i, j) for A (2x2)
        z: list of (j, k) for B (2x1, k=0)
        w: list of (k, i) for C (2x1, k=0)
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z0, z1 = z # z0=(0,0), z1=(1,0)
        w0, w1 = w # w0=(0,0), w1=(0,1)
        
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00 -> C00
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z0)][col] = 1
        W[w_idx(*w0)][col] += 1
        col += 1
        
        # Term 2: A01 B10 -> C00
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z1)][col] = 1
        W[w_idx(*w0)][col] += 1
        col += 1
        
        # Term 3: A10 B00 -> C10
        U[u_idx(*y10)][col] = 1
        V[v_idx(*z0)][col] = 1
        W[w_idx(*w1)][col] += 1
        col += 1
        
        # Term 4: A11 B10 -> C10
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z1)][col] = 1
        W[w_idx(*w1)][col] += 1
        col += 1

    def trivial_215(y, z, w):
        """
        Decomposes 2x1 * 1x5 -> 2x5 using Rank 1.
        y: (i, j) for A (2x1, j=0)
        z: (j, k) for B (1x5, j=0)
        w: (k, i) for C (2x5)
        """
        nonlocal col
        # y is list of 2 indices
        # z is list of 5 indices
        # w is list of 10 indices
        
        # U[i, col] = 1 for i in 0,1
        # V[k, col] = 1 for k in 0..4
        # W[k, i, col] = 1 for all k, i
        
        for i in range(2):
            U[u_idx(y[i][0], y[i][1])][col] = 1
        for k in range(5):
            V[v_idx(z[k][0], z[k][1])][col] = 1
        for k in range(5):
            for i in range(2):
                W[w_idx(k, i)][col] += 1
        col += 1

    # Split m=4 into 3 and 1.
    # A: 2x4 -> A_L (2x3), A_R (2x1)
    # B: 4x5 -> B_L (3x5), B_R (1x5)
    
    # Part 1: A_R (2x1) * B_R (1x5) -> C (2x5)
    # A_R indices: (0,3), (1,3)
    # B_R indices: (3,0)..(3,4)
    # C indices: (0,0)..(4,1)
    y_R = [(0,3), (1,3)]
    z_R = [(3,0), (3,1), (3,2), (3,3), (3,4)]
    w_R = [(k, i) for k in range(5) for i in range(2)]
    trivial_215(y_R, z_R, w_R)
    
    # Part 2: A_L (2x3) * B_L (3x5) -> C (2x5)
    # Split A_L (2x3) into A_L1 (2x2) and A_L2 (2x1)
    # Split B_L (3x5) into B_L1 (2x5) and B_L2 (1x5)
    
    # Sub-part 2.2: A_L2 (2x1) * B_L2 (1x5) -> C (2x5)
    # A_L2 indices: (0,2), (1,2)
    # B_L2 indices: (2,0)..(2,4)
    y_L2 = [(0,2), (1,2)]
    z_L2 = [(2,0), (2,1), (2,2), (2,3), (2,4)]
    w_L2 = [(k, i) for k in range(5) for i in range(2)]
    trivial_215(y_L2, z_L2, w_L2)
    
    # Sub-part 2.1: A_L1 (2x2) * B_L1 (2x5) -> C (2x5)
    # Split B_L1 (2x5) into B_L1a (2x2), B_L1b (2x2), B_L1c (2x1)
    
    # 2.1.a: A_L1 (2x2) * B_L1a (2x2) -> C (2x2) [Cols 0,1]
    y_L1 = [(0,0), (0,1), (1,0), (1,1)]
    z_L1a = [(0,0), (0,1), (1,0), (1,1)]
    w_L1a = [(0,0), (1,0), (0,1), (1,1)] # C[0:2, 0:2]
    strassen_222(y_L1, z_L1a, w_L1a)
    
    # 2.1.b: A_L1 (2x2) * B_L1b (2x2) -> C (2x2) [Cols 2,3]
    z_L1b = [(0,2), (0,3), (1,2), (1,3)]
    w_L1b = [(2,0), (3,0), (2,1), (3,1)] # C[0:2, 2:4]
    strassen_222(y_L1, z_L1b, w_L1b)
    
    # 2.1.c: A_L1 (2x2) * B_L1c (2x1) -> C (2x1) [Col 4]
    z_L1c = [(0,4), (1,4)]
    w_L1c = [(4,0), (4,1)] # C[0:2, 4:5]
    trivial_221(y_L1, z_L1c, w_L1c)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
