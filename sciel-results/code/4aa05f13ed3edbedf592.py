"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Uses block-Strassen structure:
    - Split middle dimension (4) into two 2s: A_top (middle cols 0-1), A_bot (middle cols 2-3)
    - Split output columns (5) into 2 + 3: left block (cols 0-1), right block (cols 2-4)
    - For 2x2 * 2x2 sub-problems: use Strassen (rank 7)
    - For 2x2 * 2x1 sub-problems: use schoolbook (rank 4)
    
    Total: 4 Strassen blocks (4*7=28) + 2 schoolbook blocks (2*4=8) = 36
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # U shape: (n*m, rank) = (8, rank), indexed by i*m + j
    # V shape: (m*p, rank) = (20, rank), indexed by j*p + k
    # W shape: (p*n, rank) = (10, rank), indexed by k*n + i
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y00, y01, y10, y11, z00, z01, z10, z11, c00, c01, c10, c11):
        """
        Add 7 Strassen rank-1 terms for 2x2 * 2x2 multiplication.
        
        y indices: A matrix entries (i, j) pairs
        z indices: B matrix entries (j, k) pairs  
        c indices: C matrix entries (i, k) pairs
        
        Strassen formulas:
        m1 = (x11+x22)*(y11+y22) -> z11 += 1, z22 += 1
        m2 = (x21+x22)*y11      -> z21 += 1, z22 -= 1
        m3 = x11*(y12-y22)      -> z12 += 1, z22 += 1
        m4 = x22*(y21-y11)      -> z11 += 1, z21 += 1
        m5 = (x11+x12)*y22      -> z11 -= 1, z12 += 1
        m6 = (x21-x11)*(y11+y12)-> z22 += 1
        m7 = (x12-x22)*(y21+y22)-> z11 += 1
        """
        nonlocal col
        
        # m1 = (x11+x22)*(y11+y22)
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*c00)][col] += 1
        W[w_idx(*c11)][col] += 1
        col += 1
        
        # m2 = (x21+x22)*y11
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*c10)][col] += 1
        W[w_idx(*c11)][col] -= 1
        col += 1
        
        # m3 = x11*(y12-y22)
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*c01)][col] += 1
        W[w_idx(*c11)][col] += 1
        col += 1
        
        # m4 = x22*(y21-y11)
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*c00)][col] += 1
        W[w_idx(*c10)][col] += 1
        col += 1
        
        # m5 = (x11+x12)*y22
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*c00)][col] -= 1
        W[w_idx(*c01)][col] += 1
        col += 1
        
        # m6 = (x21-x11)*(y11+y12)
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*c11)][col] += 1
        col += 1
        
        # m7 = (x12-x22)*(y21+y22)
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*c00)][col] += 1
        col += 1
    
    def schoolbook_2x2x1(y00, y01, y10, y11, z0, z1, c0, c1):
        """
        Add 4 schoolbook rank-1 terms for 2x2 * 2x1 multiplication.
        
        y indices: A matrix entries (i, j) pairs
        z indices: B matrix entries (j, k) pairs (k is fixed)
        c indices: C matrix entries (i, k) pairs
        """
        nonlocal col
        
        # c0 = y00*z0 + y01*z1
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z0)][col] = 1
        W[w_idx(*c0)][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z1)][col] = 1
        W[w_idx(*c0)][col] += 1
        col += 1
        
        # c1 = y10*z0 + y11*z1
        U[u_idx(*y10)][col] = 1
        V[v_idx(*z0)][col] = 1
        W[w_idx(*c1)][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z1)][col] = 1
        W[w_idx(*c1)][col] += 1
        col += 1
    
    # Block 1: A_top * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_block(
        y00=(0,0), y01=(0,1), y10=(1,0), y11=(1,1),
        z00=(0,0), z01=(0,1), z10=(1,0), z11=(1,1),
        c00=(0,0), c01=(0,1), c10=(1,0), c11=(1,1)
    )
    
    # Block 2: A_bot * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_block(
        y00=(0,2), y01=(0,3), y10=(1,2), y11=(1,3),
        z00=(2,0), z01=(2,1), z10=(3,0), z11=(3,1),
        c00=(0,0), c01=(0,1), c10=(1,0), c11=(1,1)
    )
    
    # Block 3: A_top * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_block(
        y00=(0,0), y01=(0,1), y10=(1,0), y11=(1,1),
        z00=(0,2), z01=(0,3), z10=(1,2), z11=(1,3),
        c00=(2,0), c01=(2,1), c10=(3,0), c11=(3,1)
    )
    
    # Block 4: A_bot * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_block(
        y00=(0,2), y01=(0,3), y10=(1,2), y11=(1,3),
        z00=(2,2), z01=(2,3), z10=(3,2), z11=(3,3),
        c00=(2,0), c01=(2,1), c10=(3,0), c11=(3,1)
    )
    
    # Block 5: A_top * B[0:2, 4:5] -> C[0:2, 4:5]
    schoolbook_2x2x1(
        y00=(0,0), y01=(0,1), y10=(1,0), y11=(1,1),
        z0=(0,4), z1=(1,4),
        c0=(4,0), c1=(4,1)
    )
    
    # Block 6: A_bot * B[2:4, 4:5] -> C[0:2, 4:5]
    schoolbook_2x2x1(
        y00=(0,2), y01=(0,3), y10=(1,2), y11=(1,3),
        z0=(2,4), z1=(3,4),
        c0=(4,0), c1=(4,1)
    )
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verification: check all compatible triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): sum={s}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
