"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_2x3x2_schoolbook():
    """
    Build 2x3x2 schoolbook decomposition. Rank 6.
    A is 2x3, B is 3x2, C is 2x2.
    U shape: (6, 6), V shape: (6, 6), W shape: (4, 6)
    """
    rank = 6
    U = [[0]*rank for _ in range(6)]   # 2*3 = 6 rows
    V = [[0]*rank for _ in range(6)]   # 3*2 = 6 rows
    W = [[0]*rank for _ in range(4)]   # 2*2 = 4 rows
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3 + j][col] = 1
                V[j*2 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    return U, V, W

def _build_2x2x4_strassen():
    """
    Build 2x2x4 decomposition using 2 Strassen 2x2x2 blocks. Rank 14.
    A is 2x2, B is 2x4, C is 2x4.
    U shape: (4, 14), V shape: (8, 14), W shape: (8, 14)
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(4)]   # 2*2 = 4 rows
    V = [[0]*rank for _ in range(8)]   # 2*4 = 8 rows
    W = [[0]*rank for _ in range(8)]   # 4*2 = 8 rows (p*n = 4*2)
    
    # Block 1: A * B[:, 0:2] -> C[:, 0:2]
    for t in range(7):
        col = t
        for r in range(4):
            U[r][col] = U2[r][t]
        for r in range(4):
            V[r][col] = V2[r][t]
        for r in range(4):
            W[r][col] = W2[r][t]
    
    # Block 2: A * B[:, 2:4] -> C[:, 2:4]
    for t in range(7):
        col = 7 + t
        for r in range(4):
            U[r][col] = U2[r][t]
        for r in range(4):
            V[4 + r][col] = V2[r][t]
        for r in range(4):
            W[4 + r][col] = W2[r][t]
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    Each decomp is (U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e))
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    try:
        # Split: n=[0,2,4], m=[0,3,5], p=[0,2,6]
        # 8 blocks:
        #   (0,2)x(0,3)x(0,2): 2x3x2, rank 6
        #   (0,2)x(3,5)x(0,2): 2x2x2, rank 7 (Strassen)
        #   (2,4)x(0,3)x(0,2): 2x3x2, rank 6
        #   (2,4)x(3,5)x(0,2): 2x2x2, rank 7 (Strassen)
        #   (0,2)x(0,3)x(2,6): 2x3x4, rank ? 
        #   (0,2)x(3,5)x(2,6): 2x2x4, rank 14
        #   (2,4)x(0,3)x(2,6): 2x3x4, rank ?
        #   (2,4)x(3,5)x(2,6): 2x2x4, rank 14
        
        # Wait, I need to reconsider. Let me use:
        # n=[0,2,4], m=[0,2,5], p=[0,2,6]
        # Blocks:
        #   (0,2)x(0,2)x(0,2): 2x2x2, rank 7
        #   (0,2)x(2,5)x(0,2): 2x3x2, rank 6
        #   (2,4)x(0,2)x(0,2): 2x2x2, rank 7
        #   (2,4)x(2,5)x(0,2): 2x3x2, rank 6
        #   (0,2)x(0,2)x(2,6): 2x2x4, rank 14
        #   (0,2)x(2,5)x(2,6): 2x3x4, rank ?
        #   (2,4)x(0,2)x(2,6): 2x2x4, rank 14
        #   (2,4)x(2,5)x(2,6): 2x3x4, rank ?
        
        # 2x3x4: A is 2x3, B is 3x4, C is 2x4.
        # Split B into 3x2 + 3x2: C = A*B1 + A*B2, each 2x3x2, rank 6 each. Total 12.
        # Or split C into 2x2 + 2x2: each 2x3x2, rank 6. Total 12.
        # So 2x3x4 rank = 12.
        
        # Total: 7+6+7+6+14+12+14+12 = 78
        
        # Let me build a 2x3x4 decomposition (rank 12)
        # A: 2x3, B: 3x4, C: 2x4
        # Split B into B1 (3x2, cols 0-1) and B2 (3x2, cols 2-3)
        # C1 = A*B1 (2x2), C2 = A*B2 (2x2)
        # Each 2x3x2 has rank 6
        
        def _build_2x3x4():
            """
            Build 2x3x4 decomposition. Rank 12.
            A is 2x3, B is 3x4, C is 2x4.
            U shape: (6, 12), V shape: (12, 12), W shape: (8, 12)
            """
            rank = 12
            U = [[0]*rank for _ in range(6)]   # 2*3 = 6 rows
            V = [[0]*rank for _ in range(12)]  # 3*4 = 12 rows
            W = [[0]*rank for _ in range(8)]   # 4*2 = 8 rows (p*n = 4*2)
            
            # Block 1: A * B[:, 0:2] -> C[:, 0:2]
            U1, V1, W1 = _build_2x3x2_schoolbook()
            for r in range(6):
                for c in range(6):
                    U[r][c] = U1[r][c]
            for r in range(6):
                for c in range(6):
                    V[r][c] = V1[r][c]
            for r in range(4):
                for c in range(6):
                    W[r][c] = W1[r][c]
            
            # Block 2: A * B[:, 2:4] -> C[:, 2:4]
            U2, V2, W2 = _build_2x3x2_schoolbook()
            for r in range(6):
                for c in range(6):
                    U[r][6+c] = U2[r][c]
            for r in range(6):
                for c in range(6):
                    V[6+r][6+c] = V2[r][c]
            for r in range(4):
                for c in range(6):
                    W[4+r][6+c] = W2[r][c]
            
            return U, V, W
        
        def _build_2x2x2_strassen():
            """
            Build 2x2x2 Strassen decomposition. Rank 7.
            A is 2x2, B is 2x2, C is 2x2.
            U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
            """
            return _build_strassen_2x2()
        
        # Build all 8 blocks
        decomp1 = (_build_2x2x2_strassen(), (0, 2), (0, 2), (0, 2))      # 2x2x2, rank 7
        decomp2 = (_build_2x3x2_schoolbook(), (0, 2), (2, 5), (0, 2))    # 2x3x2, rank 6
        decomp3 = (_build_2x2x2_strassen(), (2, 4), (0, 2), (0, 2))      # 2x2x2, rank 7
        decomp4 = (_build_2x3x2_schoolbook(), (2, 4), (2, 5), (0, 2))    # 2x3x2, rank 6
        decomp5 = (_build_2x2x4_strassen(), (0, 2), (0, 2), (2, 6))      # 2x2x4, rank 14
        decomp6 = (_build_2x3x4(), (0, 2), (2, 5), (2, 6))               # 2x3x4, rank 12
        decomp7 = (_build_2x2x4_strassen(), (2, 4), (0, 2), (2, 6))      # 2x2x4, rank 14
        decomp8 = (_build_2x3x4(), (2, 4), (2, 5), (2, 6))               # 2x3x4, rank 12
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        
        # Convert to format expected by _combine_decompositions
        decomps = []
        for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in [
            (decomp1[0], decomp1[0], decomp1[0], *decomp1[1]),
        ]:
            pass
        
        # Actually, let me restructure. Each decomp is (U, V, W, i_range, j_range, k_range)
        decomps = [
            (decomp1[0], decomp1[0], decomp1[0], (0, 2), (0, 2), (0, 2)),
            (decomp2[0], decomp2[0], decomp2[0], (0, 2), (2, 5), (0, 2)),
            (decomp3[0], decomp3[0], decomp3[0], (2, 4), (0, 2), (0, 2)),
            (decomp4[0], decomp4[0], decomp4[0], (2, 4), (2, 5), (0, 2)),
            (decomp5[0], decomp5[0], decomp5[0], (0, 2), (0, 2), (2, 6)),
            (decomp6[0], decomp6[0], decomp6[0], (0, 2), (2, 5), (2, 6)),
            (decomp7[0], decomp7[0], decomp7[0], (2, 4), (0, 2), (2, 6)),
            (decomp8[0], decomp8[0], decomp8[0], (2, 4), (2, 5), (2, 6)),
        ]
        
        U, V, W = _combine_decompositions(decomps, n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
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

# EVOLVE-BLOCK-END
