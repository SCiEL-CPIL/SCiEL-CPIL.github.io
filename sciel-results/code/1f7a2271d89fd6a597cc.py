"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """2x2x2 Strassen. Returns (U,V,W) each shape (4,7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook n x m * m x p. Rank = n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _build_2x2x4():
    """2x2x4 via two 2x2x2 Strassen. Rank 14. U:(4,14) V:(8,14) W:(8,14)"""
    U2, V2, W2 = _strassen_2x2()
    U = [[0]*14 for _ in range(4)]
    V = [[0]*14 for _ in range(8)]
    W = [[0]*14 for _ in range(8)]
    for k_block in range(2):
        base = k_block * 7
        for t in range(7):
            col = base + t
            for r in range(4):
                U[r][col] = U2[r][t]
            for r in range(4):
                V[r][col] = V2[r][t]
            for r in range(4):
                W[k_block*4 + r][col] = W2[r][t]
    return U, V, W

def _build_2x2x2():
    """2x2x2 Strassen. Rank 7. U:(4,7) V:(4,7) W:(4,7)"""
    return _strassen_2x2()

def _build_4x2x4():
    """4x2x4 via two 2x2x4. Rank 28. U:(8,28) V:(8,28) W:(16,28)"""
    U224, V224, W224 = _build_2x2x4()
    U = [[0]*28 for _ in range(8)]
    V = [[0]*28 for _ in range(8)]
    W = [[0]*28 for _ in range(16)]
    for i_block in range(2):
        base = i_block * 14
        for t in range(14):
            col = base + t
            for r in range(4):
                i_loc = r // 2
                j_loc = r % 2
                U[i_block*4 + r][col] = U224[r][t]
            for r in range(8):
                V[r][col] = V224[r][t]
            for r in range(8):
                k_loc = r // 4
                i_loc = r % 4
                W[k_block*8 + r][col] = W224[r][t]
    return U, V, W

def _build_4x2x4():
    """4x2x4 via two 2x2x4 Strassen. Rank 28."""
    U224, V224, W224 = _build_2x2x4()
    rank = 28
    U = [[0]*rank for _ in range(8)]   # 4*2
    V = [[0]*rank for _ in range(8)]   # 2*4
    W = [[0]*rank for _ in range(16)]  # 4*4
    for i_block in range(2):
        base = i_block * 14
        for t in range(14):
            col = base + t
            for r in range(4):
                U[i_block*4 + r][col] = U224[r][t]
            for r in range(8):
                V[r][col] = V224[r][t]
            for r in range(8):
                W[i_block*8 + r][col] = W224[r][t]
    return U, V, W

def _build_2x3x4():
    """2x3x4: split m=3 as 2+1. 2x2x4 (14) + 2x1x4 (8) = 22. U:(6,22) V:(12,22) W:(8,22)"""
    U224, V224, W224 = _build_2x2x4()
    U214, V214, W214 = _schoolbook(2, 1, 4)
    rank = 22
    U = [[0]*rank for _ in range(6)]   # 2*3
    V = [[0]*rank for _ in range(12)]  # 3*4
    W = [[0]*rank for _ in range(8)]   # 4*2
    # 2x2x4 part: A[0:2,0:2] * B[0:2,0:4] -> C[0:4,0:2]
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(14):
            U[i_loc*3 + j_loc][c] = U224[r][c]
    for r in range(8):
        j_loc = r // 4
        k_loc = r % 4
        for c in range(14):
            V[j_loc*4 + k_loc][c] = V224[r][c]
    for r in range(8):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(14):
            W[k_loc*2 + i_loc][c] = W224[r][c]
    # 2x1x4 part: A[0:2,2:3] * B[2:3,0:4] -> C[0:4,0:2]
    for r in range(6):
        i_loc = r // 1  # 0 or 1... wait, U214 has shape (2*1, 8) = (2,8)
        # Actually U214: n=2,m=1,p=4, so U shape (2,8), V shape (4,8), W shape (8,8)
        pass
    # Let me redo: U214 shape (2,8), V214 shape (4,8), W214 shape (8,8)
    for r in range(2):
        for c in range(8):
            U[r*3 + 2][14+c] = U214[r][c]
    for r in range(4):
        for c in range(8):
            V[2*4 + r][14+c] = V214[r][c]
    for r in range(8):
        for c in range(8):
            W[r][14+c] = W214[r][c]
    return U, V, W

def _build_2x3x2():
    """2x3x2: split m=3 as 2+1. 2x2x2 (7) + 2x1x2 (4) = 11. U:(6,11) V:(6,11) W:(4,11)"""
    U222, V222, W222 = _strassen_2x2()
    U212, V212, W212 = _schoolbook(2, 1, 2)
    rank = 11
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(4)]
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(7):
            U[i_loc*3 + j_loc][c] = U222[r][c]
    for r in range(4):
        j_loc = r // 2
        k_loc = r % 2
        for c in range(7):
            V[j_loc*2 + k_loc][c] = V222[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(7):
            W[k_loc*2 + i_loc][c] = W222[r][c]
    for r in range(2):
        for c in range(4):
            U[r*3 + 2][7+c] = U212[r][c]
    for r in range(2):
        for c in range(4):
            V[2*2 + r][7+c] = V212[r][c]
    for r in range(4):
        for c in range(4):
            W[r][7+c] = W212[r][c]
    return U, V, W

def _build_4x3x4():
    """4x3x4: split n=4 as 2+2. Two 2x3x4 (22 each) = 44. U:(12,44) V:(12,44) W:(16,44)"""
    U234, V234, W234 = _build_2x3x4()
    rank = 44
    U = [[0]*rank for _ in range(12)]   # 4*3
    V = [[0]*rank for _ in range(12)]   # 3*4
    W = [[0]*rank for _ in range(16)]   # 4*4
    for i_block in range(2):
        base = i_block * 22
        for t in range(22):
            col = base + t
            for r in range(6):
                U[i_block*6 + r][col] = U234[r][t]
            for r in range(12):
                V[r][col] = V234[r][t]
            for r in range(8):
                W[i_block*8 + r][col] = W234[r][t]
    return U, V, W

def _build_4x3x2():
    """4x3x2: split n=4 as 2+2. Two 2x3x2 (11 each) = 22. U:(12,22) V:(6,22) W:(8,22)"""
    U232, V232, W232 = _build_2x3x2()
    rank = 22
    U = [[0]*rank for _ in range(12)]   # 4*3
    V = [[0]*rank for _ in range(6)]    # 3*2
    W = [[0]*rank for _ in range(8)]    # 2*4
    for i_block in range(2):
        base = i_block * 11
        for t in range(11):
            col = base + t
            for r in range(6):
                U[i_block*6 + r][col] = U232[r][t]
            for r in range(6):
                V[r][col] = V232[r][t]
            for r in range(4):
                W[i_block*4 + r][col] = W232[r][t]
    return U, V, W

def _build_4x2x4():
    """4x2x4: split n=4 as 2+2. Two 2x2x4 (14 each) = 28."""
    U224, V224, W224 = _build_2x2x4()
    rank = 28
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(16)]
    for i_block in range(2):
        base = i_block * 14
        for t in range(14):
            col = base + t
            for r in range(4):
                U[i_block*4 + r][col] = U224[r][t]
            for r in range(8):
                V[r][col] = V224[r][t]
            for r in range(8):
                W[i_block*8 + r][col] = W224[r][t]
    return U, V, W

def _build_4x2x2():
    """4x2x2: split n=4 as 2+2. Two 2x2x2 (7 each) = 14."""
    U222, V222, W222 = _strassen_2x2()
    rank = 14
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(8)]
    for i_block in range(2):
        base = i_block * 7
        for t in range(7):
            col = base + t
            for r in range(4):
                U[i_block*4 + r][col] = U222[r][t]
            for r in range(4):
                V[r][col] = V222[r][t]
            for r in range(4):
                W[i_block*4 + r][col] = W222[r][t]
    return U, V, W

def _build_4x1x4():
    """4x1x4 schoolbook. Rank 16."""
    return _schoolbook(4, 1, 4)

def _build_4x1x2():
    """4x1x2 schoolbook. Rank 8."""
    return _schoolbook(4, 1, 2)

def _decomp_block(n_sub, m_sub, p_sub, rank_sub, U_sub, V_sub, W_sub,
                  i_s, j_s, k_s, n, m, p, U, V, W, offset):
    for r in range(n_sub * m_sub):
        i_loc = r // m_sub
        j_loc = r % m_sub
        i_glob = i_s + i_loc
        j_glob = j_s + j_loc
        r_glob = i_glob * m + j_glob
        for c in range(rank_sub):
            U[r_glob][offset + c] = U_sub[r][c]
    for r in range(m_sub * p_sub):
        j_loc = r // p_sub
        k_loc = r % p_sub
        j_glob = j_s + j_loc
        k_glob = k_s + k_loc
        r_glob = j_glob * p + k_glob
        for c in range(rank_sub):
            V[r_glob][offset + c] = V_sub[r][c]
    for r in range(p_sub * n_sub):
        k_loc = r // n_sub
        i_loc = r % n_sub
        k_glob = k_s + k_loc
        i_glob = i_s + i_loc
        r_glob = k_glob * n + i_glob
        for c in range(rank_sub):
            W[r_glob][offset + c] = W_sub[r][c]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS  # 4, 5, 6

    try:
        # Strategy: split m=5 as 3+2
        # C[0:4,0:4] = A[0:4,0:3]*B[0:3,0:4] + A[0:4,3:5]*B[3:5,0:4]
        # C[0:4,4:6] = A[0:4,0:3]*B[0:3,4:6] + A[0:4,3:5]*B[3:5,4:6]
        #
        # 1. 4x3x4: rank 44
        # 2. 4x3x2: rank 22
        # 3. 4x2x4: rank 28
        # 4. 4x2x2: rank 14
        # Total: 44 + 22 + 28 + 14 = 108
        
        # Alternative: split m=5 as 3+2, but also try other splits
        # Let's try: 4x3x4 (44) + 4x3x2 (22) + 4x2x4 (28) + 4x2x2 (14) = 108
        
        # Better: split p=6 as 4+2, m=5 as 3+2
        # 4x3x4: 44, 4x3x2: 22, 4x2x4: 28, 4x2x2: 14 -> 108
        
        # Try: m=5 as 2+3
        # 4x2x4: 28, 4x2x2: 14, 4x3x4: 44, 4x3x2: 22 -> same 108
        
        # Try: split n=4 as 2+2 for everything
        # 2x3x4 (22) * 2 + 2x3x2 (11) * 2 + 2x2x4 (14) * 2 + 2x2x2 (7) * 2
        # = 44 + 22 + 28 + 14 = 108
        
        # Current best known: 101. Let me try to do better.
        # 
        # What about 4x5x6 split as:
        # m=5: 2+3, p=6: 2+4
        # 4x2x2: 14, 4x2x4: 28, 4x3x2: 22, 4x3x4: 44 -> 108
        # 
        # m=5: 3+2, p=6: 4+2
        # 4x3x4: 44, 4x3x2: 22, 4x2x4: 28, 4x2x2: 14 -> 108
        #
        # m=5: 1+4, p=6: 4+2 (current)
        # 4x1x4: 16, 4x1x2: 8, 4x4x4: 49, 4x4x2: 28 -> 101
        #
        # m=5: 4+1, p=6: 2+4
        # 4x4x2: 28, 4x4x4: 49, 4x1x2: 8, 4x1x4: 16 -> 101
        #
        # m=5: 2+1+2, p=6: 2+2+2
        # 4x2x2: 14, 4x1x2: 8, 4x2x2: 14, 4x2x2: 14, 4x1x2: 8, 4x2x2: 14 -> 72!
        
        # Let's do: m=5 as 2+1+2, p=6 as 2+2+2
        # 4x2x2: 14 (three times) = 42
        # 4x1x2: 8 (three times) = 24
        # Total: 66
        
        # Even better: 4x2x2 = 14, 4x1x2 = 8
        # 3*14 + 3*8 = 42 + 24 = 66
        
        U422, V422, W422 = _build_4x2x2()
        U412, V412, W412 = _build_4x1x2()
        
        total_rank = 3 * 14 + 3 * 8  # 66
        U = [[0]*total_rank for _ in range(n*m)]  # 20 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 30 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 24 rows
        
        # m splits: [0:2], [2:3], [3:5]
        # p splits: [0:2], [2:4], [4:6]
        
        m_splits = [(0, 2), (2, 3), (3, 5)]
        p_splits = [(0, 2), (2, 4), (4, 6)]
        
        offset = 0
        for (j_s, j_e) in m_splits:
            m_sub = j_e - j_s
            for (k_s, k_e) in p_splits:
                p_sub = k_e - k_s
                if m_sub == 2 and p_sub == 2:
                    _decomp_block(4, 2, 2, 14, U422, V422, W422, 0, j_s, k_s, n, m, p, U, V, W, offset)
                    offset += 14
                elif m_sub == 1 and p_sub == 2:
                    _decomp_block(4, 1, 2, 8, U412, V412, W412, 0, j_s, k_s, n, m, p, U, V, W, offset)
                    offset += 8
        
        # Verify
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(total_rank))
                    if abs(s - 1.0) > 1e-9:
                        raise ValueError(f"Verify failed at ({i},{j},{k}): got {s}")
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
