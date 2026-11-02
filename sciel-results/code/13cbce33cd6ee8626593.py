"""Matrix multiplication 3x4x8 with rank-88 decomposition using Strassen blocks."""
import time

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification."""
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]   
    V = [[0] * total_rank for _ in range(m * p)]   
    W = [[0] * total_rank for _ in range(p * n)]   
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1
        
        if btype == 'strassen2':
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
        elif btype == 'school1':
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [
                j0 * p + k0, j1 * p + k0,
                j0 * p + k1, j1 * p + k1
            ]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i=0,1 and 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _find_3x2x2_decomposition(max_time=10):
    """Try to find a rank 10 decomposition for 3x2x2. Fallback to rank 11."""
    import random
    import time
    
    start = time.time()
    n, m, p = 3, 2, 2
    # Target rank 10
    target_rank = 10
    
    # Initialize with random small integers
    def random_init(rank):
        U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W

    def check(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True

    # Try a few random restarts
    for _ in range(50):
        if time.time() - start > max_time:
            break
        U, V, W = random_init(target_rank)
        # Simple local search
        for _ in range(1000):
            if time.time() - start > max_time:
                break
            if check(U, V, W):
                return U, V, W, target_rank
            # Perturb
            mat = random.choice([U, V, W])
            r = random.randint(0, len(mat)-1)
            c = random.randint(0, target_rank-1)
            mat[r][c] += random.choice([-1, 1])
            
    # Fallback: Rank 11 (Strassen + Schoolbook)
    # We can construct this manually or use the existing blocks
    # For 3x2x2:
    # i=0,1: Strassen 2x2x2 (rank 7)
    # i=2: Schoolbook 1x2x2 (rank 4)
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(4)]
    W = [[0]*11 for _ in range(6)]
    
    # Strassen part (i=0,1)
    for i in range(4): # 2x2 = 4 rows
        for c in range(7):
            U[i][c] += U2[i][c]
            V[i][c] += V2[i][c] # Note: V indices for 2x2x2 are 4 rows (j0k0, j0k1, j1k0, j1k1)
            # Wait, V for 2x2x2 has 4 rows. V for 3x2x2 has 4 rows.
            # W for 2x2x2 has 4 rows (k0i0, k1i0, k0i1, k1i1).
            # W for 3x2x2 has 6 rows.
            # The Strassen block covers i=0,1.
            # So W rows 0,1,2,3 correspond to i=0,1.
            W[i][c] += W2[i][c]
            
    # Schoolbook part (i=2)
    # U rows 4,5 (i=2, j=0,1)
    # V rows 0,1,2,3 (j=0,1 k=0,1) - same as Strassen V?
    # Schoolbook V is identity-like?
    # _schoolbook_1x2x2 V is 4x4 identity.
    for i in range(2):
        for c in range(4):
            U[4+i][7+c] += U1[i][c]
            W[4+i][7+c] += W1[i][c] # W rows 4,5 (i=2, k=0,1)
    for i in range(4):
        for c in range(4):
            V[i][7+c] += V1[i][c]
            
    return U, V, W, 11

def _construct_3x4x8():
    """Construct the decomposition."""
    # Try to find a better 3x2x2 decomposition
    U322, V322, W322, r322 = _find_3x2x2_decomposition(max_time=5)
    
    # Build full 3x4x8 decomposition using 8 blocks of 3x2x2
    # Each block covers a 2x2 sub-block of m and p
    # m=4 -> 2 blocks of 2
    # p=8 -> 4 blocks of 2
    # Total 8 blocks
    
    total_rank = 8 * r322
    U = [[0] * total_rank for _ in range(3*4)]
    V = [[0] * total_rank for _ in range(4*8)]
    W = [[0] * total_rank for _ in range(8*3)]
    
    col_offset = 0
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            
            # Map 3x2x2 indices to 3x4x8 indices
            # i is same
            # j: j0, j0+1
            # k: k0, k0+1
            
            for i in range(3):
                for j in range(2):
                    u_idx = i * 4 + (j0 + j)
                    u_block_idx = i * 2 + j
                    for c in range(r322):
                        U[u_idx][col_offset + c] = U322[u_block_idx][c]
                        
            for j in range(2):
                for k in range(2):
                    v_idx = (j0 + j) * 8 + (k0 + k)
                    v_block_idx = j * 2 + k
                    for c in range(r322):
                        V[v_idx][col_offset + c] = V322[v_block_idx][c]
                        
            for k in range(2):
                for i in range(3):
                    w_idx = (k0 + k) * 3 + i
                    w_block_idx = k * 3 + i
                    for c in range(r322):
                        W[w_idx][col_offset + c] = W322[w_block_idx][c]
                        
            col_offset += r322
            
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    return result
