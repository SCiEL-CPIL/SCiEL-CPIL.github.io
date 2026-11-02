"""Matrix multiplication 3,3,5 rank-minimal tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
_START = time.time()
DEADLINE = 150.0
_SAFETY = 5.0  # seconds margin before kill

def _time_left():
    return DEADLINE - _SAFETY - (time.time() - _START)

def _target_tensor(n, m, p):
    """T[i*m+j, j*p+k, k*n+i] = 1 else 0."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T

def _verify(U, V, W, T):
    """Exact integer verification that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # build candidate tensor dict
    cand = {}
    for r in range(rank):
        uc = [U[a][r] for a in range(len(U))]
        vc = [V[b][r] for b in range(len(V))]
        wc = [W[c][r] for c in range(len(W))]
        for a in range(n * m):
            uav = uc[a]
            if uav == 0:
                continue
            for b in range(m * p):
                ubv = uav * vc[b]
                if ubv == 0:
                    continue
                for c in range(p * n):
                    wcv = ubv * wc[c]
                    if wcv == 0:
                        continue
                    cand[(a, b, c)] = cand.get((a, b, c), 0) + wcv
    # compare
    if set(cand.keys()) != set(T.keys()):
        return False
    for key, val in T.items():
        if cand[key] != val:
            return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """
    Rank-7 Strassen factors for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7)
    """
    Ufac = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    Vfac = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    Wfac = [
        [(0, 1), (3, -1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, -1)],
    ]
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    for r in range(rank):
        for (idx, val) in Ufac[r]:
            U[idx][r] += val
        for (idx, val) in Vfac[r]:
            V[idx][r] += val
        for (idx, val) in Wfac[r]:
            W[idx][r] += val
    
    return U, V, W

def _block_candidate(n, m, p):
    """
    Build a candidate by replacing 2x2x2 sub-blocks with Strassen rank-7 decomposition.
    Try all possible placements and combinations.
    """
    sU, sV, sW = _strassen_222()
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb["U"]]
    V = [row[:] for row in sb["V"]]
    W = [row[:] for row in sb["W"]]
    rank0 = n * m * p  # 45
    
    # Find all 2x2x2 sub-blocks
    sub_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                sub_blocks.append((i0, j0, k0))
    
    def get_sub_block_cols(i0, j0, k0):
        cols = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i0 <= i < i0 + 2 and j0 <= j < j0 + 2 and k0 <= k < k0 + 2:
                        cols.append(col)
                    col += 1
        return cols
    
    # Try each sub-block independently
    best_candidate = None
    best_rank = rank0 + 1
    
    for (i0, j0, k0) in sub_blocks:
        if _time_left() <= 1.0:
            break
        
        sub_cols = get_sub_block_cols(i0, j0, k0)
        
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        for c in sub_cols:
            for row in new_U:
                row[c] = 0
            for row in new_V:
                row[c] = 0
            for row in new_W:
                row[c] = 0
        
        new_rank = rank0 + 7
        for row in new_U:
            row += [0] * 7
        for row in new_V:
            row += [0] * 7
        for row in new_W:
            row += [0] * 7
        
        for r in range(7):
            for a in range(4):
                i = i0 + a // 2
                j = j0 + a % 2
                new_U[i * m + j][rank0 + r] += sU[a][r]
            for b in range(4):
                j = j0 + b // 2
                k = k0 + b % 2
                new_V[j * p + k][rank0 + r] += sV[b][r]
            for c in range(4):
                k = k0 + c // 2
                i = i0 + c % 2
                new_W[k * n + i][rank0 + r] += sW[c][r]
        
        active_cols = []
        for c in range(new_rank):
            if any(new_U[a][c] for a in range(n * m)) or \
               any(new_V[b][c] for b in range(m * p)) or \
               any(new_W[c2][c] for c2 in range(p * n)):
                active_cols.append(c)
        
        final_rank = len(active_cols)
        if final_rank < best_rank:
            best_rank = final_rank
            best_U = [[new_U[a][c] for c in active_cols] for a in range(n * m)]
            best_V = [[new_V[b][c] for c in active_cols] for b in range(m * p)]
            best_W = [[new_W[c2][c] for c in active_cols] for c2 in range(p * n)]
            best_candidate = {"U": best_U, "V": best_V, "W": best_W}
    
    if best_candidate is None:
        return {"U": U, "V": V, "W": W}
    
    return best_candidate

def _build_333_strassen():
    """
    Build a rank-23 decomposition for 3x3x3 matrix multiplication using
    Strassen-like techniques. We'll use the known best rank for 3x3x3 which
    is 23 (Lader 1976, Pan 1978).
    
    Actually, let's use a simpler approach: combine schoolbook for the parts
    not covered by 2x2x2 Strassen blocks, and use Strassen for the 2x2x2 blocks.
    """
    n, m, p = 3, 3, 3
    sU, sV, sW = _strassen_222()
    
    # Schoolbook rank is 27
    # Using one 2x2x2 Strassen block saves 1 rank -> 26
    # But we can do better with multiple blocks
    
    # Let's just use the block candidate approach for 3x3x3
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb["U"]]
    V = [row[:] for row in sb["V"]]
    W = [row[:] for row in sb["W"]]
    rank0 = 27
    
    # Try all 2x2x2 sub-blocks
    sub_blocks = []
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(2):
                sub_blocks.append((i0, j0, k0))
    
    def get_sub_block_cols(i0, j0, k0):
        cols = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i0 <= i < i0 + 2 and j0 <= j < j0 + 2 and k0 <= k < k0 + 2:
                        cols.append(col)
                    col += 1
        return cols
    
    best_rank = rank0 + 1
    best_candidate = None
    
    for (i0, j0, k0) in sub_blocks:
        sub_cols = get_sub_block_cols(i0, j0, k0)
        
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        for c in sub_cols:
            for row in new_U:
                row[c] = 0
            for row in new_V:
                row[c] = 0
            for row in new_W:
                row[c] = 0
        
        new_rank = rank0 + 7
        for row in new_U:
            row += [0] * 7
        for row in new_V:
            row += [0] * 7
        for row in new_W:
            row += [0] * 7
        
        for r in range(7):
            for a in range(4):
                i = i0 + a // 2
                j = j0 + a % 2
                new_U[i * m + j][rank0 + r] += sU[a][r]
            for b in range(4):
                j = j0 + b // 2
                k = k0 + b % 2
                new_V[j * p + k][rank0 + r] += sV[b][r]
            for c in range(4):
                k = k0 + c // 2
                i = i0 + c % 2
                new_W[k * n + i][rank0 + r] += sW[c][r]
        
        active_cols = []
        for c in range(new_rank):
            if any(new_U[a][c] for a in range(n * m)) or \
               any(new_V[b][c] for b in range(m * p)) or \
               any(new_W[c2][c] for c2 in range(p * n)):
                active_cols.append(c)
        
        final_rank = len(active_cols)
        if final_rank < best_rank:
            best_rank = final_rank
            best_U = [[new_U[a][c] for c in active_cols] for a in range(n * m)]
            best_V = [[new_V[b][c] for c in active_cols] for b in range(m * p)]
            best_W = [[new_W[c2][c] for c in active_cols] for c2 in range(p * n)]
            best_candidate = {"U": best_U, "V": best_V, "W": best_W}
    
    if best_candidate is None:
        return {"U": U, "V": V, "W": W}
    
    return best_candidate

def _extended_block_candidate(n, m, p):
    """
    More sophisticated block candidate: try overlapping 2x2x2 blocks
    and also consider 2x3x2, 3x2x2, 2x2x3 blocks with appropriate
    decompositions.
    
    For now, let's try multiple non-overlapping 2x2x2 blocks.
    """
    sU, sV, sW = _strassen_222()
    
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb["U"]]
    V = [row[:] for row in sb["V"]]
    W = [row[:] for row in sb["W"]]
    rank0 = n * m * p
    
    def get_sub_block_cols(i0, j0, k0, di, dj, dk):
        cols = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i0 <= i < i0 + di and j0 <= j < j0 + dj and k0 <= k < k0 + dk:
                        cols.append(col)
                    col += 1
        return cols
    
    # Try single 2x2x2 blocks
    best_rank = rank0 + 1
    best_candidate = None
    
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                if _time_left() <= 1.0:
                    break
                
                sub_cols = get_sub_block_cols(i0, j0, k0, 2, 2, 2)
                
                new_U = [row[:] for row in U]
                new_V = [row[:] for row in V]
                new_W = [row[:] for row in W]
                
                for c in sub_cols:
                    for row in new_U:
                        row[c] = 0
                    for row in new_V:
                        row[c] = 0
                    for row in new_W:
                        row[c] = 0
                
                new_rank = rank0 + 7
                for row in new_U:
                    row += [0] * 7
                for row in new_V:
                    row += [0] * 7
                for row in new_W:
                    row += [0] * 7
                
                for r in range(7):
                    for a in range(4):
                        i = i0 + a // 2
                        j = j0 + a % 2
                        new_U[i * m + j][rank0 + r] += sU[a][r]
                    for b in range(4):
                        j = j0 + b // 2
                        k = k0 + b % 2
                        new_V[j * p + k][rank0 + r] += sV[b][r]
                    for c in range(4):
                        k = k0 + c // 2
                        i = i0 + c % 2
                        new_W[k * n + i][rank0 + r] += sW[c][r]
                
                active_cols = []
                for c in range(new_rank):
                    if any(new_U[a][c] for a in range(n * m)) or \
                       any(new_V[b][c] for b in range(m * p)) or \
                       any(new_W[c2][c] for c2 in range(p * n)):
                        active_cols.append(c)
                
                final_rank = len(active_cols)
                if final_rank < best_rank:
                    best_rank = final_rank
                    best_U = [[new_U[a][c] for c in active_cols] for a in range(n * m)]
                    best_V = [[new_V[b][c] for c in active_cols] for b in range(m * p)]
                    best_W = [[new_W[c2][c] for c in active_cols] for c2 in range(p * n)]
                    best_candidate = {"U": best_U, "V": best_V, "W": best_W}
    
    if best_candidate is None:
        return {"U": U, "V": V, "W": W}
    
    return best_candidate

def construct():
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    
    candidates = []
    
    # 1. Extended block candidate
    try:
        if _time_left() > 5.0:
            candidates.append(_extended_block_candidate(n, m, p))
    except Exception as e:
        pass
    
    # 2. Basic block candidate
    try:
        if _time_left() > 3.0:
            candidates.append(_block_candidate(n, m, p))
    except Exception as e:
        pass
    
    # 3. Schoolbook (rank 45, guaranteed valid)
    candidates.append(_schoolbook(n, m, p))
    
    best = None
    best_rank = None
    
    for cand in candidates:
        if _time_left() <= 2.0:
            break
        r = len(cand["U"][0]) if cand["U"] else 0
        if best_rank is not None and r >= best_rank:
            continue
        try:
            if _verify(cand["U"], cand["V"], cand["W"], T):
                best = cand
                best_rank = r
                if best_rank <= 40:
                    break
        except Exception:
            continue
    
    if best is None:
        best = _schoolbook(n, m, p)
    
    return best
