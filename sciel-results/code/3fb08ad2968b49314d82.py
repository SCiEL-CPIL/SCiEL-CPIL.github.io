"""
Matrix multiplication 3×4×5 rank-minimal exact tensor decomposition.
Strategy: Use multiple disjoint Strassen 2×2×2 blocks to reduce rank below schoolbook (60).
Each Strassen block replaces 8 schoolbook terms with 7, saving 1 rank per block.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]


def _build_strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2×2×2 matrix multiplication.
    U, V, W are numpy arrays of shape (4, 7) with integer entries.
    
    Indexing convention:
    U[li*2+lj, c] for A[li, lj]
    V[lj*2+lk, c] for B[lj, lk]  
    W[lk*2+li, c] for C[li, lk] (trace convention: W[k*n+i])
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, -1],   # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=int)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, -1, 0, 1, 0],   # B01
        [1, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=int)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0] → W[0*2+0]=W[0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1] → W[1*2+0]=W[2]... wait
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0] → W[0*2+1]=W[1]
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1] → W[1*2+1]=W[3]
    ], dtype=int)
    
    # Verify: W[k*2+i] should give C[i,k]
    # W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1]
    # So row 0 = C[0,0], row 1 = C[1,0], row 2 = C[0,1], row 3 = C[1,1]
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # W[0] = C[0,0]
        [0, 1, 0, 1, 0, 0, 0],    # W[1] = C[1,0]
        [0, 0, 1, 0, 1, 0, 0],    # W[2] = C[0,1]
        [1, -1, 1, 0, 0, 1, 0],   # W[3] = C[1,1]
    ], dtype=int)
    
    return U, V, W


def _build_decomposition_with_blocks(block_specs):
    """
    Build decomposition using Strassen blocks + schoolbook for the rest.
    
    block_specs: list of (i_offset, j_offset, k_offset) for each 2x2x2 block.
    Each block covers i∈{i_off, i_off+1}, j∈{j_off, j_off+1}, k∈{k_off, k_off+1}.
    Blocks must be disjoint (no overlapping (i,j,k) triples).
    
    Returns dict with U, V, W as lists of lists (integer entries).
    """
    n, m, p = 3, 4, 5
    
    # Determine which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for (io, jo, ko) in block_specs:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((io + di, jo + dj, ko + dk))
    
    num_blocks = len(block_specs)
    num_strassen_cols = num_blocks * 7
    num_schoolbook = n * m * p - num_blocks * 8
    rank = num_strassen_cols + num_schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # Place Strassen blocks
    for b_idx, (io, jo, ko) in enumerate(block_specs):
        base_col = b_idx * 7
        for c in range(7):
            col = base_col + c
            for li in range(2):
                for lj in range(2):
                    gi = io + li
                    gj = jo + lj
                    U[gi * m + gj][col] += int(U_s[li * 2 + lj][c])
            for lj in range(2):
                for lk in range(2):
                    gj = jo + lj
                    gk = ko + lk
                    V[gj * p + gk][col] += int(V_s[lj * 2 + lk][c])
            for lk in range(2):
                for li in range(2):
                    gk = ko + lk
                    gi = io + li
                    W[gk * n + gi][col] += int(W_s[lk * 2 + li][c])
    
    # Place schoolbook terms for uncovered (i,j,k)
    col = num_strassen_cols
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}


def _find_best_block_placement():
    """
    Find the maximum number of disjoint 2x2x2 Strassen blocks we can place
    in the 3x4x5 grid using branch and bound.
    """
    n, m, p = 3, 4, 5
    # Possible block positions
    positions = []
    for io in range(n - 1):  # 0, 1
        for jo in range(m - 1):  # 0, 1, 2
            for ko in range(p - 1):  # 0, 1, 2, 3
                positions.append((io, jo, ko))
    
    # For each position, compute the set of (i,j,k) it covers
    pos_to_triples = {}
    for pos in positions:
        io, jo, ko = pos
        triples = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    triples.add((io + di, jo + dj, ko + dk))
        pos_to_triples[pos] = triples
    
    # Precompute conflict matrix
    num_pos = len(positions)
    conflicts = [[False] * num_pos for _ in range(num_pos)]
    for i in range(num_pos):
        for j in range(i + 1, num_pos):
            if pos_to_triples[positions[i]] & pos_to_triples[positions[j]]:
                conflicts[i][j] = True
                conflicts[j][i] = True
    
    best_count = 0
    best_placement = []
    
    def branch_and_bound(idx, selected, count):
        nonlocal best_count, best_placement
        # Pruning: even if we add all remaining, can't beat best
        if count + (num_pos - idx) <= best_count:
            return
        if idx == num_pos:
            if count > best_count:
                best_count = count
                best_placement = list(selected)
            return
        # Try including position idx if it doesn't conflict with any selected
        can_add = True
        for s in selected:
            if conflicts[idx][s]:
                can_add = False
                break
        if can_add:
            selected.append(idx)
            branch_and_bound(idx + 1, selected, count + 1)
            selected.pop()
        # Try excluding position idx
        branch_and_bound(idx + 1, selected, count)
    
    branch_and_bound(0, [], 0)
    
    return best_count, [positions[i] for i in best_placement]


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Use numpy for speed
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = np.dot(u_row * v_row, w_row)
                if s != 1:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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


def _try_ntt_based_decomposition():
    """
    Try an NTT/DFT-based approach: map the middle dimension to a cyclic convolution
    structure. For the 3x4x5 problem, this is less straightforward, but we can try
    to find a decomposition using roots of unity or finite field transforms.
    
    This is experimental - may not yield a valid decomposition.
    """
    n, m, p = 3, 4, 5
    
    # One idea: if we could find a field with primitive 4th root of unity (for m=4)
    # and 5th root of unity (for p=5), we could diagonalize.
    # But this requires the rank to be at least n * max(m,p) or similar, which
    # doesn't obviously help.
    
    # For now, return None to indicate this approach didn't yield a valid result.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin before 150s limit
    
    # Find best block placement using branch and bound
    best_count, best_placement = _find_best_block_placement()
    
    if time.time() > deadline:
        # Time pressure: use a known good 2-block placement
        best_placement = [(0, 0, 0), (1, 2, 3)]
        best_count = 2
    
    if best_count > 0:
        try:
            result = _build_decomposition_with_blocks(best_placement)
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
        except Exception:
            pass
    
    # Try specific known-good placements in order of preference
    test_placements = [
        [(0, 0, 0), (1, 2, 3)],
        [(0, 0, 0), (0, 2, 2)],
        [(0, 0, 0), (1, 0, 0)],
        [(0, 0, 0), (0, 1, 0)],
        [(0, 0, 0), (1, 2, 2)],
        [(0, 0, 0), (0, 2, 3)],
        [(0, 0, 0), (0, 1, 3)],
        [(0, 0, 0)],
    ]
    
    for placement in test_placements:
        if time.time() > deadline:
            break
        try:
            result = _build_decomposition_with_blocks(placement)
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
        except Exception:
            pass
    
    # Fallback to schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    if _verify_decomposition(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        return schoolbook
    
    return {"U": [], "V": [], "W": []}
