"""Matrix multiplication 3x5x6: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    # Pre-compute minimal ranks and best splits for small blocks
    # R[a][b][c] = minimal rank for a x b x c block
    # best_split[a][b][c] = (type, args) where type is 'base', 'split_a', 'split_b', 'split_c'
    # For base: 'strassen' if 2x2x2, 'diag' if 1x1x1 or any 1 in dim
    # We only need to go up to max(n, m, p)
    max_dim = max(n, m, p)
    
    # Initialize R with infinity
    R = [[[10**9 for _ in range(max_dim + 1)] for _ in range(max_dim + 1)] for _ in range(max_dim + 1)]
    best_split = [[[None for _ in range(max_dim + 1)] for _ in range(max_dim + 1)] for _ in range(max_dim + 1)]
    
    # Base cases
    for a in range(1, max_dim + 1):
        for b in range(1, max_dim + 1):
            for c in range(1, max_dim + 1):
                if a == 1 and b == 1 and c == 1:
                    R[a][b][c] = 1
                    best_split[a][b][c] = ('diag',)
                elif a == 1 or b == 1 or c == 1:
                    R[a][b][c] = a * b * c
                    best_split[a][b][c] = ('diag',)
                elif a == 2 and b == 2 and c == 2:
                    R[a][b][c] = 7
                    best_split[a][b][c] = ('strassen',)

    # Fill R and best_split using dynamic programming
    # We iterate by total size a+b+c to ensure smaller sizes are computed first
    for total_size in range(3, max_dim * 3 + 1):
        for a in range(1, max_dim + 1):
            for b in range(1, max_dim + 1):
                for c in range(1, max_dim + 1):
                    if a + b + c != total_size:
                        continue
                    if R[a][b][c] < 10**9:
                        continue # Already set by base case or previous iteration? 
                        # Actually, base cases are set. We need to check if a split is better.
                    
                    # Try splitting a
                    for a1 in range(1, a):
                        a2 = a - a1
                        if R[a1][b][c] + R[a2][b][c] < R[a][b][c]:
                            R[a][b][c] = R[a1][b][c] + R[a2][b][c]
                            best_split[a][b][c] = ('split_a', a1)
                    # Try splitting b
                    for b1 in range(1, b):
                        b2 = b - b1
                        if R[a][b1][c] + R[a][b2][c] < R[a][b][c]:
                            R[a][b][c] = R[a][b1][c] + R[a][b2][c]
                            best_split[a][b][c] = ('split_b', b1)
                    # Try splitting c
                    for c1 in range(1, c):
                        c2 = c - c1
                        if R[a][b][c1] + R[a][b][c2] < R[a][b][c]:
                            R[a][b][c] = R[a][b][c1] + R[a][b][c2]
                            best_split[a][b][c] = ('split_c', c1)

    def gen_partitions(size):
        """Generate all ordered partitions of [0..size-1]."""
        if size <= 0:
            yield []
            return
        if size == 1:
            yield [[0]]
            return
        results = []
        for first_size in range(1, size + 1):
            first_group = list(range(first_size))
            for rest in gen_partitions(size - first_size):
                shifted = [[x + first_size for x in g] for g in rest]
                results.append([first_group] + shifted)
        yield from results

    i_parts = list(gen_partitions(n))
    j_parts = list(gen_partitions(m))
    k_parts = list(gen_partitions(p))

    best_rank = n * m * p
    best_partition = None

    # Search for best partition
    for ip in i_parts:
        for jp in j_parts:
            for kp in k_parts:
                if time.time() > deadline - 10:
                    break
                rank = 0
                for ig in ip:
                    for jg in jp:
                        for kg in kp:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            rank += R[si][sj][sk]
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (ip, jp, kp)
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    if best_partition is None:
        # Fallback to schoolbook
        best_partition = ([list(range(n))], [list(range(m))], [list(range(p))])
        best_rank = n * m * p

    ip, jp, kp = best_partition

    # Construct decomposition
    total_rank = best_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0

    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    def strassen_terms():
        # Returns list of (u_pat, v_pat, w_pat)
        terms = []
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}))
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}))
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    STRASSEN = strassen_terms()

    def build_block(ig, jg, kg, offset_i, offset_j, offset_k):
        """
        Recursively build the decomposition for a block defined by index groups ig, jg, kg.
        offset_i, offset_j, offset_k are the starting indices in the global U, V, W matrices.
        """
        nonlocal col
        si, sj, sk = len(ig), len(jg), len(kg)
        
        if si == 0 or sj == 0 or sk == 0:
            return

        split_info = best_split[si][sj][sk]
        
        if split_info is None:
            # Should not happen if R is computed correctly
            return

        if split_info[0] == 'diag':
            # Direct multiplication
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        u_idx = (ig[li] + offset_i) * m + (jg[lj] + offset_j)
                        v_idx = (jg[lj] + offset_j) * p + (kg[lk] + offset_k)
                        w_idx = (kg[lk] + offset_k) * n + (ig[li] + offset_i)
                        add_term(u_idx, v_idx, w_idx, 1)
        elif split_info[0] == 'strassen':
            # Strassen decomposition for 2x2x2
            for u_pat, v_pat, w_pat in STRASSEN:
                for (li, lj), val_u in u_pat.items():
                    U[(ig[li] + offset_i) * m + (jg[lj] + offset_j)][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    V[(jg[lj] + offset_j) * p + (kg[lk] + offset_k)][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    W[(kg[lk] + offset_k) * n + (ig[li] + offset_i)][col] = val_w
                col += 1
        elif split_info[0] == 'split_a':
            a1 = split_info[1]
            ig1 = ig[:a1]
            ig2 = ig[a1:]
            build_block(ig1, jg, kg, offset_i, offset_j, offset_k)
            build_block(ig2, jg, kg, offset_i + a1, offset_j, offset_k)
        elif split_info[0] == 'split_b':
            b1 = split_info[1]
            jg1 = jg[:b1]
            jg2 = jg[b1:]
            build_block(ig, jg1, kg, offset_i, offset_j, offset_k)
            build_block(ig, jg2, kg, offset_i, offset_j + b1, offset_k)
        elif split_info[0] == 'split_c':
            c1 = split_info[1]
            kg1 = kg[:c1]
            kg2 = kg[c1:]
            build_block(ig, jg, kg1, offset_i, offset_j, offset_k)
            build_block(ig, jg, kg2, offset_i, offset_j, offset_k + c1)

    # Iterate over all blocks in the best partition
    for ig in ip:
        for jg in jp:
            for kg in kp:
                # The offsets are determined by the position of the group in the partition
                # We need to track the global offset for each group
                # Since gen_partitions generates contiguous groups, the offset is just the first element of the group
                # But wait, the groups are lists of indices.
                # ig[0] is the first row index.
                offset_i = ig[0]
                offset_j = jg[0]
                offset_k = kg[0]
                build_block(ig, jg, kg, offset_i, offset_j, offset_k)

    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, best_rank):
        # Fallback to schoolbook if verification fails
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
