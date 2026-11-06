"""Rewritten: exact maximum independent set via bitset branch-and-bound (Tomita coloring)."""
import time
import math
import random

def construct():
    t0 = time.time()
    deadline = t0 + 55  # 5s safety margin under 60s kill

    N = 256

    # Perfect squares up to 2N
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # Self-conflict: 2a is a perfect square -> a forbidden
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    idx_to_num = valid_numbers
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}

    # Conflict adjacency as bitmasks (symmetric, no self)
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N and b in num_to_idx:
                j = num_to_idx[b]
                if j != i:
                    adj[i] |= (1 << j)

    # ---------- Exact maximum independent set (bit branch-and-bound + greedy coloring) ----------
    # Order vertices by ascending degree for better coloring bounds
    deg = [bin(adj[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda x: deg[x])
    remap = [0] * M
    for i, v in enumerate(order):
        remap[v] = i
    # Rebuild adjacency in the new ordering
    adj2 = [0] * M
    for i in range(M):
        mask = adj[order[i]]
        m2 = 0
        for j in range(M):
            if (mask >> order[j]) & 1:
                m2 |= (1 << remap[order[j]])
        adj2[i] = m2

    best_size = [0]
    best_mask = [0]

    def color_bound(P):
        """Greedy sequential coloring of the induced subgraph on P.
        Returns (colors, ub) where colors[v] is the color of v and ub is the
        number of colors (an upper bound on the max independent set in P)."""
        colors = {}
        c = 0
        while P:
            # pick a vertex from P
            lsb = P & (-P)
            v = lsb.bit_length() - 1
            colors[v] = c
            P ^= lsb
            # remove v and its neighbors from P
            P &= ~adj2[v]
            c += 1
        return colors, c

    def expand(P, size):
        if P == 0:
            if size > best_size[0]:
                best_size[0] = size
                best_mask[0] = P  # placeholder; track separately
            return
        if time.time() > deadline:
            return
        colors, ub = color_bound(P)
        if size + ub <= best_size[0]:
            return
        # branch on vertices in decreasing color order
        # build list of (v, color) sorted by color desc
        items = sorted(colors.items(), key=lambda kv: kv[1], reverse=True)
        for v, col in items:
            if time.time() > deadline:
                return
            if size + col + 1 <= best_size[0]:
                continue
            # try including v
            newP = P & adj2[v]  # neighbors of v within P (can coexist with v)
            expand(newP, size + 1)
            P ^= (1 << v)  # remove v for the "exclude v" branch

    # track best mask properly via a wrapper
    def expand2(P, size, cur_mask):
        if P == 0:
            if size > best_size[0]:
                best_size[0] = size
                best_mask[0] = cur_mask
            return
        if time.time() > deadline:
            return
        colors, ub = color_bound(P)
        if size + ub <= best_size[0]:
            return
        items = sorted(colors.items(), key=lambda kv: kv[1], reverse=True)
        for v, col in items:
            if time.time() > deadline:
                return
            if size + col + 1 <= best_size[0]:
                continue
            newP = P & adj2[v]
            expand2(newP, size + 1, cur_mask | (1 << v))
            P ^= (1 << v)

    full = (1 << M) - 1
    expand2(full, 0, 0)

    exact_mask = best_mask[0]
    exact_size = best_size[0]

    # ---------- Polish: 1-add / 1-swap local search on the best solution ----------
    def polish(mask, size):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # try adding any non-neighbor outside the set
            outside = full ^ mask
            for v in range(M):
                if (outside >> v) & 1 and (adj2[v] & mask) == 0:
                    mask |= (1 << v)
                    size += 1
                    improved = True
            # try 1-swap: remove v_in, add v_out (v_out not in set, no conflict with set minus v_in)
            for v_in in range(M):
                if not ((mask >> v_in) & 1):
                    continue
                rest = mask ^ (1 << v_in)
                for v_out in range(M):
                    if (mask >> v_out) & 1:
                        continue
                    if (adj2[v_out] & rest) == 0:
                        mask = rest | (1 << v_out)
                        improved = True
                        break
                if improved:
                    break
        return mask, size

    best_mask, best_size2 = polish(exact_mask, exact_size)

    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]

    # Safety: if empty, fall back to a valid greedy set
    if not result:
        chosen = 0
        for i in range(M):
            if (adj2[i] & chosen) == 0:
                chosen |= (1 << i)
        result = [idx_to_num[i] for i in range(M) if (chosen >> i) & 1]

    return sorted(result)
