"""Exact minimum cochromatic partition via subset DP over precomputed valid pieces.

Structural change: replace the current branch-and-bound vertex-assignment search
with a DP over all 2^14 vertex subsets, using a precomputed validity table that
marks each subset as a clique, an independent set, or neither.  This is the
design of the top-scoring inspirations, which achieve a perfect 1.0 on this
fixed 14-vertex instance.  The DP guarantees the true minimum number of parts
in well under a second, and reconstruction yields the exact partition.
"""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    deadline = time.time() + 75  # safety margin under 90s kill

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute: is_valid[mask] = True if mask induces a clique or independent set
    size = 1 << n
    is_valid = [False] * size
    is_valid[0] = True  # empty set (unused in DP transitions but safe)

    for mask in range(1, size):
        is_clique = True
        is_indep = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            rest = mask ^ lsb
            # For clique: every vertex in rest must be adjacent to u
            if rest & ~adj[u]:
                is_clique = False
            # For independent set: no vertex in rest may be adjacent to u
            if rest & adj[u]:
                is_indep = False
            if not is_clique and not is_indep:
                break
            m ^= lsb
        is_valid[mask] = is_clique or is_indep

    # DP: dp[mask] = minimum number of parts to partition the vertex set 'mask'
    INF = 10 ** 9
    dp = [INF] * size
    dp[0] = 0
    best_sub = [0] * size  # best_sub[mask] = the submask chosen as the last part

    for mask in range(1, size):
        if time.time() > deadline:
            break
        best_val = INF
        best_s = 0
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < best_val:
                    best_val = val
                    best_s = sub
            sub = (sub - 1) & mask
        dp[mask] = best_val
        best_sub[mask] = best_s

    # If DP didn't finish, fall back to greedy
    if dp[full_mask] == INF:
        remaining = full_mask
        parts = []
        while remaining:
            best = 0
            best_size = 0
            sub = remaining
            while sub:
                if is_valid[sub]:
                    bc = sub.bit_count()
                    if bc > best_size:
                        best_size = bc
                        best = sub
                sub = (sub - 1) & remaining
            if best == 0:
                lsb = remaining & -remaining
                best = lsb
            parts.append(best)
            remaining ^= best
        # Convert masks to lists
        result = []
        for pm in parts:
            verts = []
            m = pm
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            result.append(verts)
        return result

    # Reconstruct partition from DP
    parts = []
    cur = full_mask
    while cur:
        sub = best_sub[cur]
        verts = []
        m = sub
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        parts.append(verts)
        cur ^= sub

    # Ensure types: Python int vertices, list of lists
    out = [[int(v) for v in part] for part in parts]
    return out
# EVOLVE-BLOCK-END
