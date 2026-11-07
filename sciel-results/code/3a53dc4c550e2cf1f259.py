"""Cochromatic partition: exact minimum via precomputed tables + DP over maximal homogeneous sets."""
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

FULL = (1 << N) - 1

# Precompute is_clique and is_indep for all 2^N masks
is_clique = [False] * (1 << N)
is_indep = [False] * (1 << N)
is_clique[0] = True
is_indep[0] = True

for mask in range(1, 1 << N):
    lsb = mask & -mask
    v = lsb.bit_length() - 1
    rest = mask ^ lsb
    is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
    is_indep[mask] = is_indep[rest] and ((adj[v] & rest) == 0)

# Precompute is_homogeneous
is_hom = [False] * (1 << N)
for m in range(1 << N):
    is_hom[m] = is_clique[m] or is_indep[m]

def bron_kerbosch(adj_list, R, P_mask, X_mask, out):
    """Bron–Kerbosch with pivot to find all maximal cliques in a graph given by adj_list."""
    if P_mask == 0 and X_mask == 0:
        if R:
            out.append(R)
        return
    if P_mask | X_mask:
        lsb = (P_mask | X_mask) & -(P_mask | X_mask)
        u = lsb.bit_length() - 1
        candidates = P_mask & ~adj_list[u]
    else:
        candidates = P_mask
    while candidates:
        lsb_v = candidates & -candidates
        v = lsb_v.bit_length() - 1
        bron_kerbosch(adj_list, R | lsb_v, P_mask & adj_list[v], X_mask & adj_list[v], out)
        P_mask &= ~lsb_v
        X_mask |= lsb_v
        candidates &= ~lsb_v

# Find all maximal cliques
max_cliques = []
bron_kerbosch(adj, 0, FULL, 0, max_cliques)

# Find all maximal independent sets (cliques in complement)
comp_adj = [0] * N
for i in range(N):
    comp_adj[i] = FULL ^ (1 << i) ^ adj[i]

max_indep = []
bron_kerbosch(comp_adj, 0, FULL, 0, max_indep)

# Collect all maximal homogeneous sets (cliques or independent sets of size >= 1)
# Only maximal ones matter for optimal partition
max_homs = set(max_cliques)
max_homs.update(max_indep)
max_homs_list = list(max_homs)

# Also include single vertices (they are always maximal homogeneous in some sense)
# Actually single vertices are contained in larger homogeneous sets, but we need them
# for DP correctness. Let's include all homogeneous sets that are maximal
# (not a proper subset of another homogeneous set).
# For efficiency, let's just use the maximal cliques and maximal independent sets,
# plus single vertices.

# Actually, for the DP to work correctly, we need: any homogeneous set H can be
# extended to a maximal homogeneous set H' ⊇ H. So in an optimal partition, we can
# always replace H with H' (or a subset thereof). But H' might overlap with other parts.
# So we need to be careful. The standard approach: in a minimum partition, each part
# can be assumed to be a maximal clique or maximal independent set (within the remaining
# vertices). But this is tricky.

# Simpler approach: use DP over all homogeneous subsets. For n=14, that's 2^14 = 16384
# subsets, and for each we iterate over its submasks. Total work ~ 3^14 ≈ 4.8M, which is fine.

def solve_dp():
    """DP: dp[mask] = min number of homogeneous sets to cover exactly mask."""
    size = 1 << N
    INF = float('inf')
    dp = [INF] * size
    parent = [-1] * size
    dp[0] = 0

    for mask in range(1, size):
        sub = mask
        while sub:
            if is_hom[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask

    # Reconstruct
    parts = []
    mask = FULL
    while mask:
        s = parent[mask]
        if s == -1:
            return None
        part = [i for i in range(N) if s & (1 << i)]
        parts.append(part)
        mask ^= s
    return parts

def solve_bnb():
    """Branch and bound fallback."""
    best = [N]
    best_parts = [None]

    def backtrack(remaining, current_parts, depth):
        if remaining == 0:
            if depth < best[0]:
                best[0] = depth
                best_parts[0] = list(current_parts)
            return
        if depth + 1 >= best[0]:
            return
        verts = [i for i in range(N) if remaining & (1 << i)]
        first_v = verts[0]
        other_verts = verts[1:]
        k = len(other_verts)
        for size in range(k + 1):
            for combo in combinations(other_verts, size):
                m = (1 << first_v)
                for v in combo:
                    m |= (1 << v)
                if is_hom[m]:
                    new_remaining = remaining & ~m
                    current_parts.append(list(combo) + [first_v])
                    backtrack(new_remaining, current_parts, depth + 1)
                    current_parts.pop()

    backtrack(FULL, [], 0)
    return best_parts[0] if best_parts[0] is not None else [[i] for i in range(N)]

def construct():
    deadline = time.time() + 80.0

    try:
        # Try DP first (fast for n=14)
        result = solve_dp()
        if result is not None:
            # Validate
            seen = set()
            for part in result:
                for v in part:
                    if v in seen:
                        raise ValueError(f"Vertex {v} appears twice")
                    seen.add(v)
                m = 0
                for v in part:
                    m |= (1 << v)
                if not is_hom[m]:
                    raise ValueError(f"Part {part} is not homogeneous")
            if len(seen) != N:
                raise ValueError(f"Only {len(seen)} vertices covered")
            return [list(part) for part in result]
        else:
            raise ValueError("DP failed")
    except Exception:
        # Fallback to B&B
        try:
            result = solve_bnb()
            seen = set()
            for part in result:
                for v in part:
                    if v in seen:
                        raise ValueError(f"Vertex {v} appears twice")
                    seen.add(v)
                m = 0
                for v in part:
                    m |= (1 << v)
                if not is_hom[m]:
                    raise ValueError(f"Part {part} is not homogeneous")
            if len(seen) != N:
                raise ValueError(f"Only {len(seen)} vertices covered")
            return [list(part) for part in result]
        except Exception:
            # Final fallback: greedy
            es = {tuple(e) for e in P['edges']}
            parts = []
            for v in range(N):
                for part in parts:
                    trial = part + [v]
                    hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
                    if all(hits) or not any(hits):
                        part.append(v)
                        break
                else:
                    parts.append([v])
            return [list(part) for part in parts]

# EVOLVE-BLOCK-END
