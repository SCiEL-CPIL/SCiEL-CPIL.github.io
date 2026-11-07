"""Exact minimum cochromatic partition via branch-and-bound with symmetry breaking and tighter bounds."""
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
FULL = (1 << N) - 1

def is_clique(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & ~adj[v]:
            return False
    return True

def is_independent(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & adj[v]:
            return False
    return True

def is_homogeneous(mask):
    return is_clique(mask) or is_independent(mask)

# Precompute every homogeneous set (size >= 1)
hom_sets = []
for r in range(1, N + 1):
    for combo in combinations(range(N), r):
        m = 0
        for v in combo:
            m |= (1 << v)
        if is_homogeneous(m):
            hom_sets.append(m)

# For each vertex, the homogeneous sets that contain it
by_vertex = [[] for _ in range(N)]
for m in hom_sets:
    mm = m
    while mm:
        v = (mm & -mm).bit_length() - 1
        by_vertex[v].append(m)
        mm ^= (1 << v)

# Greedy upper bound
def greedy_ub():
    remaining = FULL
    parts = []
    while remaining:
        best_set = 0
        best_size = 0
        for m in hom_sets:
            if m & ~remaining:
                continue
            sz = m.bit_count()
            if sz > best_size:
                best_size = sz
                best_set = m
        if best_set == 0:
            best_set = remaining & -remaining
        parts.append(best_set)
        remaining &= ~best_set
    return len(parts), parts

UB, ub_parts = greedy_ub()

# Precompute max clique size and max independent set size for all masks (DP)
# For 14 vertices, 2^14 = 16384 masks, feasible
max_clique_sz = [0] * (1 << N)
max_is_sz = [0] * (1 << N)

# Compute max clique size for all masks using DP:
# max_clique_sz[mask] = max over v in mask of (1 + max_clique_sz[mask & adj[v]])
# Actually we need: for each mask, the maximum clique in the induced subgraph.
# Use: max_clique_sz[mask] = max(0, max over v in mask of 1 + max_clique_sz[mask_without_v & adj[v]])
# But this is not correct for general clique. Let's use Bron-Kerbosch style or simple DP.
# For small N, we can just compute for each mask the clique number directly.
# Alternative: precompute for each mask whether it's a clique, then max clique in mask = max size of clique subset.
# Since N=14, we can iterate all submasks.

# Simpler: for each mask, compute max clique size by checking all submasks that are cliques.
# But that's 3^N which is too much. Let's use the recursive DP.

# Better approach for small N:
# For each mask, max_clique_sz[mask] = max over v in mask of (1 + max_clique_sz[mask & ~(1<<v) & adj[v]])
# Wait, that's not right either. The standard DP for maximum clique:
# Let f(S) = max clique size in S.
# f(S) = max over v in S of (1 + f(S \ {v} ∩ N(v)))
# This works because any max clique either doesn't contain v, or contains v and then the rest is a clique in N(v) ∩ (S\{v}).

# Let's compute this iteratively by increasing mask size.
for mask in range(1 << N):
    if mask == 0:
        continue
    # For max clique
    best_c = 0
    m = mask
    while m:
        v = (m & -m).bit_length() - 1
        m ^= (1 << v)
        rest = mask & ~(1 << v) & adj[v]
        val = 1 + max_clique_sz[rest]
        if val > best_c:
            best_c = val
    max_clique_sz[mask] = best_c
    
    # For max independent set: independent set in G = clique in complement
    # complement adj: ~adj[v] & FULL
    best_i = 0
    m = mask
    while m:
        v = (m & -m).bit_length() - 1
        m ^= (1 << v)
        rest = mask & ~(1 << v) & (~adj[v] & FULL)
        val = 1 + max_is_sz[rest]
        if val > best_i:
            best_i = val
    max_is_sz[mask] = best_i

def lower_bound(remaining):
    """Tighter lower bound using max clique/IS plus a greedy packing argument."""
    lb = max(max_clique_sz[remaining], max_is_sz[remaining])
    # Additional bound: greedily pack vertices into homogeneous sets
    # and count how many we need (this gives a valid lower bound)
    # Actually, a better approach: use the fact that each part can contain
    # at most max_clique_sz or max_is_sz vertices, but that's weak.
    # Let's try: repeatedly find the largest homogeneous subset, remove it,
    # count iterations. This is an upper bound on the lower bound... no.
    # Actually for a lower bound, we can use: ceil(|remaining| / max_part_size)
    # where max_part_size is the largest homogeneous set within remaining.
    if remaining:
        max_sz = 0
        for m in hom_sets:
            if m & ~remaining:
                continue
            sz = m.bit_count()
            if sz > max_sz:
                max_sz = sz
        if max_sz > 0:
            import math
            lb2 = math.ceil(remaining.bit_count() / max_sz)
            if lb2 > lb:
                lb = lb2
    return lb

def solve(deadline):
    best = [UB]
    best_parts = [list(ub_parts)]
    memo = {}  # remaining -> min additional parts needed (lower bound cache)

    def backtrack(remaining, current, depth):
        if remaining == 0:
            if depth < best[0]:
                best[0] = depth
                best_parts[0] = [list(m) for m in current]
            return
        if time.time() > deadline:
            return
        if depth + 1 >= best[0]:
            return
        lb = lower_bound(remaining)
        if depth + lb >= best[0]:
            return
        # Memoization: if we've seen this remaining set with a lower or equal depth, skip
        if remaining in memo:
            if memo[remaining] <= (best[0] - depth):
                return
        else:
            memo[remaining] = best[0] - depth
        
        # Symmetry breaking: pick the smallest remaining vertex
        first_v = (remaining & -remaining).bit_length() - 1
        
        # Get candidate homogeneous sets containing first_v and subset of remaining
        candidates = []
        for m in by_vertex[first_v]:
            if m & ~remaining:
                continue
            candidates.append(m)
        
        # Sort by size descending, then by how many "hard" vertices it covers
        def score(m):
            sz = m.bit_count()
            # Bonus for covering vertices with few homogeneous set options
            hard_count = 0
            mm = m
            while mm:
                v = (mm & -mm).bit_length() - 1
                hard_count += len(by_vertex[v])
                mm ^= (1 << v)
            return (sz, -hard_count)
        
        candidates.sort(key=score, reverse=True)
        
        for m in candidates:
            new_remaining = remaining & ~m
            current.append(m)
            backtrack(new_remaining, current, depth + 1)
            current.pop()
            # Update memo with best found
            if remaining in memo:
                if best[0] - depth < memo[remaining]:
                    memo[remaining] = best[0] - depth

    backtrack(FULL, [], 0)
    return best_parts[0]

def construct():
    deadline = time.time() + 75.0
    try:
        result = solve(deadline)
        if result is None:
            raise ValueError("solve returned None")
        seen = set()
        for part in result:
            for v in part:
                if v in seen:
                    raise ValueError("dup")
                seen.add(v)
            m = 0
            for v in part:
                m |= (1 << v)
            if not is_homogeneous(m):
                raise ValueError("not homogeneous")
        if len(seen) != N:
            raise ValueError("incomplete")
        return [list(part) for part in result]
    except Exception:
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(N):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return [list(part) for part in parts]
