import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256
    # Precompute conflict: conflict[a][b] True if a+b is a perfect square (a<=b)
    # Include diagonal: a+a = 2a must not be square
    is_sq = [False] * (2 * N + 1)
    for k in range(1, int(math.isqrt(2 * N)) + 1):
        is_sq[k * k] = True

    # conflict[a] = set of b >= a such that a+b is square
    # We'll represent as list of lists for speed
    conflict = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            s = a + b
            if is_sq[s]:
                conflict[a].append(b)

    # Also for "can add" we need: for candidate c, check all existing members m
    # such that min(c,m)+max(c,m) is square. Since we store set, we check
    # for each m in S: is (min+max) square?
    # Faster: precompute for each c, the set of all x in [1,256] where c+x is square
    # (x can be < c or >= c)
    compat_conflict = [set() for _ in range(N + 1)]
    for c in range(1, N + 1):
        for x in range(1, N + 1):
            if is_sq[c + x]:
                compat_conflict[c].add(x)

    def is_valid(S):
        # S is a set of ints
        for a in S:
            for b in S:
                if a > b:
                    continue
                if is_sq[a + b]:
                    return False
        return True

    def can_add(S, c):
        # check if adding c to S keeps validity
        for m in S:
            if is_sq[c + m]:
                return False
        return True

    # Greedy min-degree independent set
    def greedy_min_degree():
        remaining = set(range(1, N + 1))
        chosen = set()
        # degree = number of conflicts within remaining
        # For each node, count how many of its conflict-neighbors are in remaining
        # conflict includes self if 2a is square
        # Build adjacency including self-loops for diagonal
        adj = [set() for _ in range(N + 1)]
        for a in range(1, N + 1):
            for b in conflict[a]:
                adj[a].add(b)
                adj[b].add(a)
            if is_sq[2 * a]:
                adj[a].add(a)  # self-loop: cannot pick a if 2a is square

        while remaining:
            # find node with min degree in remaining
            min_deg = float('inf')
            min_node = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
                    if deg == 0:
                        break
            chosen.add(min_node)
            remaining.discard(min_node)
            # remove neighbors of min_node from remaining
            for nb in adj[min_node]:
                remaining.discard(nb)
        return chosen

    # Local search: try to add any node, then try swaps
    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try adding
            for c in range(1, N + 1):
                if c in S:
                    continue
                if can_add(S, c):
                    S.add(c)
                    improved = True
            # Try swaps: remove one, add one (net gain 0) or remove two add three, etc.
            # Simple: for each pair (remove r, add c) where c not in S, r in S
            # Try all single swaps first
            for r in list(S):
                S_without_r = S - {r}
                for c in range(1, N + 1):
                    if c in S_without_r:
                        continue
                    if can_add(S_without_r, c):
                        # swap: remove r, add c -> same size, but maybe enables more adds
                        # Actually just do it if it allows subsequent adds
                        # For now, just try pure adds above; swaps for diversity
                        pass
        return S

    def local_search_v2(S):
        """Add + 1-for-1 swap local search."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            # Pure adds
            for c in range(1, N + 1):
                if c in S:
                    continue
                if can_add(S, c):
                    S.add(c)
                    changed = True
            # 1-for-1 swaps: remove r, add c
            # Only useful if it enables further adds; do a lookahead
            for r in list(S):
                for c in range(1, N + 1):
                    if c in S:
                        continue
                    S2 = S - {r}
                    if can_add(S2, c):
                        S2.add(c)
                        # Try to add more
                        for d in range(1, N + 1):
                            if d in S2:
                                continue
                            if can_add(S2, d):
                                S2.add(d)
                        if len(S2) > len(S):
                            S = S2
                            changed = True
                            break
                if changed:
                    break
        return S

    # Multi-restart with randomized greedy
    def randomized_greedy(seed):
        rng = random.Random(seed)
        remaining = set(range(1, N + 1))
        chosen = set()
        adj = [set() for _ in range(N + 1)]
        for a in range(1, N + 1):
            for b in conflict[a]:
                adj[a].add(b)
                adj[b].add(a)
            if is_sq[2 * a]:
                adj[a].add(a)

        while remaining:
            # Score each node: degree + random tiebreak
            candidates = []
            for v in remaining:
                deg = len(adj[v] & remaining)
                candidates.append((deg, rng.random(), v))
            candidates.sort()
            min_node = candidates[0][2]
            chosen.add(min_node)
            remaining.discard(min_node)
            for nb in adj[min_node]:
                remaining.discard(nb)
        return chosen

    # Start with deterministic greedy
    best = greedy_min_degree()
    best = local_search_v2(best)
    best_size = len(best)

    # Multi-restart with random seeds
    seed = 42
    while time.time() < deadline - 2.0:
        S = randomized_greedy(seed)
        S = local_search_v2(S)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        seed += 1

    # Final verification
    for a in best:
        for b in best:
            if a > b:
                continue
            if is_sq[a + b]:
                # Should not happen, but return something valid
                # Fallback to safe set
                safe = set()
                for x in range(1, N + 1):
                    if not is_sq[2 * x]:
                        # check against safe
                        ok = True
                        for y in safe:
                            if is_sq[x + y]:
                                ok = False
                                break
                        if ok:
                            safe.add(x)
                if safe:
                    return list(safe)
                return [1]

    return list(best)
# EVOLVE-BLOCK-END
