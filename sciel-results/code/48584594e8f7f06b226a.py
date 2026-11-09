import time
import math
import sys
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit (evaluator kills at 90s)

    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set

    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute bad triples and edge weights
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    edge_weight = [[0.0] * (N + 1) for _ in range(N + 1)]

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    edge_weight[a][b] += 1.0
                    edge_weight[b][a] += 1.0
                    edge_weight[a][c] += 1.0
                    edge_weight[c][a] += 1.0
                    edge_weight[b][c] += 1.0
                    edge_weight[c][b] += 1.0

    # Build adjacency for conflict graph
    adj = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if edge_weight[a][b] > 0:
                adj[a].append((b, edge_weight[a][b]))
                adj[b].append((a, edge_weight[b][a]))

    # Compute degree-weighted conflict score
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b, w in adj[a]:
            conflict_score[a] += w

    # Identify elements with zero conflicts - these can always be included
    zero_conflict = set()
    for a in range(1, N + 1):
        if conflict_score[a] == 0:
            zero_conflict.add(a)

    # Fast upper bound: greedy independent set gives lower bound, so we use
    # a simple bound based on the number of vertices with zero edges to available
    def fast_upper_bound(available):
        """Fast upper bound on independent set size."""
        avail_count = available.bit_count()
        # Count vertices that have no conflicts with any other available vertex
        # These can all be included
        count = 0
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            temp ^= low
            # Check if x has no conflicts with other available vertices
            has_conflict = False
            for y, w in adj[x]:
                if (available >> y) & 1:
                    has_conflict = True
                    break
            if not has_conflict:
                count += 1
        # Upper bound: at most avail_count, but at least count
        # Use a simple bound: avail_count - floor(avail_count/3) as heuristic
        # Actually, use: for each connected component, alpha <= n_component
        # Simple bound: avail_count (trivial)
        return avail_count

    # Weighted greedy with fast conflict checking
    def weighted_greedy(order, base_set=None):
        S = list(base_set) if base_set else []
        S_mask = 0
        for s in S:
            S_mask |= (1 << s)
        for x in order:
            if not (full_mask >> x) & 1:
                continue
            # Check if x is compatible with S using bitmask
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S

    # Try several greedy orderings
    best_set = []
    best_size = 0

    # Always include zero-conflict elements first
    base_set = list(zero_conflict)
    base_mask = 0
    for x in base_set:
        base_mask |= (1 << x)

    remaining = [x for x in range(1, N + 1) if x not in zero_conflict]

    # Helper to try a greedy order
    def try_greedy(order):
        nonlocal best_set, best_size
        S = base_set + weighted_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Natural order
    try_greedy(remaining)

    # Reverse order
    try_greedy(list(reversed(remaining)))

    # Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(remaining, key=lambda x: conflict_score[x])
    try_greedy(order)

    # Sort by conflict score (descending)
    order = sorted(remaining, key=lambda x: -conflict_score[x])
    try_greedy(order)

    # Sort by number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    order = sorted(remaining, key=num_prime_factors)
    try_greedy(order)

    # Sort by value (ascending)
    try_greedy(sorted(remaining))

    # Sort by value (descending)
    try_greedy(sorted(remaining, reverse=True))

    # Multiple random orderings
    for seed in range(30):
        if time.time() - start_time > deadline - 15:
            break
        rng = random.Random(seed)
        order = remaining[:]
        rng.shuffle(order)
        try_greedy(order)

    # Spectral ordering: use power iteration on (D - W) to get ranking
    random.seed(42)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = random.random()

    norm = math.sqrt(sum(v[i]**2 for i in range(1, N + 1)))
    if norm > 0:
        for i in range(1, N + 1):
            v[i] /= norm

    for _ in range(200):
        if time.time() - start_time > deadline - 10:
            break
        new_v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            new_v[i] = conflict_score[i] * v[i]
            for j, w in adj[i]:
                new_v[i] -= w * v[j]

        norm = math.sqrt(sum(new_v[i]**2 for i in range(1, N + 1)))
        if norm > 1e-10:
            for i in range(1, N + 1):
                new_v[i] /= norm
        v = new_v

    spectral_order = sorted(remaining, key=lambda x: -v[x])
    try_greedy(spectral_order)

    spectral_order_asc = sorted(remaining, key=lambda x: v[x])
    try_greedy(spectral_order_asc)

    # Branch and bound with improved upper bound
    def solve(selected, available, depth=0):
        nonlocal best_size, best_set, start_time

        if time.time() - start_time > deadline:
            return

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

        if not available:
            return

        # Upper bound: trivial
        avail_count = available.bit_count()
        if len(selected) + avail_count <= best_size:
            return

        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low

        if not avail_list:
            return

        # Pick element with highest conflict degree in available
        best_x = avail_list[0]
        best_conflict = -1
        for x in avail_list:
            c = 0.0
            for y, w in adj[x]:
                if (available >> y) & 1:
                    c += w
            if c > best_conflict:
                best_conflict = c
                best_x = x
        x = best_x

        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        for s in selected:
            new_avail &= ~conflict[s][x]

        solve(new_selected, new_avail, depth + 1)

        if time.time() - start_time > deadline:
            return

        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x), depth + 1)

    solve([], full_mask)

    # Local search polish: try to swap elements to increase size
    def local_search_polish(S, max_iterations=100):
        """Try to improve the set by swapping elements."""
        S_set = set(S)
        S_mask = 0
        for s in S_set:
            S_mask |= (1 << s)

        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() - start_time < deadline - 1:
            improved = False
            iterations += 1

            # Try adding an element
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                # Check if x is compatible
                compatible = True
                for s in S_set:
                    if conflict[s][x] & S_mask:
                        compatible = False
                        break
                if compatible:
                    S_set.add(x)
                    S_mask |= (1 << x)
                    improved = True
                    break

            # Try swapping: remove one element, add another
            if not improved:
                for remove in list(S_set):
                    temp_S_set = S_set - {remove}
                    temp_S_mask = S_mask & ~(1 << remove)

                    for add in range(1, N + 1):
                        if add in S_set:
                            continue
                        # Check if add is compatible with temp_S_set
                        compatible = True
                        for s in temp_S_set:
                            if conflict[s][add] & temp_S_mask:
                                compatible = False
                                break
                        if compatible:
                            S_set.remove(remove)
                            S_set.add(add)
                            S_mask = temp_S_mask | (1 << add)
                            improved = True
                            break
                    if improved:
                        break

        return list(S_set)

    # Apply local search polish
    if best_set:
        polished = local_search_polish(best_set)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished

    return best_set
