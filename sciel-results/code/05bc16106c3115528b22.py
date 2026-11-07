"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def build_conflict_graph(values, K):
        """
        Build a conflict graph where vertex i (representing all_pairs[i]) is adjacent
        to vertex j if edges i and j can coexist in a set of size m without
        exceeding K distinct sums AND K distinct products.
        
        We use a sufficient condition: for each pair (i,j), check if there exists
        a subset of size m-2 containing both i and j where the total distinct
        sums <= K and distinct products <= K.
        
        For efficiency, we use a greedy feasibility check.
        """
        # Precompute sum and product for each pair
        pair_sums = [values[u] + values[v] for u, v in all_pairs]
        pair_prods = [values[u] * values[v] for u, v in all_pairs]
        
        # For each pair of edges (i, j), determine if they are "compatible"
        # Two edges are compatible if we can extend {i, j} to a set of m edges
        # with at most K distinct sums and K distinct products.
        
        # Approximation: two edges are compatible if their sums and products
        # don't immediately force too many distinct values.
        # A stronger check: try greedy extension from {i, j}.
        
        def can_extend(start_set, values, pair_sums, pair_prods, target_size, K):
            """Greedy check: can we extend start_set to target_size with <= K sums and prods?"""
            sums = set()
            prods = set()
            used = set(start_set)
            for idx in start_set:
                sums.add(pair_sums[idx])
                prods.add(pair_prods[idx])
            
            if len(sums) > K or len(prods) > K:
                return False
            
            for _ in range(target_size - len(start_set)):
                best_idx = -1
                best_key = None
                for idx in range(num_pairs):
                    if idx in used:
                        continue
                    new_sums = len(sums) if pair_sums[idx] in sums else len(sums) + 1
                    new_prods = len(prods) if pair_prods[idx] in prods else len(prods) + 1
                    if new_sums > K or new_prods > K:
                        continue
                    key = (max(new_sums, new_prods), new_sums, new_prods, idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = idx
                if best_idx < 0:
                    return False
                used.add(best_idx)
                sums.add(pair_sums[best_idx])
                prods.add(pair_prods[best_idx])
            return True
        
        # For speed, we'll use a weaker compatibility check first,
        # then verify with greedy extension only for promising K values.
        
        # Weak check: edges i and j are compatible if:
        # - They don't share a vertex (they're independent) OR
        # - Even if they share a vertex, their sums/prods are "close"
        # Actually, for clique search, we need: can these two edges be part of
        # a valid set of m edges with <= K sums and <= K prods?
        
        # Efficient approach: for each pair (i,j), check if greedy extension works.
        # This is O(num_pairs^2 * m * num_pairs) which is too slow.
        # Instead, precompute for each edge i, the set of edges j that are "likely compatible".
        
        # Alternative: use a different approach. For a given K, build the graph
        # where edge i is connected to edge j if:
        # 1. They don't conflict (no shared vertices is not required - we just need
        #    a set of m edges, and edges can share vertices)
        # 2. Greedy extension from {i, j} to m edges stays within K.
        
        # For performance, let's use a heuristic: precompute for each edge i,
        # the "cost" of adding it (how many new sums/prods it introduces given
        # a baseline). Then two edges are compatible if their combined cost
        # doesn't exceed K when extended greedily.
        
        # Let's try a different formulation:
        # For each edge i, compute its "signature" (sum, prod).
        # Two edges i, j are compatible if the greedy algorithm starting from
        # {i, j} can reach m edges with <= K distinct sums and prods.
        
        # To make this tractable, we'll sample: for each K, try to find a clique
        # using a different method.
        
        return None  # Placeholder; we'll use a different approach below

    def greedy_edges_for_values(values):
        """Greedy edge selection: add edges that minimize new distinct sums, tie-break by products."""
        pair_info = []
        for idx, (u, v) in enumerate(all_pairs):
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_info.append((idx, u, v, s, p))

        sums = set()
        prods = set()
        selected = []
        selected_set = set()

        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx, u, v, s, p in pair_info:
                if idx in selected_set:
                    continue
                new_sums = len(sums) if s in sums else len(sums) + 1
                new_prods = len(prods) if p in prods else len(prods) + 1
                key = (max(new_sums, new_prods), new_sums, new_prods, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx < 0:
                break
            idx, u, v, s, p = pair_info[best_idx]
            selected.append((idx, u, v, s))
            selected_set.add(idx)
            sums.add(s)
            prods.add(p)

        edges = [(u, v) for _, u, v, _ in selected]
        return edges

    def clique_search_for_K(values, K, rng, max_time=5.0):
        """
        For a given K, try to find a set of m edges with at most K distinct sums
        and at most K distinct products using a branch-and-bound / backtracking approach.
        
        We model this as: select m edges from num_pairs such that the number of
        distinct sums <= K and distinct products <= K.
        
        Approach: use a greedy + local search hybrid.
        """
        start_t = time.time()
        
        pair_sums = [values[u] + values[v] for u, v in all_pairs]
        pair_prods = [values[u] * values[v] for u, v in all_pairs]
        
        # Group edges by their sum and product
        sum_groups = {}
        prod_groups = {}
        for idx in range(num_pairs):
            s = pair_sums[idx]
            p = pair_prods[idx]
            sum_groups.setdefault(s, []).append(idx)
            prod_groups.setdefault(p, []).append(idx)
        
        # Strategy: iteratively pick edges that reuse existing sums and products
        # Start with an empty set, greedily add edges that minimize new distinct values
        
        best_edges = None
        best_score = float('inf')
        
        # Multiple random restarts
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() - start_t > max_time:
                break
            
            # Random initialization: pick a few random edges
            initial_size = rng.randint(0, 5)
            used = set()
            sums = set()
            prods = set()
            
            for _ in range(initial_size):
                idx = rng.randint(0, num_pairs - 1)
                if idx not in used:
                    used.add(idx)
                    sums.add(pair_sums[idx])
                    prods.add(pair_prods[idx])
            
            # Greedy extension
            while len(used) < m:
                if len(sums) > K or len(prods) > K:
                    break
                best_idx = -1
                best_key = None
                for idx in range(num_pairs):
                    if idx in used:
                        continue
                    ns = len(sums) if pair_sums[idx] in sums else len(sums) + 1
                    np = len(prods) if pair_prods[idx] in prods else len(prods) + 1
                    if ns > K or np > K:
                        continue
                    key = (max(ns, np), ns, np, idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = idx
                if best_idx < 0:
                    break
                used.add(best_idx)
                sums.add(pair_sums[best_idx])
                prods.add(pair_prods[best_idx])
            
            if len(used) == m and len(sums) <= K and len(prods) <= K:
                edges = [list(all_pairs[idx]) for idx in used]
                score = max(len(sums), len(prods))
                if score < best_score:
                    best_score = score
                    best_edges = edges
        
        return best_edges, best_score

    def local_search_edges(values, edges, rng, max_iters=2000, max_time=5.0):
        """Local search: swap edges to reduce distinct sums/products."""
        start_t = time.time()
        edges = [list(e) for e in edges]
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        sums = {}
        prods = {}
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sums[s] = sums.get(s, 0) + 1
            prods[p] = prods.get(p, 0) + 1
        
        best_score = max(len(sums), len(prods))
        best_edges = [list(e) for e in edges]
        
        for it in range(max_iters):
            if time.time() - start_t > max_time:
                break
            
            # Pick a random edge to replace
            idx = rng.randint(0, m - 1)
            old_edge = tuple(sorted(edges[idx]))
            
            # Try random new edge
            new_idx = rng.randint(0, num_pairs - 1)
            new_edge = all_pairs[new_idx]
            new_edge_sorted = tuple(sorted(new_edge))
            
            if new_edge_sorted == old_edge or new_edge_sorted in edge_set:
                continue
            
            old_s = values[old_edge[0]] + values[old_edge[1]]
            old_p = values[old_edge[0]] * values[old_edge[1]]
            new_s = values[new_edge[0]] + values[new_edge[1]]
            new_p = values[new_edge[0]] * values[new_edge[1]]
            
            # Update counts
            sc = dict(sums)
            pc = dict(prods)
            sc[old_s] -= 1
            pc[old_p] -= 1
            if sc[old_s] == 0:
                del sc[old_s]
            if pc[old_p] == 0:
                del pc[old_p]
            sc[new_s] = sc.get(new_s, 0) + 1
            pc[new_p] = pc.get(new_p, 0) + 1
            
            old_score = max(len(sums), len(prods))
            new_score = max(len(sc), len(pc))
            delta = new_score - old_score
            
            if delta <= 0 or rng.random() < math.exp(-delta / max(0.1, 1.0 * (1.0 - it / max_iters))):
                edges[idx] = list(new_edge)
                edge_set.discard(old_edge)
                edge_set.add(new_edge_sorted)
                sums = sc
                prods = pc
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in edges]
        
        return best_edges, best_score

    def generate_values_strategy0(rng):
        """All multiples of 2 with small quotients."""
        candidates = list(range(1, 30))
        rng.shuffle(candidates)
        a_vals = candidates[:14]
        values = [2 * a for a in a_vals]
        return values

    def generate_values_strategy1(rng):
        """Values of form 2^a * 3^b for small a,b."""
        vals = set()
        for a in range(0, 6):
            for b in range(0, 5):
                v = (2 ** a) * (3 ** b)
                if 1 <= v <= max_val:
                    vals.add(v)
        vals = list(vals)
        rng.shuffle(vals)
        values = vals[:14]
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy2(rng):
        """Small arithmetic progression."""
        start = rng.randint(1, 10)
        step = rng.randint(2, 5)
        values = []
        v = start
        while len(values) < 14 and v <= max_val:
            values.append(v)
            v += step
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy3(rng):
        """Mix: some multiples of 2, some multiples of 3."""
        values = []
        for _ in range(7):
            v = 2 * rng.randint(1, 20)
            if v not in values and v <= max_val:
                values.append(v)
        for _ in range(7):
            v = 3 * rng.randint(1, 20)
            if v not in values and v <= max_val:
                values.append(v)
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy4(rng):
        """Powers of 2 and 3 mixed."""
        vals = set()
        for a in range(0, 10):
            vals.add(2 ** a)
        for b in range(0, 8):
            vals.add(3 ** b)
        vals = list(vals)
        rng.shuffle(vals)
        values = vals[:14]
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy5(rng):
        """Geometric-like progression with small ratios."""
        values = []
        base = rng.randint(1, 5)
        ratio = rng.randint(2, 3)
        v = base
        while len(values) < 14 and v <= max_val:
            values.append(v)
            v *= ratio
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy6(rng):
        """Concentrated values: many values close together."""
        center = rng.randint(5, 50)
        spread = rng.randint(3, 10)
        values = []
        for _ in range(14):
            v = center + rng.randint(-spread, spread)
            if v < 1:
                v = 1
            if v > max_val:
                v = max_val
            if v not in values:
                values.append(v)
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def optimize_values(values, edges, rng, max_iters=100, max_time=5.0):
        """Perturb values and re-optimize edges."""
        start_t = time.time()
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        best_score = evaluate(best_values, best_edges)

        for _ in range(max_iters):
            if time.time() - start_t > max_time:
                break

            i = rng.randint(0, n - 1)
            old_val = best_values[i]
            delta = rng.randint(-15, 15)
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val or new_val == old_val:
                continue
            if new_val in best_values:
                continue

            new_values = list(best_values)
            new_values[i] = new_val
            new_edges = greedy_edges_for_values(new_values)
            new_score = evaluate(new_values, new_edges)

            if new_score < best_score:
                best_values = new_values
                best_edges = new_edges
                best_score = new_score

        return best_values, best_edges, best_score

    # Main search
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')

    strategies = [
        generate_values_strategy0,
        generate_values_strategy1,
        generate_values_strategy2,
        generate_values_strategy3,
        generate_values_strategy4,
        generate_values_strategy5,
        generate_values_strategy6,
    ]

    # Phase 1: Try each strategy with greedy + local search + clique search
    for strat_func in strategies:
        if time.time() > deadline:
            break

        for restart in range(6):
            if time.time() > deadline:
                break

            values = strat_func(rng)
            values = list(set(values))
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            values = values[:14]

            # Greedy first
            edges = greedy_edges_for_values(values)
            score = evaluate(values, edges)

            if score < best_score:
                best_score = score
                best_result = (list(values), [list(e) for e in edges])

            # Local search refinement
            edges_ls, score_ls = local_search_edges(values, edges, rng, max_iters=1500, max_time=3.0)
            if score_ls < best_score:
                best_score = score_ls
                best_result = (list(values), edges_ls)

            # Clique search for K = best_score
            if best_score <= 15:
                edges_clique, score_clique = clique_search_for_K(values, best_score, rng, max_time=3.0)
                if edges_clique is not None and score_clique < best_score:
                    best_score = score_clique
                    best_result = (list(values), edges_clique)

            if best_score <= 10:
                break

        if best_score <= 10:
            break

    # Phase 2: Value optimization on best result
    if best_result is not None:
        values, edges = best_result

        for _ in range(3):
            if time.time() > deadline:
                break
            values, edges, score = optimize_values(values, edges, rng, max_iters=100, max_time=4.0)
            if score < best_score:
                best_score = score
                best_result = (list(values), edges)

            # Local search on edges again
            edges_ls, score_ls = local_search_edges(values, edges, rng, max_iters=1500, max_time=3.0)
            if score_ls < best_score:
                best_score = score_ls
                best_result = (list(values), edges_ls)

    # Phase 3: More random restarts with clique search
    for restart in range(30):
        if time.time() > deadline:
            break

        strat_func = strategies[rng.randint(0, len(strategies) - 1)]
        values = strat_func(rng)
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]

        edges = greedy_edges_for_values(values)
        score = evaluate(values, edges)

        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])

        # Local search
        edges_ls, score_ls = local_search_edges(values, edges, rng, max_iters=1000, max_time=2.0)
        if score_ls < best_score:
            best_score = score_ls
            best_result = (list(values), edges_ls)

        # Clique search
        if best_score <= 15:
            edges_clique, score_clique = clique_search_for_K(values, best_score, rng, max_time=2.0)
            if edges_clique is not None and score_clique < best_score:
                best_score = score_clique
                best_result = (list(values), edges_clique)

        if best_score <= 8:
            break

    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_pairs[:m]]
    else:
        values, edges = best_result

    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)

    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
