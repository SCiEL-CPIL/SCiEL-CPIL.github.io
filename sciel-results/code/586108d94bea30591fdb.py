"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    n = 14
    m = 35
    max_val = 10000

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edges_for_values(values, rng):
        """Greedy edge selection: add edges that minimize new distinct sums, tie-break by products."""
        all_pairs = list(combinations(range(n), 2))
        rng.shuffle(all_pairs)
        
        selected = []
        sums = set()
        prods = set()
        
        # Precompute sum and product for all pairs
        pair_info = {}
        for u, v in all_pairs:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_info[(u, v)] = (s, p)
        
        for u, v in all_pairs:
            if len(selected) >= m:
                break
            s, p = pair_info[(u, v)]
            # Prefer edges where sum is already seen, then product already seen
            sum_known = s in sums
            prod_known = p in prods
            # Score: lower is better
            # We want to minimize distinct sums primarily, then products
            # Use a heuristic: if sum is new, it's costly
            cost = (0 if sum_known else 1, 0 if prod_known else 1)
            selected.append((cost, u, v))
        
        # Sort by cost and take top m
        selected.sort(key=lambda x: x[0])
        edges = [(u, v) for _, u, v in selected[:m]]
        
        # Recompute actual sets
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        
        return edges, len(sums), len(prods)

    def local_search_edges(values, edges, rng, max_iters=500):
        """Local search: try swapping one edge for another to reduce distinct sums/products."""
        edges_set = set(tuple(e) for e in edges)
        current_edges = list(edges)
        sums = set()
        prods = set()
        for u, v in current_edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        best_score = max(len(sums), len(prods))
        
        all_pairs = list(combinations(range(n), 2))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing a random edge and adding a different one
            idx = rng.randint(0, m - 1)
            old_edge = tuple(current_edges[idx])
            
            # Try replacing with a random non-edge
            for _ in range(50):
                new_u, new_v = rng.choice(all_pairs)
                if (new_u, new_v) in edges_set:
                    continue
                if (new_v, new_u) in edges_set:
                    continue
                
                # Compute new sums and prods
                new_sums = set(sums)
                new_prods = set(prods)
                
                # Remove old edge's contribution
                os = values[old_edge[0]] + values[old_edge[1]]
                op = values[old_edge[0]] * values[old_edge[1]]
                # Need to count occurrences
                sum_counts = {}
                prod_counts = {}
                for u, v in current_edges:
                    s = values[u] + values[v]
                    p = values[u] * values[v]
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1
                
                sum_counts[os] -= 1
                prod_counts[op] -= 1
                if sum_counts[os] == 0:
                    new_sums.discard(os)
                if prod_counts[op] == 0:
                    new_prods.discard(op)
                
                # Add new edge's contribution
                ns = values[new_u] + values[new_v]
                np_ = values[new_u] * values[new_v]
                if ns not in new_sums:
                    new_sums.add(ns)
                if np_ not in new_prods:
                    new_prods.add(np_)
                
                new_score = max(len(new_sums), len(new_prods))
                if new_score < best_score:
                    # Accept
                    current_edges[idx] = [new_u, new_v]
                    edges_set.discard(old_edge)
                    edges_set.discard(tuple(reversed(old_edge)))
                    edges_set.add((new_u, new_v))
                    sums = new_sums
                    prods = new_prods
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                continue
        
        return current_edges, best_score

    def generate_structured_values(rng):
        """Generate values with shared structure to reduce product diversity."""
        strategy = rng.randint(0, 4)
        
        if strategy == 0:
            # Powers of 2 with small multipliers
            base = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            values = base[:13] + [rng.randint(1, 100)]
            values = sorted(set(values))
            while len(values) < 14:
                values.append(rng.randint(1, max_val))
                values = sorted(set(values))
            return values[:14]
        
        elif strategy == 1:
            # Multiples of a small number
            base_mult = rng.choice([2, 3, 4, 5, 6])
            values = []
            for i in range(1, 20):
                v = base_mult * i
                if v <= max_val and v not in values:
                    values.append(v)
                if len(values) == 14:
                    break
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 2:
            # Values of form 2^a * 3^b
            values = set()
            for a in range(0, 8):
                for b in range(0, 4):
                    v = (2 ** a) * (3 ** b)
                    if 1 <= v <= max_val:
                        values.add(v)
            values = list(values)
            rng.shuffle(values)
            values = values[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]
        
        elif strategy == 3:
            # Small consecutive multiples
            start = rng.randint(1, 5)
            step = rng.randint(1, 3)
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
        
        else:
            # Random but biased toward small numbers
            values = []
            for _ in range(14):
                v = rng.randint(1, 100)
                if v not in values:
                    values.append(v)
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

    def optimize_values(values, edges, rng, max_iters=200):
        """Try swapping pairs of values to improve score."""
        best_values = list(values)
        best_score = evaluate(best_values, edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            i, j = rng.sample(range(n), 2)
            new_values = list(best_values)
            new_values[i], new_values[j] = new_values[j], new_values[i]
            
            # Check constraints
            if len(set(new_values)) != 14:
                continue
            if any(v < 1 or v > max_val for v in new_values):
                continue
            
            new_score = evaluate(new_values, edges)
            if new_score < best_score:
                best_values = new_values
                best_score = new_score
        
        return best_values, best_score

    # Main search loop
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')
    
    # Initial structured values
    for restart in range(200):
        if time.time() > deadline:
            break
        
        values = generate_structured_values(rng)
        
        # Ensure all distinct and positive
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]
        
        # Greedy edge selection
        edges, ns, np_ = greedy_edges_for_values(values, rng)
        score = max(ns, np_)
        
        # Local search on edges
        edges, score = local_search_edges(values, edges, rng, max_iters=300)
        
        # Local search on values
        values, score = optimize_values(values, edges, rng, max_iters=100)
        
        # Re-optimize edges after value changes
        edges, ns, np_ = greedy_edges_for_values(values, rng)
        edges, score = local_search_edges(values, edges, rng, max_iters=200)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
            # Early exit if we hit a very good score
            if best_score <= 12:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
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
