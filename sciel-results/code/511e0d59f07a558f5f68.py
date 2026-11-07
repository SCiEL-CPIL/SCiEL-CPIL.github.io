"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        """Fast evaluation using numpy."""
        vals = np.array(values, dtype=np.int64)
        u = np.array([e[0] for e in edges], dtype=np.int64)
        v = np.array([e[1] for e in edges], dtype=np.int64)
        sums = vals[u] + vals[v]
        prods = vals[u] * vals[v]
        n_sums = len(np.unique(sums))
        n_prods = len(np.unique(prods))
        return max(n_sums, n_prods)
    
    def count_distinct(values, edges):
        """Return (n_sums, n_prods) for given values and edges."""
        vals = np.array(values, dtype=np.int64)
        u = np.array([e[0] for e in edges], dtype=np.int64)
        v = np.array([e[1] for e in edges], dtype=np.int64)
        sums = vals[u] + vals[v]
        prods = vals[u] * vals[v]
        return len(np.unique(sums)), len(np.unique(prods))
    
    def greedy_edges_for_sums(values, all_edges, rng, m_edges):
        """Greedy edge selection minimizing distinct sums only."""
        sums_seen = set()
        edges = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m_edges):
            best_edge = None
            best_score = float('inf')
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                score = 0 if s in sums_seen else 1
                if score < best_score:
                    best_score = score
                    best_edge = e
                    if best_score == 0:
                        break
            if best_edge is None:
                best_edge = remaining[0]
            edges.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
        
        return edges
    
    def greedy_edges_for_both(values, all_edges, rng, m_edges):
        """Greedy edge selection minimizing both distinct sums and products."""
        sums_seen = set()
        prods_seen = set()
        edges = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m_edges):
            best_edge = None
            best_score = float('inf')
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = (0 if s in sums_seen else 1) + (0 if p in prods_seen else 1)
                if score < best_score:
                    best_score = score
                    best_edge = e
                    if best_score == 0:
                        break
            if best_edge is None:
                best_edge = remaining[0]
            edges.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return edges
    
    def edge_local_search(values, edges, all_edges, rng, max_iter=20000):
        """Local search on edges only, minimizing max(n_sums, n_prods)."""
        best_edges = [list(e) for e in edges]
        n_sums, n_prods = count_distinct(values, best_edges)
        best_score = max(n_sums, n_prods)
        edge_set = set(tuple(sorted(e)) for e in best_edges)
        
        for _ in range(max_iter):
            if time.time() > deadline - 8:
                break
            
            # Pick a random edge to swap out
            idx = rng.randrange(len(best_edges))
            old_edge = tuple(sorted(best_edges[idx]))
            
            # Find candidate edges not in current set
            candidates = [e for e in all_edges if e not in edge_set]
            if not candidates:
                continue
            
            # Try several random candidates
            for _ in range(5):
                new_edge = rng.choice(candidates)
                new_edges = [list(e) for e in best_edges]
                new_edges[idx] = list(new_edge)
                ns, np_ = count_distinct(values, new_edges)
                new_score = max(ns, np_)
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    edge_set = set(tuple(sorted(e)) for e in best_edges)
                    break
        
        return best_edges, best_score
    
    def value_local_search(values, edges, rng, max_iter=10000):
        """Local search on values, keeping edges fixed."""
        best_values = list(values)
        n_sums, n_prods = count_distinct(best_values, edges)
        best_score = max(n_sums, n_prods)
        
        for _ in range(max_iter):
            if time.time() > deadline - 8:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6:
                # Perturb one value
                i = rng.randrange(n)
                delta = rng.randint(-10, 10)
                if delta == 0:
                    continue
                new_val = best_values[i] + delta
                if new_val < 1 or new_val > max_val:
                    continue
                new_values = list(best_values)
                new_values[i] = new_val
                if len(set(new_values)) != n:
                    continue
                ns, np_ = count_distinct(new_values, edges)
                new_score = max(ns, np_)
                if new_score < best_score:
                    best_score = new_score
                    best_values = new_values
                    
            else:
                # Swap two values
                i, j = rng.sample(range(n), 2)
                new_values = list(best_values)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                ns, np_ = count_distinct(new_values, edges)
                new_score = max(ns, np_)
                if new_score < best_score:
                    best_score = new_score
                    best_values = new_values
        
        return best_values, best_score
    
    def generate_structured_values(rng):
        """Generate value sets with strong multiplicative structure."""
        value_sets = []
        
        # Strategy A: Powers of small bases (minimizes product diversity)
        for base in [2, 3]:
            for offset in range(0, 5):
                vals = [base ** (i + offset) for i in range(n)]
                if max(vals) <= max_val and min(vals) >= 1 and len(set(vals)) == n:
                    value_sets.append(list(vals))
        
        # Strategy B: All multiples of a common factor
        for base in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
            for _ in range(3):
                # Use small multipliers to keep products clustered
                max_mult = max_val // base
                if max_mult < n:
                    continue
                multipliers = rng.sample(range(1, min(max_mult, 200) + 1), n)
                vals = [base * m for m in multipliers]
                if len(set(vals)) == n and max(vals) <= max_val:
                    value_sets.append(list(vals))
        
        # Strategy C: Values that are products of two small numbers
        # v_i = a_i * b_i where a_i and b_i are small
        for _ in range(50):
            a_vals = rng.sample(range(1, 50), n)
            b_vals = rng.sample(range(1, 50), n)
            vals = [a * b for a, b in zip(a_vals, b_vals)]
            if len(set(vals)) == n and max(vals) <= max_val and min(vals) >= 1:
                value_sets.append(list(vals))
        
        # Strategy D: Geometric progressions with small ratio
        for start in range(1, 20):
            for ratio in [2, 3]:
                vals = [start * (ratio ** i) for i in range(n)]
                if max(vals) <= max_val and min(vals) >= 1 and len(set(vals)) == n:
                    value_sets.append(list(vals))
        
        # Strategy E: Small values with common structure
        for _ in range(100):
            # Pick a base and add small variations
            base = rng.randint(1, 50)
            vals = []
            for i in range(n):
                v = base + rng.randint(0, 30)
                if v in vals:
                    v = base + rng.randint(31, 60)
                vals.append(v)
            if len(set(vals)) == n and max(vals) <= max_val:
                value_sets.append(list(vals))
        
        # Strategy F: Consecutive small integers
        for start in range(1, 50):
            vals = list(range(start, start + n))
            if max(vals) <= max_val:
                value_sets.append(list(vals))
        
        # Strategy G: Arithmetic progressions
        for start in range(1, 100):
            for step in range(1, 10):
                vals = [start + i * step for i in range(n)]
                if max(vals) <= max_val and len(set(vals)) == n:
                    value_sets.append(list(vals))
        
        return value_sets
    
    rng = random.Random(42)
    all_edges = list(combinations(range(n), 2))
    
    best_overall = float('inf')
    best_result = None
    
    # Generate structured value sets
    value_sets = generate_structured_values(rng)
    
    # Deduplicate
    seen_sets = set()
    unique_value_sets = []
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_value_sets.append(vs)
    
    # Score each value set with greedy edge selection
    scored_sets = []
    for vs in unique_value_sets:
        if time.time() > deadline - 20:
            break
        # Try both greedy strategies
        edges1 = greedy_edges_for_sums(vs, all_edges, rng, m)
        score1 = evaluate(vs, edges1)
        edges2 = greedy_edges_for_both(vs, all_edges, rng, m)
        score2 = evaluate(vs, edges2)
        
        if score1 <= score2:
            scored_sets.append((score1, vs, edges1))
        else:
            scored_sets.append((score2, vs, edges2))
    
    scored_sets.sort(key=lambda x: x[0])
    
    # Process top candidates with local search
    for score, vs, edges in scored_sets:
        if time.time() > deadline - 10:
            break
        if score >= best_overall:
            continue
        
        # First optimize edges
        opt_edges, edge_score = edge_local_search(vs, edges, all_edges, rng, max_iter=15000)
        
        # Then optimize values
        opt_vals, val_score = value_local_search(vs, opt_edges, rng, max_iter=8000)
        
        # Final edge optimization with new values
        final_edges, final_score = edge_local_search(opt_vals, opt_edges, all_edges, rng, max_iter=10000)
        
        if final_score < best_overall:
            best_overall = final_score
            best_result = (list(opt_vals), [list(e) for e in final_edges])
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_edges[:m]]
    else:
        vals, edges = best_result
    
    # Final validation
    assert len(vals) == n
    assert len(set(vals)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in vals)
    assert len(edges) == m
    assert all(isinstance(u, int) and isinstance(v, int) and 0 <= u < n and 0 <= v < n and u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': vals, 'edges': edges}
# EVOLVE-BLOCK-END
