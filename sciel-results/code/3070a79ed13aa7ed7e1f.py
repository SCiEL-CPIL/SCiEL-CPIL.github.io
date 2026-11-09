"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _try_improve_edges(values, m, n, max_iter=5000):
    """Try to find a better edge set for fixed values via local search."""
    all_edges = list(combinations(range(n), 2))
    # Start with a random subset of size m
    edges = random.sample(all_edges, m)
    edges_set = set(tuple(sorted(e)) for e in edges)
    current_score = _score(values, edges)
    
    best_score = current_score
    best_edges = [list(e) for e in edges]
    
    # Local search: swap one edge for another
    for _ in range(max_iter):
        if _score(values, best_edges) < best_score:
            # This shouldn't happen but just in case
            pass
            
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges) - 1)
        remove_e = tuple(sorted(edges[idx]))
        
        # Pick a random edge to add (not already present)
        # To speed up, we can just pick random pairs and check
        found = False
        for _ in range(20):  # limited attempts to find a valid add edge
            new_e = tuple(sorted(random.sample(range(n), 2)))
            if new_e not in edges_set:
                # Try the swap
                new_edges = [e for i, e in enumerate(edges) if i != idx]
                new_edges.append(list(new_e))
                new_score = _score(values, new_edges)
                if new_score <= current_score:
                    current_score = new_score
                    edges = new_edges
                    edges_set = set(tuple(sorted(e)) for e in edges)
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [list(e) for e in edges]
                    found = True
                    break
        
        if not found:
            # If we couldn't find a swap, maybe restart or just continue
            # For simplicity, just continue the loop
            pass
            
    return best_score, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Strategy 1: Powers of 2
    # Sums of distinct powers of 2 are unique (binary representation).
    # Products of distinct powers of 2 are unique (exponents add).
    # So distinct sums = 35, distinct prods = 35. Score = 35.
    values_p2 = [2 ** i for i in range(n)]
    if max(values_p2) <= max_val:
        # Find best edges for these values
        # Since all sums/prods are unique, any edge set gives score 35.
        # But we need to return a valid edge set.
        edges_p2 = [list(e) for e in combinations(range(n), 2)][:m]
        score_p2 = _score(values_p2, edges_p2)
        if score_p2 < best_overall_score:
            best_overall_score = score_p2
            best_overall_values = values_p2
            best_overall_edges = edges_p2
    
    # Strategy 2: Small integers with many factors
    # Try to find values where many sums/products collide.
    # Example: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    # This is heuristic. Let's try a few hand-picked sets and random sets.
    
    candidate_values = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    ]
    
    # Also try some random sets with small numbers
    random.seed(42)
    for _ in range(20):
        # Generate 14 distinct positive integers <= 10000
        # Bias towards small numbers to increase collision chance
        vals = []
        while len(vals) < n:
            # Pick from 1..50 with high probability, else 1..10000
            if random.random() < 0.7:
                v = random.randint(1, 50)
            else:
                v = random.randint(1, 10000)
            if v not in vals:
                vals.append(v)
        candidate_values.append(vals)
    
    for vals in candidate_values:
        if time.time() > deadline - 10:
            break
        if len(set(vals)) != n or any(v < 1 or v > max_val for v in vals):
            continue
        # Optimize edges for these values
        score, edges = _try_improve_edges(vals, m, n, max_iter=2000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vals
            best_overall_edges = edges
    
    # If we haven't found anything better than 35, return the powers of 2 solution
    if best_overall_values is None:
        best_overall_values = [1] * n  # Fallback, but this is invalid (distinct required)
        # Actually, powers of 2 is always valid
        best_overall_values = [2 ** i for i in range(n)]
        best_overall_edges = [list(e) for e in combinations(range(n), 2)][:m]
    
    # Ensure types are correct
    result_values = [int(v) for v in best_overall_values]
    result_edges = [list(map(int, e)) for e in best_overall_edges]
    
    return {'values': result_values, 'edges': result_edges}
# EVOLVE-BLOCK-END
