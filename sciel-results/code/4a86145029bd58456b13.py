"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute max(distinct sums, distinct products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_edge_select(values, m):
    """
    Greedily select m edges that minimize the marginal increase in distinct sums/products.
    Uses a priority queue where the key is the marginal increase (new sums + new products).
    """
    n = len(values)
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    
    current_sums = set()
    current_prods = set()
    selected = []
    
    # Priority queue: (marginal_increase, edge)
    # We'll recompute marginal increase dynamically as we add edges
    
    # For efficiency, track which sums/products are already present
    # and for each candidate edge, compute how many new sums/prods it adds
    
    # Initial: all edges have marginal increase = 2 (one sum, one prod) unless they collide
    # Let's use a different approach: iteratively pick the edge that adds the fewest new distinct values
    
    remaining = list(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_marginal = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        for edge in remaining:
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            
            new_s = 0 if s in current_sums else 1
            new_p = 0 if p in current_prods else 1
            marginal = new_s + new_p
            
            if marginal < best_marginal:
                best_marginal = marginal
                best_edge = edge
                best_new_sums = s
                best_new_prods = p
                # If marginal is 0, we can't do better
                if marginal == 0:
                    break
        
        if best_edge is None:
            break
        
        i, j = best_edge
        s = values[i] + values[j]
        p = values[i] * values[j]
        
        current_sums.add(s)
        current_prods.add(p)
        selected.append([i, j])
        remaining.remove(best_edge)
    
    return selected

def _try_value_set(values):
    """Try a given value set and return the best (score, edges) found."""
    edges = _greedy_edge_select(values, P['m'])
    score = _score(values, edges)
    return score, edges

def _generate_structured_values():
    """
    Generate value sets designed to create additive and multiplicative collisions.
    
    Strategy: Use small integers and numbers with many small factors.
    Also try geometric-like progressions and sets with shared factors.
    """
    candidates = []
    
    # Strategy 1: Small consecutive integers
    candidates.append(list(range(1, 15)))
    
    # Strategy 2: Multiples of a small number (creates product collisions)
    for base in [2, 3, 4, 5]:
        vals = [base * i for i in range(1, 15) if base * i <= P['maximum']]
        if len(vals) >= 14:
            vals = vals[:14]
            candidates.append(vals)
    
    # Strategy 3: Powers and small numbers mix
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Strategy 4: Numbers with many common factors
    # All even numbers
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Strategy 5: Mix of small primes and composites
    candidates.append([1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # Strategy 6: Geometric-ish progression
    candidates.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # Strategy 7: Random search around promising structures
    rng = random.Random(42)
    for _ in range(50):
        # Start from a base structure and perturb
        base = list(range(1, 15))
        for _ in range(5):
            idx = rng.randint(0, 13)
            base[idx] = rng.randint(1, P['maximum'])
        # Ensure distinct
        if len(set(base)) == 14 and all(1 <= v <= P['maximum'] for v in base):
            candidates.append(base)
    
    return candidates

def _local_search(values, edges, max_iter=100):
    """
    Local search: try swapping edges to reduce the score.
    Also try modifying values slightly.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    current_score = _score(values, edges)
    best_score = current_score
    best_edges = list(edges)
    best_values = list(values)
    
    for _ in range(max_iter):
        improved = False
        
        # Try swapping an edge in with an edge out
        edge_set = set(tuple(sorted(e)) for e in best_edges)
        
        # Try removing one edge and adding another
        for i, e_out in enumerate(best_edges):
            e_out = tuple(sorted(e_out))
            for e_in in all_edges:
                e_in = tuple(sorted(e_in))
                if e_in == e_out or e_in in edge_set:
                    continue
                
                new_edges = list(best_edges)
                new_edges[i] = list(e_in)
                new_score = _score(best_values, new_edges)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    edge_set = set(tuple(sorted(e)) for e in best_edges)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_score, best_values, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    # Generate candidate value sets
    candidates = _generate_structured_values()
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    for values in candidates:
        if time.time() > deadline - 5:
            break
        
        # Ensure values are valid: distinct, positive, <= 10000, exactly 14
        if len(values) != 14:
            continue
        if len(set(values)) != 14:
            continue
        if any(not isinstance(v, int) or v < 1 or v > P['maximum'] for v in values):
            continue
        
        score, edges = _try_value_set(values)
        
        if score < best_score:
            best_score = score
            best_values = list(values)
            best_edges = list(edges)
        
        # Local search on edges
        if time.time() < deadline - 10:
            ls_score, ls_values, ls_edges = _local_search(values, edges, max_iter=50)
            if ls_score < best_score:
                best_score = ls_score
                best_values = ls_values
                best_edges = ls_edges
    
    # If we still have time, do more random value set searches
    if time.time() < deadline - 20 and best_values is not None:
        rng = random.Random(123)
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best values found so far
            values = list(best_values)
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, 13)
                values[idx] = rng.randint(1, P['maximum'])
            
            if len(set(values)) != 14:
                continue
            if any(v < 1 or v > P['maximum'] for v in values):
                continue
            
            score, edges = _try_value_set(values)
            if score < best_score:
                best_score = score
                best_values = list(values)
                best_edges = list(edges)
                
                # Quick local search
                if time.time() < deadline - 5:
                    ls_score, ls_values, ls_edges = _local_search(values, edges, max_iter=30)
                    if ls_score < best_score:
                        best_score = ls_score
                        best_values = ls_values
                        best_edges = ls_edges
    
    # Final validation and type conversion
    if best_values is None or best_edges is None:
        # Fallback to the original
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]}
    
    # Ensure types are correct
    values = [int(v) for v in best_values]
    edges = [[int(i), int(j)] for i, j in best_edges]
    
    # Validate
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10000 for v in values)
    assert len(edges) == 35
    edge_set = set()
    for i, j in edges:
        assert isinstance(i, int) and not isinstance(i, bool)
        assert isinstance(j, int) and not isinstance(j, bool)
        assert 0 <= i < 14 and 0 <= j < 14
        assert i != j
        key = (min(i, j), max(i, j))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': values, 'edges': edges}

# EVOLVE-BLOCK-END
