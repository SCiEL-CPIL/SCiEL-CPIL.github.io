"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _all_edges(n):
    return [list(e) for e in combinations(range(n), 2)]

def _greedy_edge_select(values, target_m, max_distinct):
    """Select edges greedily, preferring those that create collisions."""
    n = len(values)
    all_edges = _all_edges(n)
    random.shuffle(all_edges)
    
    selected = []
    sums_set = set()
    prods_set = set()
    
    for e in all_edges:
        if len(selected) >= target_m:
            break
        i, j = e
        s = values[i] + values[j]
        p = values[i] * values[j]
        
        # Prefer edges that create collisions
        s_collision = s in sums_set
        p_collision = p in prods_set
        
        if s_collision or p_collision:
            selected.append(e)
            sums_set.add(s)
            prods_set.add(p)
    
    # Fill remaining with any edges
    if len(selected) < target_m:
        existing = set(tuple(sorted(e)) for e in selected)
        for e in all_edges:
            if len(selected) >= target_m:
                break
            key = tuple(sorted(e))
            if key not in existing:
                selected.append(e)
                existing.add(key)
                s = values[e[0]] + values[e[1]]
                p = values[e[0]] * values[e[1]]
                sums_set.add(s)
                prods_set.add(p)
    
    return selected

def _local_search_edges(values, edges, max_iterations=500, deadline=None):
    """Local search to improve edge configuration."""
    n = len(values)
    all_edges = _all_edges(n)
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_edges = [list(e) for e in edges]
    current_score = _raw_score(values, current_edges)
    
    best_edges = [list(e) for e in current_edges]
    best_score = current_score
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Try removing an edge and adding a different one
        for remove_idx in range(len(current_edges)):
            if deadline and time.time() > deadline:
                break
            
            removed = current_edges[remove_idx]
            new_edges = current_edges[:remove_idx] + current_edges[remove_idx+1:]
            edge_set_new = set(tuple(sorted(e)) for e in new_edges)
            
            # Try adding a random edge
            for _ in range(10):
                candidate = random.choice(all_edges)
                key = tuple(sorted(candidate))
                if key in edge_set_new:
                    continue
                
                test_edges = new_edges + [candidate]
                test_score = _raw_score(values, test_edges)
                
                if test_score < best_score:
                    best_score = test_score
                    best_edges = [list(e) for e in test_edges]
                    current_edges = test_edges
                    edge_set = set(tuple(sorted(e)) for e in current_edges)
                    break
    
    return best_edges, best_score

def _try_value_set(values, deadline, max_iters=100):
    """Try a value set with greedy + local search."""
    n = len(values)
    m = P['m']
    
    # Multiple restarts
    best_edges = None
    best_score = float('inf')
    
    for restart in range(max_iters):
        if time.time() > deadline:
            break
        
        # Greedy edge selection
        edges = _greedy_edge_select(values, m, float('inf'))
        score = _raw_score(values, edges)
        
        if score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
        
        # Local search improvement
        if len(edges) == m:
            improved_edges, improved_score = _local_search_edges(
                values, edges, 
                max_iterations=50,
                deadline=deadline
            )
            if improved_score < best_score:
                best_score = improved_score
                best_edges = improved_edges
    
    return best_edges, best_score

def _generate_orbit_values(seed_offset=0):
    """Generate values in multiplicative orbits to force product collisions.
    
    Strategy: Use values of the form a * 2^k where a is odd.
    Products: (a*2^i)(b*2^j) = ab * 2^(i+j)
    If we group values by their odd part, products within same odd-part group
    will have the same odd part, creating collisions in the odd component.
    """
    random.seed(42 + seed_offset)
    
    # Use small odd bases and powers of 2
    # Values: base * 2^k for k=0,1,2,...
    # This creates multiplicative structure
    
    n = P['n']
    max_val = P['maximum']
    
    # Choose odd bases
    odd_bases = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    random.shuffle(odd_bases)
    
    values = []
    used = set()
    
    # Generate values with multiplicative structure
    # Group by odd part to create product collisions
    for base in odd_bases[:n]:
        # Pick a power of 2 multiplier
        # Keep values distinct and <= max_val
        k = random.randint(0, 12)
        val = base * (2 ** k)
        
        # Ensure distinctness
        attempts = 0
        while val in used or val > max_val or val < 1:
            k = random.randint(0, 12)
            val = base * (2 ** k)
            attempts += 1
            if attempts > 100:
                # Fallback: try next base
                break
        
        if val not in used and 1 <= val <= max_val:
            values.append(val)
            used.add(val)
    
    # If we don't have enough, fill with distinct values
    if len(values) < n:
        for v in range(1, max_val+1):
            if len(values) >= n:
                break
            if v not in used:
                values.append(v)
                used.add(v)
    
    # Ensure exactly n distinct values
    if len(values) != n:
        # Fallback to simple distinct values
        values = list(range(1, n+1))
    
    return values

def _generate_structured_values(seed_offset=0):
    """Generate values with more deliberate multiplicative structure.
    
    Use values that are products of small primes to create many product collisions.
    """
    random.seed(42 + seed_offset)
    
    n = P['n']
    max_val = P['maximum']
    
    # Use values composed of small primes: 2, 3, 5, 7
    # This creates many factorization collisions
    
    prime_powers = [2**i for i in range(0, 14)] + [3**i for i in range(0, 8)] + \
                   [5**i for i in range(0, 5)] + [7**i for i in range(0, 4)]
    
    # Also include composites
    composites = []
    for a in [2, 3, 5, 7]:
        for b in [2, 3, 5, 7]:
            if a != b:
                for c in [2, 3, 5, 7]:
                    val = a * b * c
                    if val <= max_val:
                        composites.append(val)
    
    all_candidates = prime_powers + composites
    random.shuffle(all_candidates)
    
    values = []
    used = set()
    
    for v in all_candidates:
        if len(values) >= n:
            break
        if v not in used and 1 <= v <= max_val:
            values.append(v)
            used.add(v)
    
    # Fill if needed
    if len(values) < n:
        for v in range(1, max_val+1):
            if len(values) >= n:
                break
            if v not in used:
                values.append(v)
                used.add(v)
    
    if len(values) != n:
        values = list(range(1, n+1))
    
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_edges = None
    best_score = float('inf')
    best_values = None
    
    # Try different value generation strategies
    strategies = [
        _generate_orbit_values,
        _generate_structured_values,
    ]
    
    for strat_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            
            values = strategy(seed)
            
            if len(values) != P['n'] or len(set(values)) != P['n']:
                continue
            if any(v < 1 or v > P['maximum'] for v in values):
                continue
            
            edges, score = _try_value_set(values, deadline, max_iters=5)
            
            if edges and score < best_score:
                best_score = score
                best_edges = edges
                best_values = values
    
    # Fallback if nothing found
    if best_edges is None or best_values is None:
        best_values = list(range(1, P['n']+1))
        all_e = _all_edges(P['n'])
        best_edges = all_e[:P['m']]
        best_score = _raw_score(best_values, best_edges)
    
    # Ensure types are correct
    result = {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
    
    return result
# EVOLVE-BLOCK-END
