"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _count_distinct(vals, edges):
    """Count distinct sums and products."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = vals[u], vals[v]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    # Initial guess: Use a set of numbers with many factors to encourage product collisions
    # 1, 2, 3, 4, 6, 12, 24, 36, 48, 60, 72, 96, 120, 144
    # These are highly composite or multiples thereof.
    initial_values = [1, 2, 3, 4, 6, 12, 24, 36, 48, 60, 72, 96, 120, 144]
    
    # Ensure distinct and positive
    initial_values = sorted(set(initial_values))
    if len(initial_values) < n:
        # Pad with random small numbers if needed (should not happen with above list)
        while len(initial_values) < n:
            initial_values.append(random.randint(1, 200))
            initial_values = sorted(set(initial_values))
    initial_values = initial_values[:n]
    
    # Initial edges: Random sample of m edges
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    current_edges = [list(e) for e in all_edges[:m]]
    
    current_vals = initial_values[:]
    current_score = _score(current_vals, current_edges)
    best_score = current_score
    best_vals = current_vals[:]
    best_edges = [list(e) for e in current_edges]
    
    # Precompute some candidate values for mutation
    # Focus on small numbers and multiples of small primes
    candidate_pool = list(range(1, 100)) + [i * j for i in range(1, 10) for j in range(1, 10) if i * j <= 1000]
    candidate_pool = list(set(candidate_pool))
    
    # Local search loop
    while time.time() < deadline:
        # Try mutating values
        if random.random() < 0.5:
            new_vals = current_vals[:]
            idx = random.randint(0, n - 1)
            # Pick a value from the pool, ensuring it's distinct from others
            new_val = random.choice(candidate_pool)
            attempts = 0
            while new_val in new_vals and attempts < 50:
                new_val = random.choice(candidate_pool)
                attempts += 1
            if new_val not in new_vals:
                new_vals[idx] = new_val
                # Check constraints
                if all(1 <= v <= max_val for v in new_vals) and len(set(new_vals)) == n:
                    new_score = _score(new_vals, current_edges)
                    if new_score < current_score:
                        current_vals = new_vals
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_vals = current_vals[:]
                            best_edges = [list(e) for e in current_edges]
                    elif new_score == current_score and random.random() < 0.1:
                        # Accept equal score with small probability to escape local optima
                        current_vals = new_vals
                        current_score = new_score
        else:
            # Try mutating edges: swap one edge for another
            new_edges = [list(e) for e in current_edges]
            edge_to_remove_idx = random.randint(0, m - 1)
            edge_to_remove = new_edges.pop(edge_to_remove_idx)
            
            # Find a new edge not already in the set
            # Simple approach: pick a random pair from all possible edges
            all_pairs = list(combinations(range(n), 2))
            current_edge_set = set(tuple(sorted(e)) for e in new_edges)
            
            # Sample a few candidates to find a valid one
            for _ in range(20):
                cand = random.choice(all_pairs)
                if tuple(sorted(cand)) not in current_edge_set:
                    new_edges.append(list(cand))
                    break
            else:
                # If we couldn't find a new edge easily, restore
                new_edges.append(edge_to_remove)
                continue
                
            new_score = _score(current_vals, new_edges)
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_vals = current_vals[:]
                    best_edges = [list(e) for e in current_edges]
            elif new_score == current_score and random.random() < 0.05:
                current_edges = new_edges
                current_score = new_score
        
        # Occasional restart if stuck? 
        # For now, simple hill climbing with acceptance of equal moves is enough for a start.
        
    # Return best found
    return {
        'values': [int(v) for v in best_vals],
        'edges': [list(map(int, e)) for e in best_edges]
    }
# EVOLVE-BLOCK-END
