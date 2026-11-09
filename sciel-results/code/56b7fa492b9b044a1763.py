"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
ALL_EDGES = list(combinations(range(N), 2))

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _generate_initial_values():
    """Generate initial values using structured approach."""
    # Use small values to keep products manageable
    # Try arithmetic progression first
    return list(range(1, N + 1))

def _generate_random_edges():
    """Generate random 35 edges."""
    return [list(e) for e in random.sample(ALL_EDGES, M)]

def _local_search(values, edges, max_iters=5000, time_limit=None):
    """Perform local search by mutating values and edges."""
    best_score = _score(values, edges)
    best_values = values[:]
    best_edges = [e[:] for e in edges]
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iters):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        
        # Choose mutation type
        if random.random() < 0.7:
            # Mutate a value
            new_values = best_values[:]
            idx = random.randint(0, N - 1)
            old_val = new_values[idx]
            new_val = random.randint(1, MAX_VAL)
            if new_val != old_val and new_val not in new_values:
                new_values[idx] = new_val
            else:
                # Try nearby value
                for delta in [1, -1, 2, -2, 3, -3]:
                    candidate = old_val + delta
                    if 1 <= candidate <= MAX_VAL and candidate not in new_values:
                        new_values[idx] = candidate
                        break
            new_edges = best_edges
        else:
            # Mutate an edge: swap one edge
            new_values = best_values
            new_edges = [e[:] for e in best_edges]
            edge_idx = random.randint(0, M - 1)
            old_edge = tuple(new_edges[edge_idx])
            # Remove old edge and add a new random edge
            available = [e for e in ALL_EDGES if e != old_edge and e not in [tuple(x) for x in new_edges]]
            if available:
                new_edge = random.choice(available)
                new_edges[edge_idx] = list(new_edge)
        
        new_score = _score(new_values, new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_values = new_values
            best_edges = new_edges
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_score, best_values, best_edges

def construct():
    """Construct the solution."""
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Try multiple initializations
    initializations = [
        list(range(1, N + 1)),  # Consecutive
        [2**i for i in range(N)],  # Powers of 2
        [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],  # Fibonacci
    ]
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # Make sure values are distinct and within range
        if len(set(init)) != N or any(v < 1 or v > MAX_VAL for v in init):
            continue
        
        values = init[:]
        edges = _generate_random_edges()
        
        score, values, edges = _local_search(values, edges, max_iters=2000, 
                                              time_limit=deadline - time.time())
        
        if score < best_score:
            best_score = score
            best_values = values
            best_edges = edges
        
        # Early exit if we reach a good score
        if best_score <= 12:
            break
    
    # If we didn't find anything, use a fallback
    if best_values is None:
        best_values = list(range(1, N + 1))
        best_edges = _generate_random_edges()
        best_score = _score(best_values, best_edges)
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
