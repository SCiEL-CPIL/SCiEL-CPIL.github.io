"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000

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
    # Try several structured initializations
    candidates = []
    
    # Consecutive small numbers
    candidates.append(list(range(1, N + 1)))
    
    # Powers of 2 (scaled to stay under MAX_VAL)
    powers = [2**i for i in range(N)]
    if max(powers) <= MAX_VAL:
        candidates.append(powers)
    else:
        # Scale down
        scale = MAX_VAL // max(powers)
        if scale > 0:
            scaled = [min(MAX_VAL, p * scale) for p in powers]
            # Ensure distinct
            seen = set()
            unique = []
            for v in scaled:
                if v not in seen and v > 0:
                    seen.add(v)
                    unique.append(v)
            while len(unique) < N:
                unique.append(len(unique) + 1)
            candidates.append(unique[:N])
    
    # Fibonacci-like
    fib = [1, 1]
    for i in range(2, N):
        fib.append(fib[-1] + fib[-2])
    if max(fib) <= MAX_VAL:
        candidates.append(fib)
    else:
        scale = MAX_VAL // max(fib)
        if scale > 0:
            scaled = [min(MAX_VAL, f * scale) for f in fib]
            seen = set()
            unique = []
            for v in scaled:
                if v not in seen and v > 0:
                    seen.add(v)
                    unique.append(v)
            while len(unique) < N:
                unique.append(len(unique) + 1)
            candidates.append(unique[:N])
    
    # Random sample
    candidates.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # Small consecutive
    candidates.append(list(range(10, 10 + N)))
    
    return candidates

def _generate_edges():
    """Generate 35 random edges from all possible edges."""
    all_edges = list(combinations(range(N), 2))
    return [list(e) for e in random.sample(all_edges, M)]

def _local_search(values, edges, max_iters=5000, deadline=None):
    """Perform local search by mutating values and edges."""
    best_score = _score(values, edges)
    best_values = values[:]
    best_edges = [e[:] for e in edges]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try mutating a random value
        if random.random() < 0.5:
            # Value mutation
            new_values = best_values[:]
            idx = random.randint(0, N - 1)
            # Choose a new value
            if random.random() < 0.3:
                # Small perturbation
                delta = random.randint(-10, 10)
                new_val = max(1, min(MAX_VAL, new_values[idx] + delta))
            else:
                # Random new value
                new_val = random.randint(1, MAX_VAL)
            
            # Ensure distinct
            if new_val not in new_values or new_val == new_values[idx]:
                new_values[idx] = new_val
                # Fix duplicates if any
                seen = set()
                for i in range(N):
                    if new_values[i] in seen:
                        # Find a new unique value
                        while True:
                            new_values[i] = random.randint(1, MAX_VAL)
                            if new_values[i] not in seen:
                                break
                    seen.add(new_values[i])
                
                new_score = _score(new_values, best_edges)
                if new_score <= best_score:
                    best_score = new_score
                    best_values = new_values
        else:
            # Edge mutation
            new_edges = [e[:] for e in best_edges]
            # Remove a random edge and add another
            if len(new_edges) > 0:
                idx = random.randint(0, len(new_edges) - 1)
                removed_edge = new_edges.pop(idx)
                
                # Find a new edge
                all_edges = list(combinations(range(N), 2))
                existing = set(tuple(sorted(e)) for e in new_edges)
                candidates = [e for e in all_edges if tuple(sorted(e)) not in existing]
                if candidates:
                    new_edge = random.choice(candidates)
                    new_edges.append([new_edge[0], new_edge[1]])
                
                new_score = _score(best_values, new_edges)
                if new_score <= best_score:
                    best_score = new_score
                    best_edges = new_edges
    
    return best_values, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Generate multiple initial candidates
    init_candidates = _generate_initial_values()
    
    for init_vals in init_candidates:
        if time.time() > deadline:
            break
        
        # Ensure valid
        if len(set(init_vals)) != N or any(v < 1 or v > MAX_VAL for v in init_vals):
            continue
        
        init_edges = _generate_edges()
        
        # Local search
        vals, edges, score = _local_search(init_vals, init_edges, max_iters=3000, deadline=deadline)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vals
            best_overall_edges = edges
    
    # If we haven't found anything, use a default
    if best_overall_values is None:
        best_overall_values = list(range(1, N + 1))
        best_overall_edges = [list(e) for e in list(combinations(range(N), 2))[:M]]
        best_overall_score = _score(best_overall_values, best_overall_edges)
    
    # Final polish: try a few more iterations if time permits
    if time.time() < deadline - 5:
        vals, edges, score = _local_search(best_overall_values, best_overall_edges, max_iters=2000, deadline=deadline)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vals
            best_overall_edges = edges
    
    # Ensure types are correct
    result_values = [int(v) for v in best_overall_values]
    result_edges = [[int(i), int(j)] for i, j in best_overall_edges]
    
    return {'values': result_values, 'edges': result_edges}
# EVOLVE-BLOCK-END
