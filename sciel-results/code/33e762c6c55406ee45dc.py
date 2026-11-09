"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
ALL_EDGES = list(combinations(range(N), 2))

def _score(values, edges):
    """Compute max(distinct sums, distinct products) over edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _random_values():
    return random.sample(range(1, MAX_VAL + 1), N)

def _random_edges():
    return [list(e) for e in random.sample(ALL_EDGES, M)]

def _optimize_edges(values, edges, deadline):
    """Try to improve edge selection for fixed values."""
    best_edges = [list(e) for e in edges]
    best_score = _score(values, best_edges)
    edge_set = set(map(tuple, best_edges))
    
    # Greedy: try to swap an edge with a non-edge to reduce score
    non_edges = [e for e in ALL_EDGES if e not in edge_set]
    
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle for randomness
        random.shuffle(non_edges)
        for new_e in non_edges:
            if time.time() > deadline:
                break
            for old_e in best_edges:
                if tuple(new_e) in edge_set:
                    continue
                # Swap
                candidate = [e for e in best_edges if e != old_e]
                candidate.append(list(new_e))
                s = _score(values, candidate)
                if s < best_score:
                    best_edges = candidate
                    best_score = s
                    edge_set.remove(tuple(old_e))
                    edge_set.add(tuple(new_e))
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_edges, best_score

def _optimize_values(values, edges, deadline):
    """Try to improve values for fixed edges via local search."""
    best_values = list(values)
    best_score = _score(best_values, edges)
    
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        # Try replacing one value
        idx = random.randint(0, N - 1)
        old_val = best_values[idx]
        new_val = random.randint(1, MAX_VAL)
        while new_val == old_val or new_val in best_values:
            new_val = random.randint(1, MAX_VAL)
        
        best_values[idx] = new_val
        s = _score(best_values, edges)
        if s < best_score:
            best_score = s
            improved = True
        else:
            best_values[idx] = old_val
        if not improved:
            # Try swapping two values
            i, j = random.sample(range(N), 2)
            best_values[i], best_values[j] = best_values[j], best_values[i]
            s = _score(best_values, edges)
            if s < best_score:
                best_score = s
                improved = True
            else:
                best_values[i], best_values[j] = best_values[j], best_values[i]
    return best_values, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_result = None
    best_score = float('inf')
    
    # Initial random attempts
    for attempt in range(20):
        if time.time() > deadline:
            break
        values = _random_values()
        edges = _random_edges()
        
        # Optimize edges
        edges, score = _optimize_edges(values, edges, deadline)
        
        # Optimize values
        values, score = _optimize_values(values, edges, deadline)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
    
    # If we found something good, do more focused search
    if best_result is not None and time.time() < deadline:
        base_values, base_edges = best_result
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Perturb from best
            values = list(base_values)
            idx = random.randint(0, N - 1)
            values[idx] = random.randint(1, MAX_VAL)
            while values.count(values[idx]) > 1:
                values[idx] = random.randint(1, MAX_VAL)
            
            edges = [list(e) for e in base_edges]
            # Perturb edges
            for _ in range(random.randint(1, 3)):
                i, j = random.sample(range(M), 2)
                edges[i], edges[j] = edges[j], edges[i]
            
            edges, score = _optimize_edges(values, edges, deadline)
            values, score = _optimize_values(values, edges, deadline)
            
            if score < best_score:
                best_score = score
                best_result = (list(values), [list(e) for e in edges])
    
    if best_result is None:
        values = list(range(1, N + 1))
        edges = [list(e) for e in ALL_EDGES[:M]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
