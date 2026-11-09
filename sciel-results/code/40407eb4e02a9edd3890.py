"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return max(len(sums), len(products))

def _evaluate(values, edges):
    return _score(values, edges)

def _random_values():
    return random.sample(range(1, 10001), 14)

def _random_edges():
    all_edges = list(combinations(range(14), 2))
    return [list(e) for e in random.sample(all_edges, 35)]

def _local_search(values, edges, max_iters=5000, time_budget=45.0):
    """Local search: mutate values or edges to reduce score."""
    current_score = _score(values, edges)
    best_score = current_score
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    start = time.time()
    
    all_edges = list(combinations(range(14), 2))
    all_edges_set = set(tuple(e) for e in all_edges)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Choose mutation type
        if random.random() < 0.5:
            # Mutate a value
            i = random.randint(0, 13)
            old_val = values[i]
            # Try nearby values or random
            if random.random() < 0.3:
                new_val = random.randint(1, 10000)
            else:
                delta = random.choice([-100, -50, -10, -5, -2, -1, 1, 2, 5, 10, 50, 100])
                new_val = old_val + delta
                if new_val < 1:
                    new_val = 1
                elif new_val > 10000:
                    new_val = 10000
            
            # Ensure distinctness
            if any(new_val == values[j] for j in range(14) if j != i):
                continue
            
            values[i] = new_val
            new_score = _score(values, edges)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(values)
                    best_edges = [list(e) for e in edges]
            else:
                values[i] = old_val
        else:
            # Mutate an edge: remove one, add another
            if len(edges) == 0:
                continue
            rem_idx = random.randint(0, len(edges) - 1)
            removed = tuple(edges[rem_idx])
            # Try adding a different edge
            candidates = [e for e in all_edges if e != removed and e not in set(tuple(ed) for ed in edges)]
            if not candidates:
                continue
            added = random.choice(candidates)
            
            new_edges = [list(e) for e in edges if e != list(removed)]
            new_edges.append(list(added))
            
            new_score = _score(values, new_edges)
            if new_score < current_score:
                current_score = new_score
                edges = new_edges
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(values)
                    best_edges = [list(e) for e in edges]
            # else: keep old edges
    
    return best_values, best_edges, best_score

def _structured_initialization(method):
    """Generate structured initial values and edges."""
    if method == 'consecutive':
        values = list(range(1, 15))
    elif method == 'powers2':
        values = [2**i for i in range(14)]
    elif method == 'fibonacci':
        fibs = [1, 1]
        while len(fibs) < 14:
            fibs.append(fibs[-1] + fibs[-2])
        values = fibs[:14]
    elif method == 'arithmetic':
        values = [1 + i * 10 for i in range(14)]
    elif method == 'geometric':
        values = [10**i for i in range(14)]
        # Cap at 10000
        values = [min(v, 10000) for v in values]
        # Ensure distinct
        seen = set()
        new_vals = []
        for v in values:
            if v not in seen:
                seen.add(v)
                new_vals.append(v)
        while len(new_vals) < 14:
            new_vals.append(len(new_vals) + 1)
        values = new_vals[:14]
    else:
        values = _random_values()
    
    # For edges, try to pick edges that create sum collisions
    # Start with random edges
    edges = _random_edges()
    return values, edges

def construct():
    deadline = 85.0  # Safety margin before 90s kill
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Try multiple structured initializations with local search
    methods = ['consecutive', 'powers2', 'fibonacci', 'arithmetic', 'geometric', 'random1', 'random2', 'random3']
    
    for method in methods:
        if time.time() - start_time > deadline:
            break
        
        if method.startswith('random'):
            values = _random_values()
            edges = _random_edges()
        else:
            values, edges = _structured_initialization(method)
        
        # Run local search with time budget
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
        time_budget = max(1.0, remaining * 0.7)
        
        bv, be, bs = _local_search(values, edges, max_iters=3000, time_budget=time_budget)
        
        if bs < best_score:
            best_score = bs
            best_result = {'values': bv, 'edges': be}
        
        if time.time() - start_time > deadline:
            break
    
    # If no result found, return a valid default
    if best_result is None:
        values = list(range(1, 15))
        all_edges = list(combinations(range(14), 2))
        edges = [list(e) for e in all_edges[:35]]
        best_result = {'values': values, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
