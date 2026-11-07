"""Rewritten program for small sums/products on 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000

def _compute_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_edges_for_values(values, m=35, deadline=None):
    """Find m edges minimizing distinct sums/products."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_info = []
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_info.append((u, v, s, p))
    
    # Sort by sum first, then by product
    # We want to pick edges that reuse sums and products
    # Greedy: start with most common sums/products
    
    # Count sum frequencies
    sum_counts = {}
    prod_counts = {}
    for u, v, s, p in edge_info:
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    # Sort edges: prefer edges with more common sums and products
    def edge_priority(edge):
        u, v, s, p = edge
        # Prefer edges where sum and product are already common
        return (-sum_counts[s], -prod_counts[p], s, p)
    
    sorted_edges = sorted(edge_info, key=edge_priority)
    
    # Greedy selection with lookahead
    selected = []
    used_sums = set()
    used_prods = set()
    
    # First pass: pick edges that minimize new distinct values
    for edge in sorted_edges:
        if len(selected) >= m:
            break
        u, v, s, p = edge
        # Prefer edges that don't add new sums/products
        new_sum = s not in used_sums
        new_prod = p not in used_prods
        selected.append(edge)
        used_sums.add(s)
        used_prods.add(p)
    
    if len(selected) < m:
        # Fill remaining
        for edge in edge_info:
            if len(selected) >= m:
                break
            if edge not in selected:
                selected.append(edge)
                used_sums.add(edge[2])
                used_prods.add(edge[3])
    
    # Local search: try swapping edges to reduce distinct count
    best_score = _compute_score(values, [(e[0], e[1]) for e in selected])
    
    # Try simple swaps
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(min(len(selected), m)):
            for j in range(len(all_edges)):
                if all_edges[j] in [(e[0], e[1]) for e in selected]:
                    continue
                # Try swapping selected[i] with all_edges[j]
                new_selected = list(selected)
                new_selected[i] = (all_edges[j][0], all_edges[j][1],
                                   values[all_edges[j][0]] + values[all_edges[j][1]],
                                   values[all_edges[j][0]] * values[all_edges[j][1]])
                new_score = _compute_score(values, [(e[0], e[1]) for e in new_selected])
                if new_score < best_score:
                    selected = new_selected
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return [(e[0], e[1]) for e in selected[:m]]

def _try_value_set(values, deadline=None):
    """Given values, find best edges."""
    edges = _greedy_edges_for_values(values, M, deadline)
    if not edges:
        return None, float('inf')
    score = _compute_score(values, edges)
    return edges, score

def _generate_values_random():
    """Generate random distinct positive integers <= 10000."""
    return random.sample(range(1, MAX_VAL + 1), N)

def _generate_values_structured(style):
    """Generate structured value sets."""
    if style == 0:
        # Arithmetic progression
        start = random.randint(1, 50)
        step = random.randint(1, 200)
        vals = [start + i * step for i in range(N)]
        if max(vals) <= MAX_VAL and len(set(vals)) == N:
            return vals
        return list(range(1, N + 1))
    elif style == 1:
        # Geometric-like
        vals = []
        base = random.randint(2, 10)
        for i in range(N):
            v = base ** (i // 3) * (1 + (i % 3))
            if v > MAX_VAL:
                v = MAX_VAL - (N - i - 1)
            vals.append(v)
        if len(set(vals)) == N and all(1 <= v <= MAX_VAL for v in vals):
            return vals
        return list(range(1, N + 1))
    elif style == 2:
        # Mixed: some small, some large
        small = random.sample(range(1, 20), N // 2)
        large = random.sample(range(1000, MAX_VAL), N - N // 2)
        vals = small + large
        if len(set(vals)) == N:
            return vals
        return list(range(1, N + 1))
    elif style == 3:
        # Fibonacci-like
        vals = [1, 1]
        while len(vals) < N:
            vals.append(vals[-1] + vals[-2])
        vals = vals[:N]
        if max(vals) <= MAX_VAL and len(set(vals)) == N:
            return vals
        return list(range(1, N + 1))
    elif style == 4:
        # Squares
        vals = [(i + 1) ** 2 for i in range(N)]
        if max(vals) <= MAX_VAL:
            return vals
        return list(range(1, N + 1))
    else:
        # Prime-like
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        vals = []
        x = 1
        while len(vals) < N:
            if is_prime(x):
                vals.append(x)
            x += 1
        if max(vals) <= MAX_VAL:
            return vals
        return list(range(1, N + 1))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_values = list(range(1, N + 1))
    best_edges = None
    best_score = float('inf')
    
    # Evaluate initial set
    edges, score = _try_value_set(best_values, deadline)
    if edges:
        best_edges = [list(e) for e in edges]
        best_score = score
    
    random.seed(42)
    
    # Try various value generation strategies
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Cycle through generation strategies
        if attempts % 5 == 0:
            vals = _generate_values_structured((attempts // 5) % 6)
        else:
            vals = _generate_values_random()
        
        # Validate
        if len(set(vals)) != N or any(v <= 0 or v > MAX_VAL for v in vals):
            continue
        
        edges, score = _try_value_set(vals, deadline)
        if not edges:
            continue
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = [list(e) for e in edges]
        
        # If we've found a very good score, we can stop
        if best_score <= 10:
            break
        
        # If we've tried many, increase deadline check frequency
        if attempts % 100 == 0 and time.time() > deadline - 5:
            break
    
    # Ensure edges is not None
    if best_edges is None:
        # Fallback: first M edges from complete graph
        all_edges = list(combinations(range(N), 2))
        best_edges = [list(e) for e in all_edges[:M]]
        best_score = _compute_score(best_values, best_edges)
    
    # Convert to required types
    return {
        'values': [int(v) for v in best_values],
        'edges': [list(e) for e in best_edges]
    }
# EVOLVE-BLOCK-END
