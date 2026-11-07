"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
DEADLINE_MARGIN = 5.0  # seconds before the 90s wall-clock limit

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_edges(values, m=M):
    """Greedy edge selection to minimize max(distinct sums, distinct products)."""
    n = len(values)
    # Precompute all edge data
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((s, p, i, j))
    
    # Sort by sum first, then product (smaller values tend to create more collisions)
    all_edges.sort(key=lambda x: (x[0], x[1]))
    
    sums = set()
    prods = set()
    chosen = []
    for s, p, i, j in all_edges:
        if len(chosen) >= m:
            break
        # Prefer edges that don't add new sums/products
        new_s = 0 if s in sums else 1
        new_p = 0 if p in prods else 1
        # Score: prefer edges with minimal new additions
        # We'll just take them in sorted order, but let's be smarter:
        # Actually, let's do a proper greedy with scoring
        chosen.append((i, j))
        sums.add(s)
        prods.add(p)
    
    # Second pass: try to improve by swapping
    # For each chosen edge, see if replacing it with an unchosen edge reduces the raw value
    chosen_set = set(chosen)
    unchosen = [(s, p, i, j) for (s, p, i, j) in all_edges if (i, j) not in chosen_set and (j, i) not in chosen_set]
    
    for _ in range(3):  # A few improvement passes
        improved = False
        for idx, (ci, cj) in enumerate(chosen):
            # Remove this edge and recompute
            old_sums = set()
            old_prods = set()
            for (ki, kj) in chosen:
                if (ki, kj) != (ci, cj) and (kj, ki) != (ci, cj):
                    old_sums.add(values[ki] + values[kj])
                    old_prods.add(values[ki] * values[kj])
            
            # Try adding each unchosen edge
            best_new = None
            best_raw = _raw_value(values, chosen)
            for (s, p, ui, uj) in unchosen:
                if (ui, uj) in chosen_set or (uj, ui) in chosen_set:
                    continue
                new_sums = old_sums | {s}
                new_prods = old_prods | {p}
                raw = max(len(new_sums), len(new_prods))
                if raw < best_raw:
                    best_raw = raw
                    best_new = (ui, uj)
            if best_new is not None:
                chosen[idx] = best_new
                chosen_set.discard((ci, cj))
                chosen_set.discard((cj, ci))
                chosen_set.add(best_new)
                improved = True
        if not improved:
            break
    
    return [list(e) for e in chosen]

def _powers_of_2_solution():
    """Baseline: 14 distinct powers of 2. Sums all unique (35), products ≤ 13 distinct."""
    values = [2**i for i in range(N)]
    edges = [list(e) for e in list(combinations(range(N), 2))[:M]]
    return values, edges

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # Start with the guaranteed baseline
    best_values, best_edges = _powers_of_2_solution()
    best_raw = _raw_value(best_values, best_edges)  # Should be 35
    
    # Local search: try random value sets and greedy edge selection
    random.seed(42)
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        # Generate a random set of 14 distinct positive integers <= 10000
        # Bias toward small values to increase collision likelihood
        if attempts % 3 == 0:
            # Try structured sets: arithmetic progression
            a = random.randint(1, 50)
            d = random.randint(1, 10)
            vals = [a + d * i for i in range(N)]
            if max(vals) > MAX_VAL or len(set(vals)) != N:
                continue
        elif attempts % 3 == 1:
            # Random small values
            vals = random.sample(range(1, 200), N)
        else:
            # Random values with some structure
            base = random.sample(range(1, 100), 7)
            vals = base + [v * 2 for v in base]
            vals = random.sample(vals, N)
        
        vals = sorted(vals)
        if len(set(vals)) != N or max(vals) > MAX_VAL:
            continue
        
        edges = _greedy_edges(vals)
        if len(edges) < M:
            continue
        raw = _raw_value(vals, edges)
        if raw < best_raw:
            best_raw = raw
            best_values = list(vals)
            best_edges = [list(e) for e in edges]
            if best_raw <= 34:
                break  # Found improvement, stop early
    
    # Ensure types are correct
    result = {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
    
    # Final validation
    assert len(result['values']) == N
    assert len(set(result['values'])) == N
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in result['values'])
    assert all(1 <= v <= MAX_VAL for v in result['values'])
    assert len(result['edges']) == M
    edge_set = set()
    for e in result['edges']:
        assert isinstance(e, list) and len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert i != j
        assert 0 <= i < N and 0 <= j < N
        key = (min(i, j), max(i, j))
        assert key not in edge_set
        edge_set.add(key)
    
    return result
# EVOLVE-BLOCK-END
