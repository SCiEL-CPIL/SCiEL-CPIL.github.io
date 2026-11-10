"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _all_edges(n):
    """Generate all possible edges for n vertices."""
    return list(combinations(range(n), 2))

def _greedy_edge_selection(values, n, m):
    """For fixed values, greedily select m edges to minimize max(distinct sums, distinct products)."""
    all_e = _all_edges(n)
    # Precompute sum and product for each edge
    edge_props = []
    for i, j in all_e:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_props.append((i, j, s, p))
    
    # Greedy: at each step, pick the edge that minimizes the resulting max(distinct_sums, distinct_prods)
    # Use a heuristic: prefer edges whose sum or product already exists
    selected = []
    selected_set = set()
    sum_counts = defaultdict(int)
    prod_counts = defaultdict(int)
    
    for _ in range(m):
        best_edge = None
        best_metric = float('inf')
        for idx, (i, j, s, p) in enumerate(edge_props):
            e_key = (i, j)
            if e_key in selected_set:
                continue
            # Metric: how many new distinct sums and products this edge would add
            new_sums = 0 if sum_counts[s] > 0 else 1
            new_prods = 0 if prod_counts[p] > 0 else 1
            # Prefer edges that add fewer new distinct values
            # Also consider the current total
            cur_distinct_sums = len(sum_counts)
            cur_distinct_prods = len(prod_counts)
            new_total_sums = cur_distinct_sums + new_sums
            new_total_prods = cur_distinct_prods + new_prods
            metric = max(new_total_sums, new_total_prods)
            # Tie-break: prefer smaller products (more collision potential)
            if metric < best_metric or (metric == best_metric and (best_edge is None or p < edge_props[best_edge][3])):
                best_metric = metric
                best_edge = idx
        
        if best_edge is not None:
            i, j, s, p = edge_props[best_edge]
            selected.append([i, j])
            selected_set.add((i, j))
            sum_counts[s] += 1
            prod_counts[p] += 1
    
    return selected

def _generate_value_sets(n, max_val=10000, count=20):
    """Generate diverse candidate value sets."""
    sets = []
    
    # 1. Arithmetic progressions with various bases
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for start in range(1, min(10, max_val // n)):
            values = [start + base * i for i in range(n)]
            if all(1 <= v <= max_val for v in values) and len(set(values)) == n:
                sets.append(values)
    
    # 2. Geometric-like sequences (powers of small integers)
    for base in [2, 3]:
        values = [base ** i for i in range(n)]
        if all(1 <= v <= max_val for v in values):
            sets.append(values)
    
    # 3. Multiples with small perturbations
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25]:
        for _ in range(3):
            values = []
            used = set()
            for i in range(n):
                k = random.randint(1, max_val // base)
                offset = random.choice([0, 0, 0, 1, -1, 2, -2, 3])
                v = base * k + offset
                attempts = 0
                while (v in used or v < 1 or v > max_val) and attempts < 100:
                    k = random.randint(1, max_val // base)
                    offset = random.choice([0, 0, 0, 1, -1, 2, -2, 3])
                    v = base * k + offset
                    attempts += 1
                if 1 <= v <= max_val and v not in used:
                    used.add(v)
                    values.append(v)
            if len(values) == n and len(set(values)) == n:
                sets.append(sorted(values))
    
    # 4. Values clustered around a few centers
    for _ in range(5):
        centers = sorted(random.sample(range(100, max_val), 4))
        values = []
        used = set()
        for c in centers:
            for _ in range(n // 4):
                v = c + random.randint(-20, 20)
                attempts = 0
                while (v in used or v < 1 or v > max_val) and attempts < 50:
                    v = c + random.randint(-50, 50)
                    attempts += 1
                if 1 <= v <= max_val and v not in used:
                    used.add(v)
                    values.append(v)
        # Fill remaining
        while len(values) < n:
            v = random.randint(1, max_val)
            if v not in used:
                used.add(v)
                values.append(v)
        if len(set(values)) == n:
            sets.append(sorted(values[:n]))
    
    # 5. Twin-prime-like or structured pairs
    for _ in range(5):
        values = []
        used = set()
        # Generate in pairs that share factors
        for i in range(n // 2):
            base = random.randint(2, 100)
            v1 = base * random.randint(2, 50)
            v2 = base * random.randint(51, 100)
            for v in [v1, v2]:
                if 1 <= v <= max_val and v not in used:
                    used.add(v)
                    values.append(v)
        while len(values) < n:
            v = random.randint(1, max_val)
            if v not in used:
                used.add(v)
                values.append(v)
        if len(set(values)) == n:
            sets.append(sorted(values[:n]))
    
    # Deduplicate
    unique_sets = []
    seen = set()
    for s in sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(list(s))
    
    return unique_sets[:count]

def _simulated_annealing_values(values, n, m, max_time=10):
    """Simulated annealing on values with greedy edge re-selection."""
    start_time = time.time()
    best_values = list(values)
    best_edges = _greedy_edge_selection(best_values, n, m)
    best_score = _score(best_values, best_edges)
    
    current_values = list(values)
    current_edges = _greedy_edge_selection(current_values, n, m)
    current_score = _score(current_values, current_edges)
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time and T > T_min:
        if no_improve > max_no_improve:
            break
        
        # Pick a random vertex to modify
        idx = random.randint(0, n - 1)
        old_val = current_values[idx]
        
        # Generate new value
        move_size = random.choice([1, 2, 3, 5, 10, 20, 50, 100])
        if random.random() < 0.5:
            new_val = old_val + random.randint(-move_size, move_size)
        else:
            new_val = random.randint(1, 10000)
        
        if new_val < 1 or new_val > 10000:
            new_val = random.randint(1, 10000)
        
        # Check distinctness
        if new_val in current_values and new_val != old_val:
            T *= cooling
            continue
        
        trial_values = list(current_values)
        trial_values[idx] = new_val
        trial_edges = _greedy_edge_selection(trial_values, n, m)
        trial_score = _score(trial_values, trial_edges)
        
        # Accept or reject
        if trial_score < current_score or random.random() < pow(2.718281828, -(trial_score - current_score) / T):
            current_values = trial_values
            current_edges = trial_edges
            current_score = trial_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = list(current_values)
                best_edges = [list(e) for e in current_edges]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling
    
    return best_values, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline for safety (90s limit)
    
    n = 14
    m = 35
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Try multiple structured value sets with greedy edge selection
    value_sets = _generate_value_sets(n, max_val=10000, count=40)
    
    for vs in value_sets:
        if time.time() > deadline - 15:
            break
        edges = _greedy_edge_selection(vs, n, m)
        score = _score(vs, edges)
        if score < best_score:
            best_score = score
            best_values = list(vs)
            best_edges = [list(e) for e in edges]
    
    # Phase 2: Simulated annealing from best and several other starting points
    anneal_starts = [best_values]
    # Add a few diverse starting points
    for vs in value_sets[:5]:
        if vs != best_values:
            anneal_starts.append(vs)
    
    for start_vals in anneal_starts:
        if time.time() > deadline - 5:
            break
        
        remaining_time = deadline - time.time() - 2
        if remaining_time < 2:
            break
        
        vals, edges, score = _simulated_annealing_values(start_vals, n, m, max_time=min(remaining_time, 12))
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Phase 3: Final local refinement - try small perturbations on best values
    if time.time() < deadline - 2 and best_values is not None:
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            
            idx = random.randint(0, n - 1)
            old_val = best_values[idx]
            new_val = old_val + random.choice([-2, -1, 1, 2, -3, 3, -5, 5])
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in best_values and new_val != old_val:
                continue
            
            trial_values = list(best_values)
            trial_values[idx] = new_val
            trial_edges = _greedy_edge_selection(trial_values, n, m)
            trial_score = _score(trial_values, trial_edges)
            
            if trial_score < best_score:
                best_score = trial_score
                best_values = trial_values
                best_edges = trial_edges
    
    # Ensure all values are Python ints
    best_values = [int(v) for v in best_values]
    # Ensure edges are lists of Python ints
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    # Verify constraints
    assert len(best_values) == n, f"Expected {n} values, got {len(best_values)}"
    assert len(set(best_values)) == n, "Values must be distinct"
    assert all(1 <= v <= 10000 for v in best_values), "Values must be in [1, 10000]"
    assert len(best_edges) == m, f"Expected {m} edges, got {len(best_edges)}"
    
    edge_set = set()
    for i, j in best_edges:
        assert i != j, "No loops allowed"
        assert 0 <= i < n and 0 <= j < n, "Edge indices out of range"
        e = tuple(sorted((i, j)))
        assert e not in edge_set, "Duplicate edge"
        edge_set.add(e)
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
