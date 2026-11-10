"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _score(values, edges):
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _all_edges(n):
    return list(combinations(range(n), 2))

def _greedy_edge_selection(values, n, m):
    """Greedy: at each step pick the edge that minimizes max(distinct sums, distinct prods)."""
    all_e = _all_edges(n)
    edge_props = []
    for i, j in all_e:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_props.append((i, j, s, p))
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
            new_sums = 0 if sum_counts[s] > 0 else 1
            new_prods = 0 if prod_counts[p] > 0 else 1
            cur_s = len(sum_counts)
            cur_p = len(prod_counts)
            metric = max(cur_s + new_sums, cur_p + new_prods)
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

def _dp_select_edges(values, n, m, max_distinct):
    """
    Try to select m edges such that distinct sums <= max_distinct AND distinct prods <= max_distinct.
    Uses DP over subsets of distinct sums, tracking distinct products.
    Returns (edges, distinct_sums, distinct_prods) or None.
    """
    all_e = _all_edges(n)
    edge_props = []
    for i, j in all_e:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_props.append((i, j, s, p))
    # Deduplicate edges by (s,p) - keep one representative per (s,p) pair
    # Actually we need distinct edges, but same (s,p) is fine.
    # DP state: (sum_mask, prod_count, edge_count) -> feasible
    # But sum_mask is 2^90 which is too big. Instead, use a different approach.
    # Greedy with lookahead: try to find m edges with limited distinct sums.
    # For each possible set of distinct sums (up to max_distinct), check if we can pick m edges.
    # This is hard. Instead, use a heuristic: group edges by sum, pick edges from few sum-groups.
    
    # Strategy: sort edges by sum, try to pick m edges from the fewest distinct sums
    # such that the number of distinct products is also <= max_distinct.
    
    # Group edges by sum
    sum_groups = defaultdict(list)
    for idx, (i, j, s, p) in enumerate(edge_props):
        sum_groups[s].append((idx, i, j, s, p))
    
    # Try increasing number of distinct sums
    for target_sums in range(1, max_distinct + 1):
        # Try all combinations of target_sums distinct sum-values
        # Too many. Instead, greedily pick sum-groups.
        # Sort sum-groups by size (largest first) to maximize edge availability
        sorted_sums = sorted(sum_groups.keys(), key=lambda s: -len(sum_groups[s]))
        # Try picking the first target_sums sum-groups
        chosen_sums = sorted_sums[:target_sums]
        chosen_set = set(chosen_sums)
        # Collect all edges with these sums
        candidate_edges = []
        for s in chosen_sums:
            for idx, i, j, ss, p in sum_groups[s]:
                candidate_edges.append((idx, i, j, ss, p))
        if len(candidate_edges) < m:
            continue
        # Now pick m edges from candidate_edges to minimize distinct products
        # Greedy: pick edges whose product is already seen
        prod_counts = defaultdict(int)
        selected = []
        selected_set = set()
        # First pass: pick edges with most common products
        prod_freq = defaultdict(int)
        for idx, i, j, ss, p in candidate_edges:
            prod_freq[p] += 1
        # Sort candidate edges by product frequency (descending)
        candidate_edges.sort(key=lambda x: -prod_freq[x[4]])
        for idx, i, j, ss, p in candidate_edges:
            if len(selected) >= m:
                break
            e_key = (i, j)
            if e_key in selected_set:
                continue
            selected.append([i, j])
            selected_set.add(e_key)
            prod_counts[p] += 1
        if len(selected) < m:
            continue
        if len(prod_counts) <= max_distinct:
            return selected, len(chosen_set), len(prod_counts)
    return None

def _geometric_values(base, ratio, n, max_val=10000):
    """Generate geometric progression values: base * ratio^i."""
    values = []
    for i in range(n):
        v = base * (ratio ** i)
        if v > max_val:
            return None
        values.append(v)
    if len(set(values)) != n:
        return None
    return values

def _structured_initial_values(n, max_val=10000):
    """Generate structured initial value sets."""
    sets = []
    # Geometric progressions
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        for ratio in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            vals = _geometric_values(base, ratio, n, max_val)
            if vals:
                sets.append(vals)
    # Arithmetic progressions
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]:
        for start in range(1, min(20, max_val // n)):
            vals = [start + base * i for i in range(n)]
            if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
                sets.append(vals)
    # Mixed: geometric with small perturbations
    for base in [2, 3, 4, 5, 6, 7, 8, 10]:
        for ratio in [2, 3, 4, 5, 6, 8, 10]:
            vals = []
            for i in range(n):
                v = base * (ratio ** i)
                if v > max_val:
                    break
                vals.append(v)
            if len(vals) == n and len(set(vals)) == n:
                sets.append(vals)
    # Deduplicate
    unique_sets = []
    seen = set()
    for s in sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(list(s))
    return unique_sets

def _simulated_annealing(values, n, m, max_time=10, deadline=None):
    """Simulated annealing on values with edge re-selection."""
    start_time = time.time()
    best_values = list(values)
    best_edges = _greedy_edge_selection(best_values, n, m)
    best_score = _score(best_values, best_edges)
    
    current_values = list(values)
    current_edges = _greedy_edge_selection(current_values, n, m)
    current_score = _score(current_values, current_edges)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.998
    no_improve = 0
    max_no_improve = 80
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        if deadline is not None and time.time() > deadline - 3:
            break
        if no_improve > max_no_improve:
            break
        
        idx = random.randint(0, n - 1)
        old_val = current_values[idx]
        
        # Generate new value: small perturbation or random
        if random.random() < 0.7:
            move_size = random.choice([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])
            new_val = old_val + random.randint(-move_size, move_size)
        else:
            new_val = random.randint(1, 10000)
        
        if new_val < 1 or new_val > 10000:
            new_val = random.randint(1, 10000)
        
        if new_val in current_values and new_val != old_val:
            T *= cooling
            continue
        
        trial_values = list(current_values)
        trial_values[idx] = new_val
        trial_edges = _greedy_edge_selection(trial_values, n, m)
        trial_score = _score(trial_values, trial_edges)
        
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

def _local_refinement(values, edges, n, m, max_time=5, deadline=None):
    """Try small perturbations on values and edge swaps."""
    start_time = time.time()
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    best_score = _score(best_values, best_edges)
    
    for _ in range(500):
        if time.time() - start_time > max_time:
            break
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Try perturbing a value
        idx = random.randint(0, n - 1)
        old_val = best_values[idx]
        new_val = old_val + random.choice([-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20])
        
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
    
    return best_values, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = 14
    m = 35
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Try structured value sets
    value_sets = _structured_initial_values(n, max_val=10000)
    
    for vs in value_sets:
        if time.time() > deadline - 20:
            break
        edges = _greedy_edge_selection(vs, n, m)
        score = _score(vs, edges)
        if score < best_score:
            best_score = score
            best_values = list(vs)
            best_edges = [list(e) for e in edges]
    
    # Also try DP-based edge selection for best value sets
    for vs in value_sets[:10]:
        if time.time() > deadline - 15:
            break
        result = _dp_select_edges(vs, n, m, max_distinct=15)
        if result:
            edges, ds, dp = result
            score = max(ds, dp)
            if score < best_score:
                best_score = score
                best_values = list(vs)
                best_edges = [list(e) for e in edges]
    
    # Phase 2: Simulated annealing from best and diverse starts
    anneal_starts = [best_values]
    for vs in value_sets[:5]:
        if vs != best_values:
            anneal_starts.append(vs)
    
    for start_vals in anneal_starts:
        if time.time() > deadline - 8:
            break
        remaining = deadline - time.time() - 3
        if remaining < 2:
            break
        vals, edges, score = _simulated_annealing(start_vals, n, m, max_time=min(remaining, 12), deadline=deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Phase 3: Final local refinement
    if time.time() < deadline - 3 and best_values is not None:
        vals, edges, score = _local_refinement(best_values, best_edges, n, m, max_time=5, deadline=deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Ensure Python int types
    best_values = [int(v) for v in best_values]
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
