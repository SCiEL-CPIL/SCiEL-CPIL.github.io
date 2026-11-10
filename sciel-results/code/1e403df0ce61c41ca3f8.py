import time
import random
import heapq
from itertools import combinations

# EVOLVE-BLOCK-START

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_edge_select(values, m, deadline):
    n = len(values)
    # Precompute all pair sums and products
    all_pairs = list(combinations(range(n), 2))
    pair_sum = {}
    pair_prod = {}
    for i, j in all_pairs:
        pair_sum[(i, j)] = values[i] + values[j]
        pair_prod[(i, j)] = values[i] * values[j]
    
    # Frequency of each sum and product
    from collections import Counter
    sum_freq = Counter(pair_sum.values())
    prod_freq = Counter(pair_prod.values())
    
    # Score an edge: prefer high sum_freq and high prod_freq
    # Greedy: add edge minimizing increase in max(distinct_sums, distinct_products)
    selected = []
    selected_set = set()
    cur_sums = set()
    cur_prods = set()
    
    # Priority: edges sorted by (sum_freq, prod_freq) descending as tiebreaker
    # But we need to evaluate incrementally.
    # Use a simple greedy: at each step, pick edge that minimizes new max.
    # To speed up, pre-rank edges by a heuristic score.
    
    def edge_score(e):
        i, j = e
        sf = sum_freq[pair_sum[e]]
        pf = prod_freq[pair_prod[e]]
        return (sf + pf, sf, pf)
    
    ranked = sorted(all_pairs, key=edge_score, reverse=True)
    
    for e in ranked:
        if len(selected) >= m:
            break
        if time.time() > deadline:
            break
        i, j = e
        new_sums = cur_sums | {pair_sum[e]}
        new_prods = cur_prods | {pair_prod[e]}
        new_val = max(len(new_sums), len(new_prods))
        # Accept if it doesn't worsen beyond current best-so-far for this selection
        selected.append(e)
        selected_set.add(e)
        cur_sums = new_sums
        cur_prods = new_prods
    
    return selected

def _local_search_edges(values, edges, deadline, max_iters=200):
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    edge_set = set(tuple(e) for e in edges)
    cur_val = _raw_value(values, edges)
    
    # Precompute
    from collections import Counter
    pair_sum = {}
    pair_prod = {}
    for i, j in all_pairs:
        pair_sum[(i, j)] = values[i] + values[j]
        pair_prod[(i, j)] = values[i] * values[j]
    
    best_val = cur_val
    best_edges = list(edges)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try swapping one edge in with one edge out
        for e_in in all_pairs:
            if e_in in edge_set:
                continue
            for e_out in list(edge_set):
                new_set = (edge_set - {e_out}) | {e_in}
                # Quick compute
                sums = set()
                prods = set()
                for i, j in new_set:
                    sums.add(pair_sum[(i, j)])
                    prods.add(pair_prod[(i, j)])
                v = max(len(sums), len(prods))
                if v < best_val:
                    best_val = v
                    best_edges = [list(e) for e in new_set]
                    edge_set = new_set
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_edges, best_val

def _gen_symmetric_values(n, maxv, rng):
    # Generate n distinct positive ints <= maxv with additive symmetry
    # Pick a center, create pairs summing to 2*center
    center = rng.randint(20, maxv // 2)
    vals = []
    used = set()
    half = n // 2
    for _ in range(half):
        a = rng.randint(1, center - 1)
        b = 2 * center - a
        if a == b or a in used or b in used or a <= 0 or b <= 0 or a > maxv or b > maxv:
            continue
        vals.append(a)
        vals.append(b)
        used.add(a)
        used.add(b)
    if n % 2 == 1:
        vals.append(center)
        used.add(center)
    # Fill remaining randomly
    while len(vals) < n:
        x = rng.randint(1, maxv)
        if x not in used:
            vals.append(x)
            used.add(x)
    rng.shuffle(vals)
    return vals[:n]

def construct():
    import time
    start = time.time()
    deadline = start + 78.0  # safety margin
    rng = random.Random(42)
    
    best_val = float('inf')
    best_result = None
    
    n = 14
    m = 35
    maxv = 10000
    
    # Try several value configurations
    value_configs = []
    
    # Config 1: symmetric
    value_configs.append(_gen_symmetric_values(n, maxv, rng))
    
    # Config 2: arithmetic progression
    value_configs.append(list(range(1, n + 1)))
    
    # Config 3: random
    for _ in range(5):
        vals = rng.sample(range(1, maxv + 1), n)
        value_configs.append(vals)
    
    # Config 4: powers and multiples for product collisions
    base_vals = [2**i for i in range(4)] + [3**i for i in range(4)] + [6, 12, 24, 48, 96, 192]
    base_vals = [v for v in base_vals if v <= maxv]
    if len(base_vals) >= n:
        value_configs.append(base_vals[:n])
    
    for vals in value_configs:
        if time.time() > deadline:
            break
        # Ensure distinct and valid
        if len(set(vals)) != n or any(v <= 0 or v > maxv for v in vals):
            continue
        
        # Greedy edge selection
        edges = _greedy_edge_select(vals, m, deadline)
        if len(edges) != m:
            continue
        
        # Local search
        edges, val = _local_search_edges(vals, edges, deadline, max_iters=100)
        
        if val < best_val:
            best_val = val
            best_result = {'values': [int(v) for v in vals], 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result

# EVOLVE-BLOCK-END
