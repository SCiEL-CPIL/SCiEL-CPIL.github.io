"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    """Compute the objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def _max_bipartite_matching(adj):
    """Hungarian-style bipartite matching using augmenting paths.
    adj: list of lists, adj[u] = list of v's connected to u (left->right)
    Returns size of maximum matching."""
    n_left = len(adj)
    match_r = [-1] * n_left  # match_r[v] = u that matches to v, or -1
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    result = 0
    for u in range(n_left):
        visited = [False] * n_left
        if dfs(u, visited):
            result += 1
    return result

def _find_best_edges_for_k(values, k, all_edges, m):
    """Find m edges such that #distinct sums <= k AND #distinct products <= k.
    Uses bipartite matching: left=possible sum values, right=possible product values.
    Returns list of edges if found, else None."""
    n = len(values)
    # Precompute sum and product for each edge
    edge_sum = {}
    edge_prod = {}
    for e in all_edges:
        i, j = e
        edge_sum[e] = values[i] + values[j]
        edge_prod[e] = values[i] * values[j]
    
    # Get unique sums and products
    all_sums = set(edge_sum.values())
    all_prods = set(edge_prod.values())
    
    if len(all_sums) > k or len(all_prods) > k:
        return None
    
    # Build bipartite graph: left nodes = sums (0..len-1), right nodes = prods (0..len-1)
    # Edge exists between sum s and prod p if there's a graph edge with that sum and prod
    sum_to_idx = {s: i for i, s in enumerate(all_sums)}
    prod_to_idx = {p: i for i, p in enumerate(all_prods)}
    
    # adj[u] = list of v indices where u is sum index, v is prod index
    adj = [[] for _ in range(len(all_sums))]
    edge_lookup = {}  # (sum_idx, prod_idx) -> edge
    
    for e in all_edges:
        s = edge_sum[e]
        p = edge_prod[e]
        si = sum_to_idx[s]
        pi = prod_to_idx[p]
        adj[si].append(pi)
        edge_lookup[(si, pi)] = e
    
    # Find maximum matching
    n_left = len(all_sums)
    match_r = [-1] * n_left  # match_r[prod_idx] = sum_idx
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    matching = 0
    for u in range(n_left):
        visited = [False] * n_left
        if dfs(u, visited):
            matching += 1
    
    # We need at least m matched edges
    if matching < m:
        return None
    
    # Extract matched edges
    selected_edges = []
    for pi in range(n_left):
        si = match_r[pi]
        if si != -1:
            e = edge_lookup.get((si, pi))
            if e is not None:
                selected_edges.append(e)
    
    if len(selected_edges) < m:
        return None
    
    # Take first m edges (they're distinct since each (sum,prod) pair maps to one edge)
    return [list(e) for e in selected_edges[:m]]

def _structured_values(base, spread):
    """Generate structured values with good collision properties."""
    n = 14
    vals = []
    for i in range(n):
        v = base + i * spread
        vals.append(v)
    return vals

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    all_edges = list(combinations(range(n), 2))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    def update_best(vals, edges):
        nonlocal best_score, best_values, best_edges
        score = _raw_value(vals, edges)
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [e[:] for e in edges]
            return score
        return score
    
    # Phase 1: Try structured value sets with bipartite matching edge selection
    base_configs = [
        (1, 50), (1, 100), (1, 200), (2, 50), (3, 40),
        (1, 30), (1, 70), (1, 150), (2, 80), (5, 30),
        (1, 25), (1, 40), (1, 60), (1, 80), (1, 120),
        (2, 40), (3, 30), (4, 25), (1, 35), (1, 90),
    ]
    
    for base, spread in base_configs:
        if time.time() > deadline - 10:
            break
        vals = _structured_values(base, spread)
        if max(vals) > maximum:
            continue
        if len(set(vals)) != n:
            continue
        
        # Try different k values from small to large
        for k in range(5, 20):
            if time.time() > deadline - 10:
                break
            edges = _find_best_edges_for_k(vals, k, all_edges, m)
            if edges is not None:
                score = _raw_value(vals, edges)
                if score < best_score:
                    best_score = score
                    best_values = vals[:]
                    best_edges = [e[:] for e in edges]
                if score <= k:
                    break  # Found optimal for this k
    
    # Phase 2: Try geometric sequences
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        if time.time() > deadline - 10:
            break
        vals = []
        v = 1
        for i in range(n):
            vals.append(int(v))
            v *= ratio
        if max(vals) > maximum or len(set(vals)) != n:
            continue
        
        for k in range(5, 20):
            if time.time() > deadline - 10:
                break
            edges = _find_best_edges_for_k(vals, k, all_edges, m)
            if edges is not None:
                score = _raw_value(vals, edges)
                if score < best_score:
                    best_score = score
                    best_values = vals[:]
                    best_edges = [e[:] for e in edges]
                if score <= k:
                    break
    
    # Phase 3: Random value sets with bipartite matching
    random.seed(42)
    for trial in range(300):
        if time.time() > deadline - 10:
            break
        vals = random.sample(range(1, maximum + 1), n)
        
        # Try a few k values
        for k in [8, 10, 12, 15, 20]:
            if time.time() > deadline - 10:
                break
            edges = _find_best_edges_for_k(vals, k, all_edges, m)
            if edges is not None:
                score = _raw_value(vals, edges)
                if score < best_score:
                    best_score = score
                    best_values = vals[:]
                    best_edges = [e[:] for e in edges]
    
    # Phase 4: Local search on best solution found
    if best_values is not None and time.time() < deadline - 5:
        current_values = best_values[:]
        current_edges = [e[:] for e in best_edges]
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for trial in range(200):
                if time.time() > deadline - 5:
                    break
                # Try modifying a single value
                idx = random.randint(0, n - 1)
                old_val = current_values[idx]
                new_val = random.randint(1, maximum)
                if new_val in current_values:
                    continue
                current_values[idx] = new_val
                new_score = _raw_value(current_values, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_values[idx] = old_val
            
            # Try edge swaps
            for trial in range(100):
                if time.time() > deadline - 5:
                    break
                edge_idx = random.randint(0, m - 1)
                old_edge = tuple(current_edges[edge_idx])
                new_edge = random.choice(all_edges)
                if new_edge in set(tuple(e) for e in current_edges):
                    continue
                current_edges[edge_idx] = list(new_edge)
                new_score = _raw_value(current_values, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_edges[edge_idx] = list(old_edge)
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values[:]
                best_edges = [e[:] for e in current_edges]
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_edges[:m]]
    
    result = {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
    
    return result
# EVOLVE-BLOCK-END
