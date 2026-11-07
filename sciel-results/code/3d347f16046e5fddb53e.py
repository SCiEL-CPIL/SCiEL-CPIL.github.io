"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s
    
    n = P['n']  # 14
    m = P['m']  # 35
    max_val = P['maximum']  # 10000
    all_pairs = list(combinations(range(n), 2))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    def _validate(values, edges):
        """Check if values and edges are valid."""
        if len(values) != n:
            return False
        if len(set(values)) != n:
            return False
        if any(not isinstance(v, int) or v < 1 or v > max_val for v in values):
            return False
        if len(edges) != m:
            return False
        edge_set = set()
        for u, v in edges:
            if not isinstance(u, int) or not isinstance(v, int):
                return False
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            t = (min(u, v), max(u, v))
            if t in edge_set:
                return False
            edge_set.add(t)
        return True
    
    def _try_update(values, edges):
        """Try to update best if this solution is better."""
        nonlocal best_score, best_values, best_edges
        if not _validate(values, edges):
            return
        score = _score(values, edges)
        if score < best_score:
            best_score = score
            best_values = list(values)
            best_edges = [list(e) for e in edges]
    
    def _generate_structured_values():
        """Generate values with shared factors to reduce distinct products."""
        # Strategy: pick a few base factors and create values from them
        # This can create many product collisions
        strategies = []
        
        # Strategy 1: Powers of small primes
        for base in [2, 3, 5, 7, 11]:
            vals = []
            for i in range(1, 15):
                v = base ** i
                if v <= max_val:
                    vals.append(v)
            if len(vals) >= n:
                strategies.append(vals[:n])
        
        # Strategy 2: Multiples of a common factor
        for factor in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
            vals = []
            for i in range(1, 15):
                v = factor * i
                if v <= max_val:
                    vals.append(v)
            if len(vals) >= n:
                strategies.append(vals[:n])
        
        # Strategy 3: Products of two small numbers
        for a in range(2, 12):
            for b in range(a + 1, 20):
                vals = []
                for i in range(1, 15):
                    v = a * b * i
                    if v <= max_val:
                        vals.append(v)
                if len(vals) >= n:
                    strategies.append(vals[:n])
        
        # Strategy 4: Mixed - some multiples, some powers
        for factor in [2, 3, 5]:
            vals = []
            for i in range(1, 8):
                v = factor * i
                if v <= max_val:
                    vals.append(v)
            for i in range(2, 8):
                v = factor ** i
                if v <= max_val and v not in vals:
                    vals.append(v)
            if len(vals) >= n:
                strategies.append(vals[:n])
        
        # Strategy 5: Arithmetic progressions
        for start in range(1, 20):
            for step in range(1, 10):
                vals = [start + step * i for i in range(n)]
                if all(1 <= v <= max_val for v in vals):
                    strategies.append(vals)
        
        # Strategy 6: Random with bias toward small numbers
        for _ in range(10):
            vals = sorted(random.sample(range(1, 200), n))
            strategies.append(vals)
        
        # Strategy 7: Random with shared small factors
        for _ in range(10):
            base_factors = random.sample([2, 3, 5, 7, 11, 13], 3)
            vals = []
            for i in range(n):
                f = random.choice(base_factors)
                v = f * random.randint(1, 500)
                if v <= max_val and v not in vals:
                    vals.append(v)
            if len(vals) < n:
                while len(vals) < n:
                    v = random.randint(1, max_val)
                    if v not in vals:
                        vals.append(v)
            strategies.append(vals[:n])
        
        # Strategy 8: Pure random
        for _ in range(5):
            strategies.append(random.sample(range(1, max_val + 1), n))
        
        if not strategies:
            strategies.append(list(range(1, n + 1)))
        
        return random.choice(strategies)
    
    def _greedy_edge_selection(values, num_restarts=3):
        """Select edges greedily to minimize distinct sums and products."""
        # Precompute sums and products for all pairs
        pair_sum = {}
        pair_prod = {}
        for u, v in all_pairs:
            a, b = values[u], values[v]
            pair_sum[(u, v)] = a + b
            pair_prod[(u, v)] = a * b
        
        best_edges = None
        best_local_score = float('inf')
        
        for _ in range(num_restarts):
            selected = []
            selected_set = set()
            current_sums = set()
            current_prods = set()
            
            # Shuffle pairs for randomization
            shuffled_pairs = all_pairs[:]
            random.shuffle(shuffled_pairs)
            
            for u, v in shuffled_pairs:
                if len(selected) >= m:
                    break
                key = (u, v)
                if key in selected_set:
                    continue
                
                s = pair_sum[key]
                p = pair_prod[key]
                
                # Count how many new distinct values this edge adds
                new_sums = 1 if s not in current_sums else 0
                new_prods = 1 if p not in current_prods else 0
                
                # Prefer edges that add fewer new distinct values
                # Use a weighted score
                cost = new_sums + new_prods
                
                if cost == 0:
                    # Always accept if no new distinct values
                    selected.append([u, v])
                    selected_set.add(key)
                    current_sums.add(s)
                    current_prods.add(p)
                elif len(selected) < m:
                    # Accept with some probability based on cost
                    # This allows some exploration
                    if cost <= 1 or random.random() < 0.3:
                        selected.append([u, v])
                        selected_set.add(key)
                        current_sums.add(s)
                        current_prods.add(p)
            
            # If we didn't get enough edges, fill with remaining
            if len(selected) < m:
                for u, v in all_pairs:
                    if len(selected) >= m:
                        break
                    key = (u, v)
                    if key not in selected_set:
                        selected.append([u, v])
                        selected_set.add(key)
                        current_sums.add(pair_sum[key])
                        current_prods.add(pair_prod[key])
            
            local_score = max(len(current_sums), len(current_prods))
            if local_score < best_local_score:
                best_local_score = local_score
                best_edges = selected[:]
        
        return best_edges, best_local_score
    
    def _local_search_values(values, edges, max_iters=200):
        """Local search to improve values for fixed edges."""
        current_score = _score(values, edges)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(n):
                old_val = values[i]
                # Try a few random new values
                for _ in range(20):
                    new_val = random.randint(1, max_val)
                    if new_val == old_val or new_val in values:
                        continue
                    values[i] = new_val
                    new_score = _score(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        values[i] = old_val
                        break
                values[i] = old_val
            
            # Try swaps
            for i in range(n - 1):
                for j in range(i + 1, n):
                    values[i], values[j] = values[j], values[i]
                    new_score = _score(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        values[i], values[j] = values[j], values[i]
        
        return values, current_score
    
    def _local_search_edges(values, edges, max_iters=100):
        """Local search to improve edges for fixed values."""
        current_score = _score(values, edges)
        improved = True
        iters = 0
        edge_set = set(tuple(e) for e in edges)
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for idx in range(m):
                e_remove = tuple(edges[idx])
                for e_add in all_pairs:
                    if e_add in edge_set:
                        continue
                    new_edges = edges[:idx] + edges[idx+1:] + [list(e_add)]
                    new_score = _score(values, new_edges)
                    if new_score < current_score:
                        edges = new_edges
                        edge_set.discard(e_remove)
                        edge_set.add(e_add)
                        current_score = new_score
                        improved = True
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return edges, current_score
    
    # Initial solution
    initial_values = list(range(1, n + 1))
    initial_edges = [list(e) for e in all_pairs[:m]]
    _try_update(initial_values, initial_edges)
    
    # Main search loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Generate new candidate values
        if iteration % 4 == 0:
            cur_values = _generate_structured_values()
        else:
            if best_values is not None:
                # Perturb best values
                cur_values = best_values[:]
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, n - 1)
                    old = cur_values[i]
                    new = random.randint(1, max_val)
                    if new != old and new not in cur_values:
                        cur_values[i] = new
            else:
                cur_values = _generate_structured_values()
        
        # Get initial edges via greedy
        cur_edges, _ = _greedy_edge_selection(cur_values, num_restarts=3)
        
        # Local search on values
        cur_values, _ = _local_search_values(cur_values, cur_edges, max_iters=100)
        
        if time.time() > deadline:
            break
        
        # Re-optimize edges
        cur_edges, _ = _local_search_edges(cur_values, cur_edges, max_iters=50)
        
        if time.time() > deadline:
            break
        
        # Try to update best
        _try_update(cur_values, cur_edges)
        
        # Early stopping if we've found a very good solution
        if best_score <= 14:
            break
    
    # Final validation and return
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_pairs[:m]]
    
    assert _validate(best_values, best_edges)
    
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
