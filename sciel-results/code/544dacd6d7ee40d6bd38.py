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
    deadline = start_time + 82.0  # safety margin before 90s
    
    n = P['n']  # 14
    m = P['m']  # 35
    max_val = P['maximum']  # 10000
    
    all_pairs = list(combinations(range(n), 2))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Initialize with a simple valid solution
    initial_values = list(range(1, n + 1))
    initial_edges = [list(e) for e in all_pairs[:m]]
    best_score = _score(initial_values, initial_edges)
    best_values = initial_values.copy()
    best_edges = initial_edges.copy()
    
    def random_values():
        """Generate 14 distinct positive integers <= 10000."""
        return random.sample(range(1, max_val + 1), n)
    
    def random_edges():
        """Generate 35 random edges."""
        return [list(e) for e in random.sample(all_pairs, m)]
    
    def structured_values():
        """Generate values with shared factors to create product collisions."""
        # Use multiples of small numbers to create product overlaps
        base = random.choice([2, 3, 4, 5, 6, 8, 10, 12])
        vals = set()
        attempts = 0
        while len(vals) < n and attempts < 1000:
            v = base * random.randint(1, max_val // base)
            if 1 <= v <= max_val:
                vals.add(v)
            attempts += 1
        # Fill remaining with random values
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)
    
    def power_values():
        """Generate values as powers of 2 times a common factor."""
        base = random.choice([2, 3, 5])
        vals = set()
        for i in range(n):
            v = base * (2 ** i)
            if v <= max_val:
                vals.add(v)
            else:
                vals.add(base * random.randint(1, max_val // base))
        # Ensure distinctness and fill if needed
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)
    
    def local_search_values(values, edges, max_iter=200):
        """Local search to improve values for fixed edges."""
        current_score = _score(values, edges)
        values_set = set(values)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try adjusting individual values
            for i in range(n):
                old_val = values[i]
                values_set.discard(old_val)
                
                # Try several candidate values
                candidates = set()
                # Random candidates
                for _ in range(20):
                    candidates.add(random.randint(1, max_val))
                # Neighbors of existing values
                for v in values:
                    for delta in [-1, 1, -2, 2, -3, 3]:
                        nv = v + delta
                        if 1 <= nv <= max_val:
                            candidates.add(nv)
                # Multiples of common factors
                for factor in [2, 3, 4, 5, 6, 8, 10]:
                    candidates.add(factor * random.randint(1, max_val // factor))
                
                for new_val in candidates:
                    if new_val == old_val or new_val in values_set:
                        continue
                    values[i] = new_val
                    values_set.add(new_val)
                    new_score = _score(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        values[i] = old_val
                        values_set.discard(new_val)
                
                values[i] = old_val
                values_set.add(old_val)
            
            # Try pairwise swaps
            for i in range(n - 1):
                for j in range(i + 1, n):
                    values[i], values[j] = values[j], values[i]
                    new_score = _score(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        values[i], values[j] = values[j], values[i]
                    
                    if time.time() > deadline:
                        return values, current_score
        
        return values, current_score
    
    def local_search_edges(values, edges, max_iter=50):
        """Local search to improve edges for fixed values."""
        current_score = _score(values, edges)
        edge_set = set(tuple(e) for e in edges)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try removing one edge and adding another
            for idx in range(m):
                if time.time() > deadline:
                    return edges, current_score
                e_remove = tuple(edges[idx])
                
                # Try several candidate edges to add
                candidate_edges = random.sample([e for e in all_pairs if e not in edge_set], min(50, len(all_pairs) - m))
                
                for e_add in candidate_edges:
                    new_edges = edges[:idx] + edges[idx+1:] + [list(e_add)]
                    new_score = _score(values, new_edges)
                    if new_score < current_score:
                        edges = new_edges
                        edge_set.discard(e_remove)
                        edge_set.add(e_add)
                        current_score = new_score
                        improved = True
                        break
        
        return edges, current_score
    
    # Run several iterations with different strategies
    strategies = [random_values, random_values, structured_values, power_values, random_values]
    
    for iteration in range(30):
        if time.time() > deadline:
            break
        
        # Choose strategy
        if iteration % 5 == 0:
            cur_values = strategies[iteration % len(strategies)]()
        else:
            cur_values = best_values.copy()
        
        cur_edges = random_edges() if iteration % 3 == 0 else best_edges.copy()
        
        # Alternate between optimizing values and edges
        for _ in range(8):
            if time.time() > deadline:
                break
            
            # Optimize values for fixed edges
            cur_values, vscore = local_search_values(cur_values, cur_edges, max_iter=100)
            
            if time.time() > deadline:
                break
            
            # Optimize edges for fixed values
            cur_edges, escore = local_search_edges(cur_values, cur_edges, max_iter=30)
            
            if time.time() > deadline:
                break
            
            cur_score = _score(cur_values, cur_edges)
            if cur_score < best_score:
                best_score = cur_score
                best_values = cur_values.copy()
                best_edges = cur_edges.copy()
        
        # Early stopping if we've found a very good solution
        if best_score <= 14:
            break
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_tuples = set()
    for u, v in best_edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n and 0 <= v < n
        assert u != v
        t = (min(u, v), max(u, v))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
