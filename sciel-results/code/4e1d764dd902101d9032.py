import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _optimize_edges(values, max_iterations=500):
    """Optimize edges for given values using greedy init and local search."""
    n = len(values)
    num_edges = 35
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_props = {}
    for i, j in all_edges:
        edge_props[(i, j)] = (values[i] + values[j], values[i] * values[j])
        
    # Greedy initialization
    # We want to pick edges that reuse sums and products
    # Start with empty sets
    current_edges = []
    current_sums = set()
    current_products = set()
    remaining = set(all_edges)
    
    # To make greedy smarter, we can look ahead or just pick min cost
    # Cost = 0 if both exist, 1 if one exists, 2 if neither exists
    for _ in range(num_edges):
        best_edge = None
        best_cost = 3
        # Randomize tie-breaking to escape local minima in greedy phase
        candidates = list(remaining)
        random.shuffle(candidates)
        for e in candidates:
            s, p = edge_props[e]
            cost = (0 if s in current_sums else 1) + (0 if p in current_products else 1)
            if cost < best_cost:
                best_cost = cost
                best_edge = e
                if cost == 0: # Can't get better
                    break
            elif cost == best_cost:
                # Tie-breaker: prefer edges that add more common factors? 
                # Or just keep the first one found in shuffled list
                pass
        
        if best_edge is None:
            best_edge = candidates[0]
            
        current_edges.append(best_edge)
        s, p = edge_props[best_edge]
        current_sums.add(s)
        current_products.add(p)
        remaining.discard(best_edge)
        
    # Local Search
    current_score = _score(values, current_edges)
    
    for _ in range(max_iterations):
        if not remaining:
            break
        # Pick a random edge to remove
        out_idx = random.randint(0, len(current_edges) - 1)
        e_out = current_edges[out_idx]
        s_out, p_out = edge_props[e_out]
        
        # Pick a random edge to add from remaining
        e_in = random.choice(list(remaining))
        s_in, p_in = edge_props[e_in]
        
        # Calculate new score efficiently
        # Remove e_out from sets
        # This is tricky because other edges might share the same sum/product.
        # So we must recount or use counters.
        
        # For N=14, recounting is fast.
        new_edges = current_edges[:]
        new_edges[out_idx] = e_in
        
        new_score = _score(values, new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            remaining.add(e_out)
            remaining.discard(e_in)
        elif new_score == current_score and random.random() < 0.1:
            # Accept equal score to explore
            current_edges = new_edges
            remaining.add(e_out)
            remaining.discard(e_in)
            
    return current_edges, current_score

def construct():
    """Construct values and edges minimizing max(distinct sums, distinct products)."""
    random.seed(42)
    n = 14
    num_edges = 35
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Initial candidate sets
    candidates = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ]
    
    # Add some random small integer sets
    for _ in range(10):
        vals = random.sample(range(1, 50), n)
        candidates.append(vals)
        
    # Optimize edges for each candidate
    for vals in candidates:
        edges, score = _optimize_edges(vals)
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [list(e) for e in edges]
            
    # Global search on values
    # We will perform a random walk on the value space
    # Mutate one value at a time
    
    current_values = best_values[:]
    current_edges = best_edges[:]
    current_score = best_score
    
    for iter in range(200): # Limit iterations to stay within time
        # Pick a random index to mutate
        idx = random.randint(0, n - 1)
        old_val = current_values[idx]
        
        # Generate a new value
        # Strategy: Try small values first, then random
        if random.random() < 0.5:
            new_val = random.randint(1, 30)
        else:
            new_val = random.randint(1, 100)
            
        # Ensure distinctness
        if new_val in current_values and current_values.count(new_val) > 1:
            continue
        if new_val == old_val:
            continue
            
        # Check if new_val creates duplicates
        # We need 14 distinct values
        temp_vals = current_values[:]
        # Remove old_val
        temp_vals[idx] = new_val
        if len(set(temp_vals)) != n:
            continue
            
        current_values[idx] = new_val
        
        # Re-optimize edges for this new value set
        edges, score = _optimize_edges(current_values, max_iterations=100) # Fewer iterations for speed
        
        if score < current_score:
            current_score = score
            current_edges = edges
            if score < best_score:
                best_score = score
                best_values = current_values[:]
                best_edges = [list(e) for e in edges]
        else:
            # Revert
            current_values[idx] = old_val
            
    return {'values': best_values, 'edges': best_edges}
