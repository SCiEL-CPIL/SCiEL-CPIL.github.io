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

def _count_distinct(values, edges):
    """Return (num_distinct_sums, num_distinct_products)."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return len(sums), len(products)

def _generate_values_with_common_factor():
    """Generate 14 distinct positive integers <= 10000 with shared structure."""
    # Strategy: use multiples of a base number to create product collisions
    # If values are k*a, k*b, then products are k^2*a*b
    # Using small multipliers with a common base creates many collisions
    
    base = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 28, 30])
    # Generate 14 distinct multipliers
    max_mult = 10000 // base
    if max_mult < 14:
        base = 1
        max_mult = 10000
    multipliers = random.sample(range(1, max_mult + 1), 14)
    values = [base * m for m in multipliers]
    # Ensure all <= 10000 and distinct
    if len(set(values)) != 14 or any(v > 10000 for v in values):
        # Fallback: use small consecutive multiples
        values = [base * i for i in range(1, 15)]
        if any(v > 10000 for v in values):
            values = list(range(1, 15))
    return values

def _select_edges_for_product_collisions(values, n_edges=35):
    """Select edges that maximize product collisions."""
    all_edges = list(combinations(range(14), 2))
    
    # Compute product for each edge
    edge_products = {}
    for i, j in all_edges:
        p = values[i] * values[j]
        edge_products[(i, j)] = p
    
    # Group edges by product value
    from collections import defaultdict
    prod_to_edges = defaultdict(list)
    for (i, j), p in edge_products.items():
        prod_to_edges[p].append((i, j))
    
    # Sort products by frequency (most frequent first)
    sorted_products = sorted(prod_to_edges.keys(), key=lambda p: -len(prod_to_edges[p]))
    
    # Greedily select edges, preferring those with more product collisions
    selected = []
    selected_set = set()
    
    # First pass: add edges from most frequent products
    for p in sorted_products:
        for edge in prod_to_edges[p]:
            if edge not in selected_set and len(selected) < n_edges:
                selected.append(edge)
                selected_set.add(edge)
    
    # If we still need more edges, add remaining randomly
    if len(selected) < n_edges:
        remaining = [e for e in all_edges if e not in selected_set]
        random.shuffle(remaining)
        for edge in remaining:
            if len(selected) >= n_edges:
                break
            selected.append(edge)
            selected_set.add(edge)
    
    return [list(e) for e in selected[:n_edges]]

def _local_search(values, edges, time_limit=55):
    """Perform local search by mutating values and edges."""
    start_time = time.time()
    best_score = _score(values, edges)
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    
    all_edges = list(combinations(range(14), 2))
    
    # Precompute edge products and sums for current state
    def evaluate(vals, eds):
        s_set = set()
        p_set = set()
        for i, j in eds:
            s_set.add(vals[i] + vals[j])
            p_set.add(vals[i] * vals[j])
        return max(len(s_set), len(p_set))
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        current_score = _score(best_values, best_edges)
        
        # Try different mutation strategies
        strategy = random.random()
        
        if strategy < 0.4:
            # Mutate a single value
            new_values = list(best_values)
            idx = random.randint(0, 13)
            old_val = new_values[idx]
            # Try nearby values or values that share factors
            candidates = []
            # Try small perturbations
            for delta in [-3, -2, -1, 1, 2, 3, 5, 7, 10, -5, -7, -10]:
                nv = old_val + delta
                if 1 <= nv <= 10000 and nv != old_val:
                    candidates.append(nv)
            # Try values that are multiples of common factors
            for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
                for m in range(1, 10000 // base + 1):
                    nv = base * m
                    if 1 <= nv <= 10000 and nv != old_val and nv not in new_values:
                        candidates.append(nv)
                        if len(candidates) > 50:
                            break
                if len(candidates) > 50:
                    break
            random.shuffle(candidates)
            for nv in candidates[:20]:
                if nv in new_values:
                    continue
                new_values[idx] = nv
                s = evaluate(new_values, best_edges)
                if s < best_score:
                    best_score = s
                    best_values = new_values
                    no_improve = 0
                    break
                new_values[idx] = old_val
        elif strategy < 0.7:
            # Swap two values
            new_values = list(best_values)
            i, j = random.sample(range(14), 2)
            new_values[i], new_values[j] = new_values[j], new_values[i]
            s = evaluate(new_values, best_edges)
            if s < best_score:
                best_score = s
                best_values = new_values
                no_improve = 0
            else:
                no_improve += 1
        elif strategy < 0.9:
            # Replace an edge
            new_edges = [list(e) for e in best_edges]
            idx = random.randint(0, len(new_edges) - 1)
            old_edge = tuple(new_edges[idx])
            # Find a replacement edge that creates more collisions
            current_products = set()
            for ei, ej in new_edges:
                current_products.add(best_values[ei] * best_values[ej])
            
            best_new_edge = None
            best_new_score = best_score
            # Try random replacement edges
            for _ in range(30):
                new_e = random.choice(all_edges)
                if new_e == old_edge:
                    continue
                # Check if it's already in edges
                if any(tuple(e) == new_e for e in new_edges):
                    continue
                test_edges = [list(e) for e in new_edges]
                test_edges[idx] = list(new_e)
                s = evaluate(best_values, test_edges)
                if s < best_new_score:
                    best_new_score = s
                    best_new_edge = new_e
            if best_new_edge is not None:
                new_edges[idx] = list(best_new_edge)
                best_edges = new_edges
                best_score = best_new_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Replace multiple edges at once
            new_edges = [list(e) for e in best_edges]
            num_replace = random.randint(1, 3)
            for _ in range(num_replace):
                idx = random.randint(0, len(new_edges) - 1)
                new_e = random.choice(all_edges)
                # Remove duplicates
                if any(tuple(e) == new_e for e in new_edges if e is not new_edges[idx]):
                    continue
                new_edges[idx] = list(new_e)
            # Remove any duplicate edges
            seen = set()
            unique_edges = []
            for e in new_edges:
                t = tuple(sorted(e))
                if t not in seen:
                    seen.add(t)
                    unique_edges.append(e)
            if len(unique_edges) == 35:
                s = evaluate(best_values, unique_edges)
                if s < best_score:
                    best_score = s
                    best_edges = unique_edges
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
    
    return best_values, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Try multiple random restarts with different base structures
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate values
        values = _generate_values_with_common_factor()
        
        # Select edges optimized for product collisions
        edges = _select_edges_for_product_collisions(values, 35)
        
        # Evaluate
        score = _score(values, edges)
        
        # Local search
        if time.time() > deadline - 10:
            if score < best_score:
                best_score = score
                best_values = list(values)
                best_edges = [list(e) for e in edges]
            continue
        
        ls_time = min(5, max(0.5, deadline - time.time() - 2))
        values_ls, edges_ls, score_ls = _local_search(values, edges, time_limit=ls_time)
        
        if score_ls < best_score:
            best_score = score_ls
            best_values = list(values_ls)
            best_edges = [list(e) for e in edges_ls]
    
    # If we didn't find anything, use a default
    if best_values is None:
        best_values = list(range(1, 15))
        all_edges = list(combinations(range(14), 2))
        best_edges = [list(e) for e in all_edges[:35]]
    
    # Final validation and type conversion
    # Ensure values are Python int
    best_values = [int(v) for v in best_values]
    # Ensure edges are lists of Python int
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    # Verify constraints
    assert len(best_values) == 14
    assert len(set(best_values)) == 14
    assert all(1 <= v <= 10000 for v in best_values)
    assert len(best_edges) == 35
    assert all(len(e) == 2 for e in best_edges)
    assert all(e[0] != e[1] for e in best_edges)
    assert len(set(tuple(sorted(e)) for e in best_edges)) == 35
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
