import time
import random
from itertools import combinations
from collections import defaultdict

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _max_matching(n, adj, weights):
    """
    Find a maximum cardinality matching in a graph.
    We use a simple greedy approach with backtracking for small N=14.
    Actually, for N=14, we can just do a simple recursive DFS or even brute force 
    if needed, but greedy + local search is sufficient.
    Here we implement a simple greedy matching that prioritizes high weights.
    """
    matched = [False] * n
    match = [-1] * n
    
    # Sort edges by weight descending
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v] > 0: # Only consider edges that exist in our 'collision graph'
                edges_list.append((weights[u][v], u, v))
    edges_list.sort(reverse=True)
    
    for w, u, v in edges_list:
        if not matched[u] and not matched[v]:
            matched[u] = True
            matched[v] = True
            match[u] = v
            match[v] = u
            
    # Return the list of matched edges
    result = []
    for i in range(n):
        if match[i] != -1 and i < match[i]:
            result.append((i, match[i]))
    return result

def _select_edges_optimized(values, deadline):
    """
    Select 35 edges to minimize max(distinct sums, distinct products).
    Strategy:
    1. Precompute all possible sums and products.
    2. Identify 'high collision' edges: edges where the sum or product is shared by many other potential edges.
    3. Use a greedy set-cover-like approach but with a 'density' metric.
    4. To ensure we get exactly 35 edges, we first pick the 'best' 35 based on a combined score of 
       sum-collision-count and product-collision-count.
    5. Then, we perform a local search (swap) to improve the score.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        edge_sums[(u, v)] = values[u] + values[v]
        edge_prods[(u, v)] = values[u] * values[v]
        
    # Count frequency of each sum and product across ALL possible edges
    sum_freq = defaultdict(int)
    prod_freq = defaultdict(int)
    for e in all_edges:
        sum_freq[edge_sums[e]] += 1
        prod_freq[edge_prods[e]] += 1
        
    # Score an edge based on how 'common' its sum and product are.
    # Higher frequency means adding this edge is less likely to increase distinct counts.
    edge_score = {}
    for e in all_edges:
        s = edge_sums[e]
        p = edge_prods[e]
        # We want to maximize the frequency. 
        # But we need to balance sums and prods.
        # Let's use a simple additive score.
        edge_score[e] = sum_freq[s] + prod_freq[p]
        
    # Initial selection: Top 35 edges by score
    sorted_edges = sorted(all_edges, key=lambda x: edge_score[x], reverse=True)
    selected_edges = sorted_edges[:35]
    selected_set = set(selected_edges)
    
    # Calculate initial score
    current_score = _raw_score(values, selected_edges)
    
    # Local Search: Try to swap an edge in with an edge out to improve score
    # We limit iterations to respect time budget.
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < (deadline - 0.5):
        improved = False
        # Randomize the order of current edges to avoid getting stuck in the same local minimum
        random.shuffle(selected_edges)
        
        for i, e_in in enumerate(selected_edges):
            # Try swapping e_in with some e_out not in selected_set
            # We only need to check a subset of e_out to save time
            # Let's pick a few random candidates for e_out
            candidates_out = []
            for _ in range(10):
                e_out = random.choice(all_edges)
                if e_out not in selected_set:
                    candidates_out.append(e_out)
            
            for e_out in candidates_out:
                new_selected = list(selected_edges)
                new_selected[i] = e_out
                new_set = set(new_selected)
                
                # Quick check: is this swap likely to help?
                # Calculate new score
                new_score = _raw_score(values, new_selected)
                
                if new_score < current_score:
                    # Accept the swap
                    selected_edges[i] = e_out
                    selected_set.remove(e_in)
                    selected_set.add(e_out)
                    current_score = new_score
                    improved = True
                    break # Move to next e_in
                    
                elif new_score == current_score:
                    # Accept with some probability to escape plateaus? 
                    # For now, let's just accept to diversify
                    if random.random() < 0.1:
                        selected_edges[i] = e_out
                        selected_set.remove(e_in)
                        selected_set.add(e_out)
                        improved = True
                        break
                        
    return [list(e) for e in selected_edges], current_score

def _generate_value_candidates():
    """
    Generate value sets designed to maximize collisions.
    Focus on numbers that share factors and are close in magnitude.
    """
    candidates = []
    
    # 1. Small integers 1-14
    candidates.append(list(range(1, 15)))
    
    # 2. Multiples of 2
    candidates.append([i * 2 for i in range(1, 15)])
    
    # 3. Multiples of 3
    candidates.append([i * 3 for i in range(1, 15)])
    
    # 4. Highly composite small numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36])
    
    # 5. Arithmetic progression diff 2
    candidates.append([1 + i*2 for i in range(14)])
    
    # 6. Multiples of 6
    candidates.append([6*i for i in range(1, 15)])
    
    # 7. Random small numbers
    random.seed(123)
    base = list(range(1, 50))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 8. Another random set
    random.seed(456)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 9. Mixed small primes and composites
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 10. Values centered around 10
    candidates.append([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    
    # 11. Values with many divisors
    candidates.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # 12. Random 1-20
    random.seed(789)
    base = list(range(1, 21))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 13. Close numbers
    candidates.append([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    
    # 14. Multiples of 4
    candidates.append([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56])
    
    # 15. New: Numbers with lots of small prime factors (Highly Composite)
    # 1, 2, 4, 6, 8, 12, 16, 24, 30, 32, 36, 40, 48, 60
    candidates.append([1, 2, 4, 6, 8, 12, 16, 24, 30, 32, 36, 40, 48, 60])
    
    # 16. New: Geometric progression with ratio close to 1? No, integers.
    # Try: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 is already there.
    # Try: 100, 101, ...
    candidates.append([100 + i for i in range(14)])
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_candidates()
    
    # Shuffle value sets to avoid bias
    random.seed(100)
    random.shuffle(value_sets)
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Allocate time for edge optimization
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Give each value set a chunk of time
        edge_time = min(3.0, time_left / 2.0)
        
        edges, score = _select_edges_optimized(values, edge_time)
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {
                'values': values,
                'edges': edges
            }
            
        # Early exit if we hit a very low score
        if best_global_score <= 10:
            break

    if best_global_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {
            'values': values,
            'edges': edges
        }
        
    return best_global_result
