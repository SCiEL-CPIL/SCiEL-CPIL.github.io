"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

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

def _generate_value_candidates():
    """
    Generate diverse sets of 14 distinct positive integers <= 10000.
    We want sets that naturally produce many collisions in sums and products.
    """
    candidates = []
    
    # 1. Small integers (high additive collision)
    candidates.append(list(range(1, 15)))
    
    # 2. Highly composite numbers (high multiplicative collision)
    # 1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24
    candidates.append([1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24])
    
    # 3. Powers of 2 mixed with small primes? No, powers of 2 are bad (unique products).
    # Let's try multiples of a small base to force product collisions.
    # e.g., multiples of 2: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28
    candidates.append([2*i for i in range(1, 15)])
    
    # 4. Random sets with small numbers (1-50)
    random.seed(123)
    base = list(range(1, 51))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 5. Another random set
    random.seed(456)
    base = list(range(1, 101))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 6. Numbers with many divisors
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18])
    
    # 7. Arithmetic progression
    candidates.append([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    
    return candidates

def _local_search_graph(values, initial_edges, max_iters=2000, time_limit=2.0):
    """
    Improve the graph structure (edges) for a fixed set of values using local search.
    Moves: Swap an existing edge with a non-existing edge.
    """
    n = len(values)
    all_edges = set(combinations(range(n), 2))
    current_edges = set(tuple(sorted(e)) for e in initial_edges)
    
    # Precompute sums and products for all possible edges to speed up evaluation
    # But we need dynamic sets. 
    # Optimization: Keep track of counts for each sum/product value.
    
    def get_score_counts(edges):
        sum_counts = {}
        prod_counts = {}
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        return max(distinct_sums, distinct_prods), sum_counts, prod_counts

    current_score, current_sum_counts, current_prod_counts = get_score_counts(current_edges)
    
    start_time = time.time()
    best_score = current_score
    best_edges = set(current_edges)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random existing edge to remove
        edge_to_remove = random.choice(list(current_edges))
        
        # Pick a random non-existing edge to add
        # To make it faster, we can pick from a precomputed list of all edges
        # and filter, or just generate random pairs until we find a non-edge.
        # With 91 edges and 35 selected, probability of collision is ~40%.
        # Generating random pairs is O(1) expected.
        
        while True:
            u, v = random.sample(range(n), 2)
            candidate_edge = (min(u, v), max(u, v))
            if candidate_edge not in current_edges:
                break
        
        # Simulate the move
        # Remove edge_to_remove, add candidate_edge
        new_edges = current_edges - {edge_to_remove} | {candidate_edge}
        
        # Calculate new score efficiently?
        # For small N, just recalculating is fast enough.
        new_score, _, _ = get_score_counts(new_edges)
        
        if new_score <= current_score:
            # Accept (Hill Climbing with plateau acceptance)
            current_edges = new_edges
            current_score = new_score
            current_sum_counts, current_prod_counts = get_score_counts(current_edges)
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        else:
            # Simulated Annealing style acceptance? 
            # For now, strict hill climbing is safer for deterministic improvement.
            # But to escape local minima, we might need to accept worse moves occasionally.
            # Given the small search space, let's stick to strict improvement or equal.
            pass
            
    return [list(e) for e in best_edges], best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_score = float('inf')
    best_result = None
    
    value_sets = _generate_value_candidates()
    
    # Shuffle value sets to vary the order of exploration
    random.seed(42)
    random.shuffle(value_sets)
    
    for values in value_sets:
        if time.time() > deadline:
            break
            
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Try multiple random initializations of the graph
        # We will run local search from different random starts
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Randomly select 35 edges from the complete graph
            all_possible_edges = list(combinations(range(14), 2))
            random.shuffle(all_possible_edges)
            initial_edges = [list(e) for e in all_possible_edges[:35]]
            
            # Run local search
            optimized_edges, score = _local_search_graph(values, initial_edges, max_iters=1000, time_limit=1.5)
            
            # Validate edges
            if len(optimized_edges) != 35:
                continue
                
            edge_tuples = set(tuple(sorted(e)) for e in optimized_edges)
            if len(edge_tuples) != 35:
                continue
                
            # Check for loops
            if any(u == v for u, v in optimized_edges):
                continue
                
            if score < best_score:
                best_score = score
                best_result = {
                    'values': values,
                    'edges': optimized_edges
                }
                
                # Early exit if we hit a very low score
                # Theoretical minimum?
                # If we can get down to ~10-15, that's great.
                if best_score <= 12:
                    break
                    
        if time.time() > deadline:
            break
            
    if best_result is None:
        # Fallback to a simple deterministic construction
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {
            'values': values,
            'edges': edges
        }
        
    return best_result
# EVOLVE-BLOCK-END
