"""Construct a 14-node, 35-edge graph with 14 distinct positive ints <=10000 minimizing
max(#distinct edge sums, #distinct edge products)."""

import random
import math
from itertools import combinations


def _calculate_score(values, edges):
    """Calculate the score: max(distinct sums, distinct products)."""
    sset = set()
    pset = set()
    for u, v in edges:
        sset.add(values[u] + values[v])
        pset.add(values[u] * values[v])
    return max(len(sset), len(pset))


def _initialize_state(rng):
    """Initialize a random valid state."""
    # Generate 14 distinct positive integers <= 10000
    # Start with small numbers to encourage collisions
    values = rng.sample(range(1, 200), 14)
    # Ensure distinctness is handled by sample
    # Ensure they are ints
    values = [int(v) for v in values]
    
    # Generate 35 random edges from all possible 91
    all_edges = list(combinations(range(14), 2))
    edges = rng.sample(all_edges, 35)
    edges = [list(e) for e in edges]
    
    return values, edges


def _simulate_annealing(iters=500000, start_temp=10.0, end_temp=0.1, rng=None):
    if rng is None:
        rng = random.Random(42)
        
    values, edges = _initialize_state(rng)
    
    # Precompute all possible edges
    all_edges = list(combinations(range(14), 2))
    all_edges_set = set(all_edges)
    
    # Initial score
    current_score = _calculate_score(values, edges)
    best_score = current_score
    best_values = values[:]
    best_edges = [e[:] for e in edges]
    
    # To speed up scoring, we can maintain counts, but given the small size,
    # recalculating is ~35*2 operations. 500k iters * 70 ops = 35M ops.
    # Python might be slow for 500k full recalcs.
    # Let's use incremental updates or a faster scoring method if possible.
    # However, SA needs many iterations.
    # Let's try a hybrid: keep counts of sums and products for the current edge set.
    
    # Re-initialize with counts for speed
    sum_counts = {}
    prod_counts = {}
    for u, v in edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    current_distinct_sums = len(sum_counts)
    current_distinct_prods = len(prod_counts)
    current_score = max(current_distinct_sums, current_distinct_prods)
    
    def get_score():
        return max(len(sum_counts), len(prod_counts))
    
    def update_counts_add(u, v):
        s = values[u] + values[v]
        p = values[u] * values[v]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    def update_counts_remove(u, v):
        s = values[u] + values[v]
        p = values[u] * values[v]
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
        prod_counts[p] -= 1
        if prod_counts[p] == 0:
            del prod_counts[p]
            
    for i in range(iters):
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / iters)
        
        # Generate a neighbor
        # 50% chance to mutate values, 50% chance to mutate edges
        
        new_values = values[:]
        new_edges = edges[:]
        
        if rng.random() < 0.5:
            # Mutate values
            # Strategy: Pick two indices and swap, or pick one and add small random noise
            op = rng.randint(0, 1)
            if op == 0:
                # Swap
                idx1, idx2 = rng.sample(range(14), 2)
                new_values[idx1], new_values[idx2] = new_values[idx2], new_values[idx1]
            else:
                # Random walk on one value
                idx = rng.randint(0, 13)
                delta = rng.randint(-5, 5)
                new_val = new_values[idx] + delta
                if new_val > 0 and new_val <= 10000:
                    # Check distinctness
                    if new_val not in new_values or new_values.count(new_val) == 0: 
                        # Actually, need to check if new_val is already in the list at another index
                        if new_val in new_values and new_values.index(new_val) != idx:
                            pass # Collision, reject mutation or try again? Let's just reject this step.
                        else:
                            new_values[idx] = new_val
                    else:
                        # Out of bounds, reject mutation
                        new_values = values[:]
                else:
                    new_values = values[:]
            
            # If values changed, we must recalculate all sums/products for current edges
            # This is expensive: 35 edges * 2 ops.
            # Clear counts
            sum_counts.clear()
            prod_counts.clear()
            for u, v in edges:
                s = new_values[u] + new_values[v]
                p = new_values[u] * new_values[v]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
                
            new_score = get_score()
            
            # Acceptance
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                # Accept
                values = new_values
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_values = values[:]
                    best_edges = edges[:]
            else:
                # Reject
                # No need to revert counts as we cleared and rebuilt, 
                # but wait, we cleared the global counts. We need to restore them.
                # This implementation is buggy because we mutated global sum_counts/prod_counts 
                # inside the "if" block but didn't restore them on reject.
                # Let's fix the logic: 
                # 1. Save old counts? No, too heavy.
                # 2. Recalculate counts from current values/edges on reject? Heavy.
                # 3. Better: Don't modify global counts until acceptance.
                #    Calculate new_score in a local scope.
                
                # Let's refactor to calculate new score without modifying global state first.
                # But that's what we did. The problem is we MODIFIED sum_counts/prod_counts.
                # We must revert.
                # Reverting is as expensive as calculating.
                # So, on reject, we just recalculate sum_counts/prod_counts from current values/edges.
                sum_counts.clear()
                prod_counts.clear()
                for u, v in edges:
                    s = values[u] + values[v]
                    p = values[u] * values[v]
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1

        else:
            # Mutate edges
            # Swap an edge or replace an edge
            op = rng.randint(0, 1)
            if op == 0:
                # Replace one edge with a random different edge
                edge_idx = rng.randint(0, 34)
                old_u, old_v = edges[edge_idx]
                
                # Pick a random edge from all possible edges
                # Ensure it's not the same as old one
                while True:
                    new_u, new_v = rng.choice(all_edges)
                    if (new_u, new_v) != (old_u, old_v) and (new_v, new_u) != (old_u, old_v):
                        break
                
                # Check if new edge already exists in edges
                edge_exists = False
                for e in edges:
                    if (e[0] == new_u and e[1] == new_v) or (e[0] == new_v and e[1] == new_u):
                        edge_exists = True
                        break
                
                if edge_exists:
                    continue # Skip this step if duplicate edge created
                
                # Calculate new score
                # Remove old edge contributions
                # Add new edge contributions
                # To avoid modifying global state, we calculate delta.
                # But distinct counts are hard to delta.
                # Let's just recalculate fully. 35 edges is small.
                
                # Temporarily modify edges list? No, create new list
                new_edges = edges[:]
                new_edges[edge_idx] = [new_u, new_v]
                
                sset = set()
                pset = set()
                for u, v in new_edges:
                    sset.add(values[u] + values[v])
                    pset.add(values[u] * values[v])
                new_score = max(len(sset), len(pset))
                
                delta = new_score - current_score
                if delta <= 0 or rng.random() < math.exp(-delta / temp):
                    # Accept
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = values[:]
                        best_edges = edges[:]
                    
                    # Update global counts for next iteration
                    sum_counts.clear()
                    prod_counts.clear()
                    for u, v in edges:
                        s = values[u] + values[v]
                        p = values[u] * values[v]
                        sum_counts[s] = sum_counts.get(s, 0) + 1
                        prod_counts[p] = prod_counts.get(p, 0) + 1
                # If rejected, do nothing (global counts remain valid for current state)

            else:
                # Swap two edges? Or just random restart?
                # Let's stick to single edge replacement for simplicity and efficiency.
                pass

    return best_values, best_edges, best_score


def construct():
    rng = random.Random(123)
    
    best_overall_score = 10**9
    best_overall_values = None
    best_overall_edges = None
    
    # Run SA multiple times with different seeds/temps
    for seed in [0, 1, 2]:
        rng_run = random.Random(seed)
        vals, edges, score = _simulate_annealing(iters=100000, start_temp=5.0, end_temp=0.1, rng=rng_run)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vals
            best_overall_edges = edges
            
            if best_overall_score <= 14:
                break
                
    # Final validation and type conversion
    values = [int(v) for v in best_overall_values]
    edges = [[int(u), int(v)] for u, v in best_overall_edges]
    
    # Ensure distinct values
    if len(set(values)) != 14:
        # Fallback to simple heuristic if SA failed distinctness (unlikely with sample init and check)
        values = list(range(1, 15))
        all_edges = list(combinations(range(14), 2))
        edges = [list(e) for e in all_edges[:35]]
        
    return {'values': values, 'edges': edges}
