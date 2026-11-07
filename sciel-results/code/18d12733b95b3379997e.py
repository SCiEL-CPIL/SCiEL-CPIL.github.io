import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    all_pairs = list(itertools.combinations(range(n), 2))
    
    def get_score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))
    
    def optimize_edges(vals):
        # Precompute sums and products for all pairs
        pair_data = []
        for i, j in all_pairs:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            pair_data.append((s, p, i, j))
            
        # Frequency counts to guide greedy choice
        sum_freq = Counter(s for s, p, i, j in pair_data)
        prod_freq = Counter(p for s, p, i, j in pair_data)
        
        # Greedy Selection
        # We want to pick 35 edges that minimize the number of distinct sums and products.
        # Heuristic: Pick edges that have sums/products already seen, or that are "popular".
        
        selected = []
        selected_set = set()
        seen_sums = set()
        seen_prods = set()
        
        # Sort pairs by a heuristic score: prefer those with high frequency sums/products
        # This helps in creating "clusters" of edges with same sum/prod
        scored_pairs = []
        for s, p, i, j in pair_data:
            # Weight: high frequency is good
            score = sum_freq[s] * 2 + prod_freq[p] * 2
            scored_pairs.append((score, s, p, i, j))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Iterate and pick best available
        # We can't just pick top 35 by score because they might all have unique sums.
        # We need to check the marginal cost.
        
        remaining = list(range(len(pair_data)))
        # To make it faster, we can just iterate through the sorted list and pick if it's beneficial?
        # No, we need exactly m edges.
        
        # Simple Greedy:
        # At each step, pick the edge that minimizes max(len(seen_sums), len(seen_prods)) after addition.
        # If tie, pick one with higher frequency score.
        
        available_indices = set(range(len(pair_data)))
        
        for _ in range(m):
            best_idx = -1
            best_cost = float('inf')
            best_score_tie = -1
            
            for idx in list(available_indices):
                s, p, i, j = pair_data[idx]
                
                new_sums = len(seen_sums) + (1 if s not in seen_sums else 0)
                new_prods = len(seen_prods) + (1 if p not in seen_prods else 0)
                cost = max(new_sums, new_prods)
                
                # Tie breaker: prefer higher frequency sum/prod to encourage future collisions
                tie_score = sum_freq[s] + prod_freq[p]
                
                if cost < best_cost or (cost == best_cost and tie_score > best_score_tie):
                    best_cost = cost
                    best_idx = idx
                    best_score_tie = tie_score
            
            if best_idx == -1:
                break
                
            s, p, i, j = pair_data[best_idx]
            selected.append([i, j])
            selected_set.add(frozenset((i, j)))
            seen_sums.add(s)
            seen_prods.add(p)
            available_indices.remove(best_idx)
            
        # Local Search on Edges
        # Try swapping an edge in 'selected' with one not in 'selected'
        # We need to be careful to maintain exactly m edges.
        
        def calc_score_for_edges(edge_list):
            sums = set()
            prods = set()
            for i, j in edge_list:
                sums.add(vals[i] + vals[j])
                prods.add(vals[i] * vals[j])
            return max(len(sums), len(prods))
        
        current_edges = [e[:] for e in selected]
        current_score = calc_score_for_edges(current_edges)
        
        # Precompute data for all pairs for fast lookup
        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            pair_sums[frozenset((i, j))] = vals[i] + vals[j]
            pair_prods[frozenset((i, j))] = vals[i] * vals[j]
            
        # Non-selected edges
        selected_keys = set(frozenset(e) for e in current_edges)
        non_selected_keys = [k for k in pair_sums.keys() if k not in selected_keys]
        
        improved = True
        max_swaps = 200
        swap_count = 0
        
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            # Shuffle non_selected to avoid bias
            random.shuffle(non_selected_keys)
            
            for new_edge_key in non_selected_keys:
                # Try removing each existing edge and adding new_edge_key
                for k in range(m):
                    old_edge_key = frozenset(current_edges[k])
                    if old_edge_key == new_edge_key:
                        continue
                    
                    # Calculate new score
                    # Remove old edge's contribution, add new edge's contribution
                    # This is expensive if done naively. 
                    # Since m=35, we can just recalculate.
                    
                    temp_edges = current_edges[:]
                    temp_edges[k] = list(new_edge_key)
                    
                    # Check for duplicates? 
                    # If new_edge_key is already in current_edges (other than k), it's invalid.
                    # But we picked from non_selected, so it's not there.
                    # However, we must ensure we don't create a duplicate edge in the list.
                    # Since we replace index k with a new edge not in the set, and old edge is removed,
                    # the set of edges remains simple.
                    
                    new_score = calc_score_for_edges(temp_edges)
                    
                    if new_score < current_score:
                        current_edges = temp_edges
                        current_score = new_score
                        improved = True
                        # Update non_selected_keys? 
                        # The edge we removed becomes non-selected, the one we added becomes selected.
                        # For simplicity in this loop, we just continue searching from the new state.
                        # We break inner loop and restart outer loop?
                        # Or just update and continue.
                        # Let's update and break to re-evaluate from scratch (safer).
                        # Actually, just update and break inner loop, then break outer?
                        # No, just update and break inner, then continue outer with updated state?
                        # The list non_selected_keys is now stale.
                        # Let's just restart the local search if we find an improvement.
                        break
                if improved:
                    # Rebuild non_selected_keys
                    selected_keys = set(frozenset(e) for e in current_edges)
                    non_selected_keys = [k for k in pair_sums.keys() if k not in selected_keys]
                    break
            swap_count += 1
            
        return current_edges, current_score
    
    # Value Optimization
    # Start with a good initial set
    # Small numbers are generally good for sums.
    # Numbers with many divisors are good for products.
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Candidate pool for values
    # We want distinct positive integers <= 10000
    # Small values are better.
    # Let's try a range of 1 to 100 initially.
    
    # Initialize with a few random sets
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Mix of small numbers and some highly composite numbers
        candidates = list(range(1, 50))
        random.shuffle(candidates)
        current_values = candidates[:n]
        current_values.sort()
        
        # Optimize edges
        current_edges, current_score = optimize_edges(current_values)
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values[:]
            best_edges = [e[:] for e in current_edges]
            
        # Local Search on Values
        # Simulated Annealing
        temperature = 1.0
        cooling = 0.95
        min_temp = 0.01
        
        while temperature > min_temp and time.time() < deadline:
            # Pick a random index to change
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            
            # Pick a new value
            # Try to stay in the range of current values to maintain "compactness"
            min_v = min(current_values)
            max_v = max(current_values)
            
            # Random new value
            # Bias towards small values
            new_val = random.randint(1, max(20, max_v))
            
            if new_val == old_val or new_val in current_values:
                continue
            
            temp_values = current_values[:]
            temp_values[idx] = new_val
            temp_values.sort()
            
            # Optimize edges for new values
            temp_edges, temp_score = optimize_edges(temp_values)
            
            # Accept if better, or with probability exp(-(delta)/T)
            delta = temp_score - current_score
            if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                current_values = temp_values
                current_edges = temp_edges
                current_score = temp_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = current_values[:]
                    best_edges = [e[:] for e in current_edges]
            
            temperature *= cooling
    
    # Final check and type conversion
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges, _ = optimize_edges(best_values)
        
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
