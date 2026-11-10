import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    maximum = 10000
    
    # Precompute all pairs
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.append((i, j))
    
    # Candidate value sets: small, factor-rich
    candidate_values = [
        [2,3,4,5,6,8,9,10,12,15,16,18,20,24],
        [1,2,3,4,5,6,7,8,9,10,12,15,18,20],
        [2,4,6,8,10,12,14,16,18,20,24,30,36,40],
        [3,4,5,6,8,9,10,12,15,16,18,20,24,30],
        [1,2,3,4,6,8,9,12,15,18,20,24,30,36],
        [2,4,6,9,10,12,15,16,18,20,24,27,30,36],
        [1,2,4,6,8,9,10,12,15,16,18,20,24,30],
        [2,3,6,8,9,10,12,15,16,18,20,24,27,30],
    ]
    
    # Also generate some random small value sets
    random.seed(12345)
    for _ in range(50):
        vals = random.sample(range(1, 50), n)
        candidate_values.append(vals)
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    def evaluate_edges(values, edge_indices):
        """Given values and a set of edge indices (into all_pairs), compute raw value."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            i, j = all_pairs[idx]
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    def greedy_edge_selection(values, seed=0):
        """Greedy: pick edges that minimize new distinct sums+prods."""
        rng = random.Random(seed)
        # Precompute sum/prod for each pair
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        
        used_sums = set()
        used_prods = set()
        selected = []
        remaining = list(range(len(all_pairs)))
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            best_tie = -1
            for k in remaining:
                s = pair_sum[k]
                p = pair_prod[k]
                new_sums = len(used_sums) + (1 if s not in used_sums else 0)
                new_prods = len(used_prods) + (1 if p not in used_prods else 0)
                score = max(new_sums, new_prods)
                if score < best_score:
                    best_score = score
                    best_idx = k
                    best_tie = 0
                elif score == best_score:
                    # Tie-break: prefer pairs whose sum/prod are already common
                    tie = (1 if s in used_sums else 0) + (1 if p in used_prods else 0)
                    if tie > best_tie:
                        best_tie = tie
                        best_idx = k
            if best_idx == -1:
                break
            selected.append(best_idx)
            used_sums.add(pair_sum[best_idx])
            used_prods.add(pair_prod[best_idx])
            remaining.remove(best_idx)
        
        return selected
    
    def local_search(values, initial_edges, max_iters=20000, seed=0):
        """Local search: swap edges to improve objective."""
        rng = random.Random(seed)
        edge_set = set(initial_edges)
        current_raw = evaluate_edges(values, edge_set)
        
        # Precompute
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        
        # Maintain counts for fast evaluation
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[pair_sum[idx]] += 1
            prod_counts[pair_prod[idx]] += 1
        
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        current_raw = max(distinct_sums, distinct_prods)
        
        unselected = list(set(range(len(all_pairs))) - edge_set)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if it % 5000 == 0 and time.time() > deadline - 5:
                break
            
            # Pick a random selected edge to remove
            remove_idx = rng.choice(list(edge_set))
            # Pick a random unselected edge to add
            add_idx = rng.choice(unselected)
            
            if remove_idx == add_idx:
                continue
            
            # Compute new distinct counts
            rs = pair_sum[remove_idx]
            rp = pair_prod[remove_idx]
            as_ = pair_sum[add_idx]
            ap = pair_prod[add_idx]
            
            # Effect on sum_counts
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            sum_counts[as_] = sum_counts.get(as_, 0) + 1
            
            # Effect on prod_counts
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            prod_counts[ap] = prod_counts.get(ap, 0) + 1
            
            new_distinct_sums = len(sum_counts)
            new_distinct_prods = len(prod_counts)
            new_raw = max(new_distinct_sums, new_distinct_prods)
            
            if new_raw < current_raw:
                # Accept
                edge_set.remove(remove_idx)
                edge_set.add(add_idx)
                unselected.remove(remove_idx)
                unselected.append(add_idx)
                current_raw = new_raw
            else:
                # Revert
                sum_counts[rs] = sum_counts.get(rs, 0) + 1
                sum_counts[as_] -= 1
                if sum_counts[as_] == 0:
                    del sum_counts[as_]
                
                prod_counts[rp] = prod_counts.get(rp, 0) + 1
                prod_counts[ap] -= 1
                if prod_counts[ap] == 0:
                    del prod_counts[ap]
        
        return list(edge_set), current_raw
    
    # Try each candidate value set
    for vi, values in enumerate(candidate_values):
        if time.time() > deadline:
            break
        
        if len(set(values)) != n or any(v < 1 or v > maximum for v in values):
            continue
        
        # Multiple greedy starts + local search
        for restart in range(5):
            if time.time() > deadline:
                break
            
            greedy_edges = greedy_edge_selection(values, seed=restart * 100 + vi)
            ls_edges, ls_raw = local_search(values, greedy_edges, max_iters=15000, seed=restart * 1000 + vi)
            
            if ls_raw < best_raw:
                best_raw = ls_raw
                best_values = list(values)
                best_edges = [list(all_pairs[idx]) for idx in ls_edges]
                
                if best_raw <= 15:
                    return {'values': best_values, 'edges': best_edges}
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in __import__('itertools').combinations(range(n), 2)[:m]]
    
    return {'values': best_values, 'edges': best_edges}
