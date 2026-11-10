import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_edges = list(combinations(range(n), 2))
    E = len(all_edges)
    
    best_score = float('inf')
    best_result = None  # (values, edges_list)
    
    def update_best(vals, edges_list, score):
        nonlocal best_score, best_result
        if score < best_score:
            best_score = score
            best_result = (list(vals), [list(e) for e in edges_list])
    
    # Generate diverse value candidates
    val_candidates = []
    seen_val_sets = set()
    
    def add_candidate(vals):
        if max(vals) > maximum or len(set(vals)) != n:
            return
        key = tuple(sorted(vals))
        if key not in seen_val_sets:
            seen_val_sets.add(key)
            val_candidates.append(sorted(vals))
    
    # 1. Small consecutive integers
    for start_v in range(1, 30):
        vals = [start_v + i for i in range(n)]
        add_candidate(vals)
    
    # 2. Arithmetic progressions
    for base in range(1, 50):
        for diff in range(1, 20):
            vals = [base + i * diff for i in range(n)]
            add_candidate(vals)
    
    # 3. Geometric progressions (multiplicative structure)
    for base in range(1, 50):
        for ratio in range(2, 15):
            vals = [base * (ratio ** i) for i in range(n)]
            add_candidate(vals)
    
    # 4. Powers of small integers
    for base in range(2, 20):
        vals = [base ** i for i in range(n)]
        add_candidate(vals)
    
    # 5. Prime-based sets
    def first_primes(count):
        primes = []
        candidate = 2
        while len(primes) < count:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    primes = first_primes(20)
    for i in range(len(primes) - n + 1):
        add_candidate(primes[i:i+n])
    
    # 6. Random small values (sorted)
    for _ in range(100):
        vals = sorted(random.sample(range(1, 300), n))
        add_candidate(vals)
    
    # 7. Near-Sidon sets: values where sums are mostly distinct but with some structure
    for _ in range(50):
        vals = []
        for i in range(n):
            # Use quadratic growth for near-Sidon property
            vals.append(i * i + random.randint(0, 5))
        add_candidate(sorted(vals))
    
    # For each value set, perform edge optimization
    for vals in val_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Precompute sums and products for all edges
        edge_sums = [vals[u] + vals[v] for u, v in all_edges]
        edge_prods = [vals[u] * vals[v] for u, v in all_edges]
        
        # Calculate frequency of each sum and product
        sum_freq = {}
        prod_freq = {}
        for s in edge_sums:
            sum_freq[s] = sum_freq.get(s, 0) + 1
        for p in edge_prods:
            prod_freq[p] = prod_freq.get(p, 0) + 1
        
        # Create edge priority: prefer edges that contribute to common sums/products
        edge_priority = []
        for idx in range(E):
            s = edge_sums[idx]
            p = edge_prods[idx]
            # Higher priority if sum and product are more frequent
            priority = sum_freq[s] * 2 + prod_freq[p] * 2 + (sum_freq[s] + prod_freq[p])
            edge_priority.append((priority, idx))
        edge_priority.sort(reverse=True)
        
        # Greedy initialization: pick top m edges by priority
        selected_indices = []
        used = set()
        for priority, idx in edge_priority:
            if len(selected_indices) >= m:
                break
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
        
        if len(selected_indices) < m:
            continue
        
        edges_list = [all_edges[idx] for idx in selected_indices]
        
        # Calculate initial score
        sums_set = set()
        prods_set = set()
        for idx in selected_indices:
            sums_set.add(edge_sums[idx])
            prods_set.add(edge_prods[idx])
        current_score = max(len(sums_set), len(prods_set))
        
        update_best(vals, edges_list, current_score)
        
        # Local search: hill climbing with swaps
        # Count occurrences of each sum/product in current edges
        sum_counts = {}
        prod_counts = {}
        for idx in selected_indices:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        improved = True
        max_iterations = 2000
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if time.time() > deadline - 5.0:
                break
            
            # Try systematic swaps: for each edge in, find best edge out
            # This is O(m * E) per iteration but more thorough
            best_swap = None
            best_new_score = current_score
            
            for e_in in range(E):
                if e_in in used:
                    continue
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                # For each possible e_out, compute score change
                for e_out_pos, e_out in enumerate(selected_indices):
                    s_out = edge_sums[e_out]
                    p_out = edge_prods[e_out]
                    
                    # Calculate how many distinct sums/products would change
                    # Remove e_out
                    sum_count_out = sum_counts.get(s_out, 0)
                    prod_count_out = prod_counts.get(p_out, 0)
                    
                    # Add e_in
                    sum_count_in = sum_counts.get(s_in, 0)
                    prod_count_in = prod_counts.get(p_in, 0)
                    
                    # New distinct sums: if sum_count_out == 1 and sum_count_in == 0, we lose one distinct sum
                    # If sum_count_in == 0 and sum_count_out >= 1, we gain one distinct sum
                    delta_sums = 0
                    if sum_count_out == 1 and sum_count_in == 0:
                        delta_sums -= 1
                    elif sum_count_out >= 1 and sum_count_in == 0:
                        delta_sums += 1
                    elif sum_count_out == 0 and sum_count_in >= 1:
                        delta_sums += 1
                    
                    delta_prods = 0
                    if prod_count_out == 1 and prod_count_in == 0:
                        delta_prods -= 1
                    elif prod_count_out >= 1 and prod_count_in == 0:
                        delta_prods += 1
                    elif prod_count_out == 0 and prod_count_in >= 1:
                        delta_prods += 1
                    
                    new_sums = len(sum_counts) + delta_sums
                    new_prods = len(prod_counts) + delta_prods
                    new_score = max(new_sums, new_prods)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e_out_pos, e_out, e_in)
            
            if best_swap is not None:
                e_out_pos, e_out, e_in = best_swap
                s_out = edge_sums[e_out]
                p_out = edge_prods[e_out]
                s_in = edge_sums[e_in]
                p_in = edge_prods[e_in]
                
                # Update counts
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                selected_indices[e_out_pos] = e_in
                used.discard(e_out)
                used.add(e_in)
                current_score = best_new_score
                improved = True
                
                edges_list = [all_edges[idx] for idx in selected_indices]
                update_best(vals, edges_list, current_score)
        
        # Additional random restarts for local search
        if time.time() < deadline - 15.0:
            for restart in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                # Random restart from current best
                rs_selected = list(selected_indices)
                rs_used = set(rs_selected)
                rs_sum_counts = dict(sum_counts)
                rs_prod_counts = dict(prod_counts)
                rs_score = current_score
                
                rs_improved = True
                rs_iter = 0
                while rs_improved and rs_iter < 500:
                    rs_improved = False
                    rs_iter += 1
                    
                    if time.time() > deadline - 5.0:
                        break
                    
                    for _ in range(200):
                        if time.time() > deadline - 5.0:
                            break
                        
                        e_out = random.choice(rs_selected)
                        e_in = random.randint(0, E - 1)
                        
                        if e_in in rs_used:
                            continue
                        
                        s_out = edge_sums[e_out]
                        p_out = edge_prods[e_out]
                        s_in = edge_sums[e_in]
                        p_in = edge_prods[e_in]
                        
                        rs_sum_counts[s_out] -= 1
                        if rs_sum_counts[s_out] == 0:
                            del rs_sum_counts[s_out]
                        rs_prod_counts[p_out] -= 1
                        if rs_prod_counts[p_out] == 0:
                            del rs_prod_counts[p_out]
                        
                        rs_sum_counts[s_in] = rs_sum_counts.get(s_in, 0) + 1
                        rs_prod_counts[p_in] = rs_prod_counts.get(p_in, 0) + 1
                        
                        new_score = max(len(rs_sum_counts), len(rs_prod_counts))
                        
                        if new_score < rs_score:
                            pos = rs_selected.index(e_out)
                            rs_selected[pos] = e_in
                            rs_used.discard(e_out)
                            rs_used.add(e_in)
                            rs_score = new_score
                            rs_improved = True
                            
                            if rs_score < current_score:
                                selected_indices = list(rs_selected)
                                used = set(rs_used)
                                sum_counts = dict(rs_sum_counts)
                                prod_counts = dict(rs_prod_counts)
                                current_score = rs_score
                                edges_list = [all_edges[idx] for idx in selected_indices]
                                update_best(vals, edges_list, current_score)
                        else:
                            rs_sum_counts[s_in] -= 1
                            if rs_sum_counts[s_in] == 0:
                                del rs_sum_counts[s_in]
                            rs_prod_counts[p_in] -= 1
                            if rs_prod_counts[p_in] == 0:
                                del rs_prod_counts[p_in]
                            
                            rs_sum_counts[s_out] = rs_sum_counts.get(s_out, 0) + 1
                            rs_prod_counts[p_out] = rs_prod_counts.get(p_out, 0) + 1
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges_list = [list(e) for e in all_edges[:m]]
        best_result = (vals, edges_list)
    
    vals, edges_list = best_result
    
    # Ensure types are correct
    vals = [int(v) for v in vals]
    edges_list = [[int(i), int(j)] for i, j in edges_list]
    
    return {'values': vals, 'edges': edges_list}
