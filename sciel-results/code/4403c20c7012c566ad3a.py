"""Strategy: Joint Simulated Annealing over Value Sets and Edge Sets.
Uses multiplicative structures (powers of small primes) to create product collisions
and dense subgraphs to create sum collisions. Vectorized evaluation for speed."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    
    # Precompute all possible edges
    all_edges = np.array([(i, j) for i in range(n) for j in range(i+1, n)])
    edge_indices = all_edges  # (91, 2)
    
    # Generate a diverse set of initial value configurations
    # Strategy: Mix powers of 2, 3, 5 to ensure many product collisions
    # Strategy: Mix arithmetic progressions to ensure sum collisions
    
    initial_configs = []
    
    # Config 1: Pure powers of 2 and 3 (High product collision)
    # 2^a * 3^b
    pow_2_3 = []
    for a in range(0, 5):
        for b in range(0, 5):
            v = (2**a) * (3**b)
            if v <= 10000:
                pow_2_3.append(v)
    random.shuffle(pow_2_3)
    if len(pow_2_3) >= n:
        initial_configs.append(np.array(pow_2_3[:n], dtype=np.int64))
        
    # Config 2: Arithmetic progression (High sum collision potential if edges are chosen well)
    # Values: 10, 20, ..., 140
    ap_vals = np.array([10 * (i+1) for i in range(n)], dtype=np.int64)
    initial_configs.append(ap_vals)
    
    # Config 3: Mixed small integers with some structure
    # 1, 2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 40, 80
    mixed = [1, 2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 40, 80]
    initial_configs.append(np.array(mixed, dtype=np.int64))
    
    # Config 4: Random smooth numbers
    smooth_primes = [2, 3, 5, 7]
    random.seed(123)
    smooth_vals = set()
    while len(smooth_vals) < n:
        v = 1
        for _ in range(random.randint(1, 6)):
            v *= random.choice(smooth_primes)
        if v <= 10000:
            smooth_vals.add(v)
    initial_configs.append(np.array(list(smooth_vals), dtype=np.int64))
    
    def evaluate(vals, edge_idx_subset):
        """
        vals: np.array of shape (n,)
        edge_idx_subset: np.array of indices into all_edges, shape (m,)
        Returns: max(distinct_sums, distinct_prods)
        """
        # Get the actual (u, v) pairs
        u = edge_indices[edge_idx_subset, 0]
        v = edge_indices[edge_idx_subset, 1]
        
        # Vectorized sums and products
        sums = vals[u] + vals[v]
        prods = vals[u] * vals[v]
        
        # Count distinct
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    def get_best_edges_for_vals(vals):
        """
        Greedily select m edges that minimize the number of distinct sums and products.
        This is a heuristic initialization for the edge set.
        """
        # Precompute all edge sums and products
        u_all = edge_indices[:, 0]
        v_all = edge_indices[:, 1]
        all_sums = vals[u_all] + vals[v_all]
        all_prods = vals[u_all] * vals[v_all]
        
        # We want to pick m edges such that the number of unique values in 
        # the selected sums and prods is minimized.
        
        # Strategy: Group edges by (sum, prod). Pick edges from the largest groups first.
        # This maximizes collisions.
        
        # Create a key for each edge
        # We can't easily hash numpy arrays of tuples, so let's use a dictionary
        edge_groups = {}
        for i in range(len(all_edges)):
            key = (all_sums[i], all_prods[i])
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(i)
            
        # Sort groups by size descending
        sorted_groups = sorted(edge_groups.items(), key=lambda x: -len(x[1]))
        
        selected_edge_indices = []
        selected_count = 0
        for key, indices in sorted_groups:
            for idx in indices:
                if selected_count >= m:
                    break
                selected_edge_indices.append(idx)
                selected_count += 1
            if selected_count >= m:
                break
                
        return np.array(selected_edge_indices, dtype=np.int32)

    best_score = float('inf')
    best_vals = None
    best_edges = None
    
    # Simulated Annealing
    # State: (vals, edge_indices_subset)
    # Move 1: Change a value
    # Move 2: Swap an edge in the subset
    
    for init_vals in initial_configs:
        if time.time() > deadline - 10.0:
            break
            
        current_vals = init_vals.copy()
        current_edges = get_best_edges_for_vals(current_vals)
        
        if len(current_edges) < m:
            continue
            
        current_score = evaluate(current_vals, current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_vals = current_vals.copy()
            best_edges = current_edges.copy()
            
        # SA Loop
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.995
        
        # Precompute random edge indices for swapping
        all_edge_indices = np.arange(len(all_edges))
        
        steps = 0
        max_steps = 50000
        no_improve_steps = 0
        
        while T > T_min and steps < max_steps and time.time() < deadline - 5.0:
            steps += 1
            
            # Choose move type: 50% value, 50% edge
            if random.random() < 0.5:
                # Value Move
                idx = random.randint(0, n - 1)
                old_val = current_vals[idx]
                
                # Generate new value
                # Bias towards smooth numbers or nearby values
                if random.random() < 0.5:
                    # Smooth number
                    v = 1
                    for _ in range(random.randint(1, 5)):
                        v *= random.choice([2, 3, 5, 7])
                    if v > 10000:
                        v = random.randint(1, 10000)
                else:
                    # Random
                    v = random.randint(1, 10000)
                    
                # Ensure distinctness
                if v in current_vals and v != old_val:
                    # Try a few times to find a distinct value
                    for _ in range(10):
                        v = random.randint(1, 10000)
                        if v not in current_vals:
                            break
                    if v in current_vals:
                        continue
                        
                current_vals[idx] = v
                new_score = evaluate(current_vals, current_edges)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_vals = current_vals.copy()
                        best_edges = current_edges.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    current_vals[idx] = old_val
                    no_improve_steps += 1
            else:
                # Edge Move
                # Pick an edge in the current set and swap it with a random edge not in the set
                pos = random.randint(0, m - 1)
                old_edge_idx = current_edges[pos]
                
                # Pick a new edge
                # To avoid duplicates, we can pick a random edge and check if it's already in the set
                # For speed, we can just pick a random edge index and hope it's not the same, 
                # or maintain a set of used indices.
                # Given m=35 and total=91, probability of collision is ~35/91 ~ 38%.
                # Let's just try a few times.
                
                new_edge_idx = None
                for _ in range(20):
                    cand = random.choice(all_edge_indices)
                    if cand not in current_edges:
                        new_edge_idx = cand
                        break
                        
                if new_edge_idx is None:
                    continue
                    
                current_edges[pos] = new_edge_idx
                new_score = evaluate(current_vals, current_edges)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_vals = current_vals.copy()
                        best_edges = current_edges.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    current_edges[pos] = old_edge_idx
                    no_improve_steps += 1
            
            T *= cooling_rate
            
            # Early stop if stuck for a long time
            if no_improve_steps > 2000:
                break

    # Final Polish: Local search on best result
    if best_vals is not None and time.time() < deadline - 2.0:
        current_vals = best_vals.copy()
        current_edges = best_edges.copy()
        current_score = best_score
        
        # Try swapping values to improve score
        for _ in range(5000):
            if time.time() > deadline - 1.0:
                break
            idx = random.randint(0, n - 1)
            old_val = current_vals[idx]
            # Try a few random values
            for _ in range(5):
                v = random.randint(1, 10000)
                if v in current_vals and v != old_val:
                    continue
                current_vals[idx] = v
                new_score = evaluate(current_vals, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    best_vals = current_vals.copy()
                    best_edges = current_edges.copy()
                    break
                else:
                    current_vals[idx] = old_val
                    break
        
        # Try swapping edges to improve score
        for _ in range(5000):
            if time.time() > deadline - 1.0:
                break
            pos = random.randint(0, m - 1)
            old_edge_idx = current_edges[pos]
            for _ in range(5):
                cand = random.choice(all_edge_indices)
                if cand in current_edges:
                    continue
                current_edges[pos] = cand
                new_score = evaluate(current_vals, current_edges)
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    best_vals = current_vals.copy()
                    best_edges = current_edges.copy()
                    break
                else:
                    current_edges[pos] = old_edge_idx
                    break

    # Convert to required format
    # Ensure Python ints and lists
    vals_list = [int(v) for v in best_vals]
    edges_list = []
    for idx in best_edges:
        u, v = all_edges[idx]
        edges_list.append([int(u), int(v)])
        
    # Sort edges for consistency (not required but good practice)
    edges_list.sort()
    
    # Validation
    if len(set(vals_list)) != 14:
        # Fallback if distinctness failed
        vals_list = list(range(1, 15))
        edges_list = [[i, j] for i in range(14) for j in range(i+1, 14)][:35]
        
    return {'values': vals_list, 'edges': edges_list}
# EVOLVE-BLOCK-END
