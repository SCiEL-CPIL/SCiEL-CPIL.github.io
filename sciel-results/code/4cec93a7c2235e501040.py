import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    d = 7
    n = 2 ** d  # 128 vertices

    # Precompute all edges: (u, v, dim) where u < v and dim is the bit that differs
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    total_edges = len(all_edges)  # 448

    # Build a mapping from edge tuple to index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx

    # For each edge, precompute the "forbidden triples" - the other 3 edges that would complete a C4
    edge_to_forbidden_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        forbidden = []
        for j in range(d):
            if j != i:
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                e1 = (min(u, c), max(u, c))
                e2 = (min(v, e), max(v, e))
                e3 = (min(c, e), max(c, e))
                i1 = edge_to_idx.get(e1)
                i2 = edge_to_idx.get(e2)
                i3 = edge_to_idx.get(e3)
                if i1 is not None and i2 is not None and i3 is not None:
                    forbidden.append((i1, i2, i3))
        edge_to_forbidden_idx[idx] = forbidden

    # Build conflict graph: edge i conflicts with edge j if they share a 2-face
    # Two edges conflict if adding both could be part of a C4
    # Actually, for the independent set formulation:
    # Two edges are "conflicting" if they are in the same 2-face
    # In Q7, two edges are in the same 2-face if they differ in exactly 2 bits and the 
    # 4 vertices form a square.
    
    # Precompute for each edge, which other edges are in a 2-face with it
    # Edge (u,v) with dim i: the 2-faces containing it are for each j != i:
    #   vertices u, v, u^(1<<j), v^(1<<j)
    # The other edges in this face are:
    #   (u, u^(1<<j)), (v, v^(1<<j)), (u^(1<<j), v^(1<<j))
    # So edge idx conflicts with these 3 edges for each j != i.
    
    # But for C4 constraint: we need all 4 edges of a face to NOT be selected.
    # This is a 4-uniform hypergraph constraint, not a simple graph constraint.
    # The "conflict graph" approach for maximum independent set works for 2-uniform constraints.
    # Here we have 4-uniform constraints (C4s).
    
    # Let's stick with the hypergraph approach but use a smarter greedy.
    
    def is_addable(idx, chosen_arr):
        """Check if edge idx can be added to chosen set (numpy bool array) without creating a C4"""
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                return False
        return True

    def is_addable_fast(idx, chosen_arr):
        """Vectorized check if edge idx can be added"""
        triples = edge_to_forbidden_idx[idx]
        if not triples:
            return True
        # Convert list of tuples to numpy array for vectorization
        arr = np.array(triples)
        # Get boolean values for all indices in triples
        vals = chosen_arr[arr]
        # Reshape to (num_triples, 3)
        vals = vals.reshape(-1, 3)
        # Check if any row is all True
        if np.all(vals, axis=1).any():
            return False
        return True

    def count_conflicts(idx, chosen_arr):
        """Count how many C4s would be completed if we added edge idx"""
        count = 0
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                count += 1
        return count

    def greedy_mcs(seed=None):
        """Maximum Cardinality Search heuristic: iteratively select the edge with 
        the fewest conflicts (most addable options remaining)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = np.zeros(total_edges, dtype=bool)
        remaining = list(range(total_edges))
        
        while remaining:
            if time.time() - start_time > time_limit - 5:
                break
            
            # For each remaining edge, compute how many of its forbidden triples are fully chosen
            # (i.e., how many C4s would be completed)
            best_idx = None
            best_score = None
            
            # Sample a subset for speed if remaining is large
            if len(remaining) > 100:
                sample = rng.sample(remaining, 100)
            else:
                sample = remaining
            
            for idx in sample:
                # Count how many forbidden triples are "almost complete" (2 of 3 chosen)
                # This is a proxy for how constrained this edge is
                score = 0
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    c = (chosen_arr[i1] and chosen_arr[i2]) or \
                        (chosen_arr[i1] and chosen_arr[i3]) or \
                        (chosen_arr[i2] and chosen_arr[i3])
                    if c:
                        score += 1
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        # This edge cannot be added
                        score = 1000000
                        break
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None and best_score < 1000000:
                chosen_arr[best_idx] = True
                remaining.remove(best_idx)
            else:
                # Try to find any addable edge
                found = False
                for idx in sample:
                    if is_addable_fast(idx, chosen_arr):
                        chosen_arr[idx] = True
                        remaining.remove(idx)
                        found = True
                        break
                if not found:
                    break
        
        return chosen_arr

    def greedy_randomized(seed=None):
        """Randomized greedy: shuffle edges and add if possible"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = np.zeros(total_edges, dtype=bool)
        order = list(range(total_edges))
        rng.shuffle(order)
        
        for idx in order:
            if is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
        
        return chosen_arr

    def greedy_decreasing_degree(seed=None):
        """Greedy by decreasing degree in conflict hypergraph: 
        prefer edges that are in fewer C4s (less constraining)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = np.zeros(total_edges, dtype=bool)
        
        # Each edge is in (d-1) * 2 = 12 C4s (for d=7)
        # But the constraint is that all 4 edges of a C4 cannot be selected.
        # Let's just use a weighted random selection
        
        # Precompute: for each edge, how many C4s it participates in
        # All edges participate in exactly (d-1) C4s = 6 C4s
        
        # Use a priority queue approach: repeatedly pick the edge with fewest 
        # "active" C4s (C4s where some edges are already chosen)
        
        remaining = list(range(total_edges))
        
        while remaining:
            if time.time() - start_time > time_limit - 5:
                break
            
            best_idx = None
            best_score = None
            
            # Sample for speed
            if len(remaining) > 80:
                sample = rng.sample(remaining, 80)
            else:
                sample = remaining
            
            for idx in sample:
                # Count how many C4s involving idx have at least one other edge chosen
                score = 0
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if chosen_arr[i1] or chosen_arr[i2] or chosen_arr[i3]:
                        score += 1
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        score = 1000000
                        break
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None and best_score < 1000000:
                chosen_arr[best_idx] = True
                remaining.remove(best_idx)
            else:
                found = False
                for idx in sample:
                    if is_addable_fast(idx, chosen_arr):
                        chosen_arr[idx] = True
                        remaining.remove(idx)
                        found = True
                        break
                if not found:
                    break
        
        return chosen_arr

    def local_search(chosen_arr, max_iters=10000, seed=None):
        """Local search: try to add unchosen edges, removing blocking edges if needed"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Build list of unchosen edges
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) == 0:
                break
            
            # Try a random unchosen edge
            idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
            
            if is_addable_fast(idx, chosen_arr):
                # Can add directly
                chosen_arr[idx] = True
                current_size += 1
            else:
                # Find blocking triples
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        blocking.append((i1, i2, i3))
                
                if not blocking:
                    chosen_arr[idx] = True
                    current_size += 1
                else:
                    # Try removing one edge from a random blocking triple
                    triple = blocking[rng.randint(0, len(blocking) - 1)]
                    for rem_idx in triple:
                        chosen_arr[rem_idx] = False
                        if is_addable_fast(idx, chosen_arr):
                            chosen_arr[idx] = True
                            # Net change: 0 (removed 1, added 1)
                            break
                        else:
                            # Restore
                            chosen_arr[rem_idx] = True
        
        return chosen_arr

    def simulated_annealing(chosen_arr, max_iters=20000, seed=None):
        """Simulated Annealing: accept worse states with temperature-dependent probability"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        best_arr = chosen_arr.copy()
        best_size = current_size
        
        T_start = 5.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
        T = T_start
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            T *= cooling_rate
            
            unchosen = np.where(~chosen_arr)[0]
            chosen_indices = np.where(chosen_arr)[0]
            
            if len(unchosen) == 0:
                break
            
            # Random move: either add an unchosen edge or remove a chosen edge
            if rng.random() < 0.7 and len(unchosen) > 0:
                # Try to add a random unchosen edge
                idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
                
                if is_addable_fast(idx, chosen_arr):
                    chosen_arr[idx] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_arr = chosen_arr.copy()
                else:
                    # Find blocking triples and try a swap
                    blocking = []
                    for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                        if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                            blocking.append((i1, i2, i3))
                    
                    if blocking:
                        triple = blocking[rng.randint(0, len(blocking) - 1)]
                        rem_idx = triple[rng.randint(0, 2)]
                        old_size = current_size
                        chosen_arr[rem_idx] = False
                        if is_addable_fast(idx, chosen_arr):
                            chosen_arr[idx] = True
                            # Net 0, always accept
                            if current_size > best_size:
                                best_size = current_size
                                best_arr = chosen_arr.copy()
                        else:
                            # Restore
                            chosen_arr[rem_idx] = True
            else:
                # Try to remove a random chosen edge (to escape local optima)
                if len(chosen_indices) > 0:
                    rem_idx = int(chosen_indices[rng.randint(0, len(chosen_indices) - 1)])
                    old_size = current_size
                    chosen_arr[rem_idx] = False
                    current_size -= 1
                    
                    # Accept with probability exp(-delta / T) where delta = -1 (size decreased)
                    delta = -1
                    accept_prob = np.exp(-delta / T) if T > 1e-6 else 0.0
                    if rng.random() < accept_prob:
                        # Accept the removal
                        if current_size > best_size:
                            best_size = current_size
                            best_arr = chosen_arr.copy()
                    else:
                        # Reject: restore
                        chosen_arr[rem_idx] = True
                        current_size = old_size
        
        return best_arr

    def swap_and_add(chosen_arr, max_iters=5000, seed=None):
        """Try to swap 1 edge for 2 edges (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) < 2:
                break
            
            # Pick 2 random unchosen edges
            s = rng.sample(range(len(unchosen)), 2)
            idx1 = int(unchosen[s[0]])
            idx2 = int(unchosen[s[1]])
            
            # Check if both can be added directly
            if is_addable_fast(idx1, chosen_arr) and is_addable_fast(idx2, chosen_arr):
                chosen_arr[idx1] = True
                chosen_arr[idx2] = True
                current_size += 2
            else:
                # Try removing 1 edge to enable adding both
                block1 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block1.update([i1, i2, i3])
                
                block2 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block2.update([i1, i2, i3])
                
                # Try removing one edge from the union of blocking sets
                candidates = list(block1 | block2)
                rng.shuffle(candidates)
                
                for rem_idx in candidates[:5]:  # Try at most 5 removals
                    chosen_arr[rem_idx] = False
                    if is_addable_fast(idx1, chosen_arr) and is_addable_fast(idx2, chosen_arr):
                        chosen_arr[idx1] = True
                        chosen_arr[idx2] = True
                        current_size += 1  # Net +1
                        break
                    else:
                        chosen_arr[rem_idx] = True
        
        return chosen_arr

    best_solution = None
    best_size = 0

    # Phase 1: Multiple greedy strategies with different seeds
    num_restarts = 0
    while time.time() - start_time < time_limit - 35:
        seed = num_restarts
        
        # Alternate between different greedy strategies
        if num_restarts % 3 == 0:
            solution = greedy_randomized(seed)
        elif num_restarts % 3 == 1:
            solution = greedy_mcs(seed)
        else:
            solution = greedy_decreasing_degree(seed)
        
        size = int(solution.sum())
        if size > best_size:
            best_size = size
            best_solution = solution
        num_restarts += 1
        
        # Additional polish: try local search on each new solution if it's close to best
        if size >= best_size - 5 and time.time() - start_time < time_limit - 30:
            polished = local_search(solution, max_iters=2000, seed=seed + 1000)
            polished_size = int(polished.sum())
            if polished_size > best_size:
                best_size = polished_size
                best_solution = polished
    
    # Phase 2: Local search on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 25:
        improved = local_search(best_solution, max_iters=15000, seed=42)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 3: Simulated Annealing
    if best_solution is not None and time.time() - start_time < time_limit - 15:
        improved = simulated_annealing(best_solution, max_iters=25000, seed=789)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 4: Swap and add
    if best_solution is not None and time.time() - start_time < time_limit - 10:
        improved = swap_and_add(best_solution, max_iters=8000, seed=123)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 5: More local search + SA iterations if time permits
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            improved = local_search(best_solution, max_iters=8000, seed=random.randint(0, 100000))
            if int(improved.sum()) > best_size:
                best_size = int(improved.sum())
                best_solution = improved
            if time.time() - start_time > time_limit - 5:
                break
            improved = simulated_annealing(best_solution, max_iters=10000, seed=random.randint(0, 100000))
            if int(improved.sum()) > best_size:
                best_size = int(improved.sum())
                best_solution = improved
    
    if best_solution is None:
        # Fallback: deterministic greedy
        best_solution = greedy_randomized(0)
    
    # Convert to output format
    result = []
    for idx in np.where(best_solution)[0]:
        u, v, i = all_edges[idx]
        result.append([u, v])
    
    return result
