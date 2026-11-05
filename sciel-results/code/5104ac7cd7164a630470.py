import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
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
    # Edge (u, v) with dimension i: for each j != i, the C4 is u, v, u^bit_j, v^bit_j
    # The three other edges are: (u, u^bit_j), (v, v^bit_j), (u^bit_j, v^bit_j)
    edge_to_forbidden = {}
    for idx, (u, v, i) in enumerate(all_edges):
        forbidden = []
        for j in range(d):
            if j != i:
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                # Three other edges in this C4
                e1 = (min(u, c), max(u, c))
                e2 = (min(v, e), max(v, e))
                e3 = (min(c, e), max(c, e))
                forbidden.append((e1, e2, e3))
        edge_to_forbidden[idx] = forbidden
    
    # Convert forbidden to index lists for faster lookup
    edge_to_forbidden_idx = {}
    for idx in range(total_edges):
        forbidden = []
        for (e1, e2, e3) in edge_to_forbidden[idx]:
            i1 = edge_to_idx.get(e1)
            i2 = edge_to_idx.get(e2)
            i3 = edge_to_idx.get(e3)
            if i1 is not None and i2 is not None and i3 is not None:
                forbidden.append((i1, i2, i3))
        edge_to_forbidden_idx[idx] = forbidden
    
    def is_addable(idx, chosen):
        """Check if edge idx can be added to chosen set without creating a C4"""
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            if i1 in chosen and i2 in chosen and i3 in chosen:
                return False
        return True
    
    def greedy_randomized(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen = set()  # set of edge indices chosen
        # Shuffle edges for random order
        order = list(range(total_edges))
        rng.shuffle(order)
        
        for idx in order:
            if is_addable(idx, chosen):
                chosen.add(idx)
        
        return chosen
    
    def local_search(chosen, max_iters=10000, seed=None):
        """Local search: try to add unchosen edges, removing blocking edges if needed"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen = set(chosen)
        current_size = len(chosen)
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Build list of unchosen edges
            unchosen = [i for i in range(total_edges) if i not in chosen]
            if not unchosen:
                break
            
            # Try a random unchosen edge
            idx = rng.choice(unchosen)
            
            if is_addable(idx, chosen):
                # Can add directly
                chosen.add(idx)
                current_size += 1
            else:
                # Find blocking triples
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if i1 in chosen and i2 in chosen and i3 in chosen:
                        blocking.append((i1, i2, i3))
                
                if not blocking:
                    chosen.add(idx)
                    current_size += 1
                else:
                    # Try removing one edge from a random blocking triple
                    triple = rng.choice(blocking)
                    for rem_idx in triple:
                        chosen.discard(rem_idx)
                        if is_addable(idx, chosen):
                            chosen.add(idx)
                            # Net change: 0 (removed 1, added 1)
                            break
                        else:
                            # Restore
                            chosen.add(rem_idx)
        
        return chosen
    
    def swap_search(chosen, max_iters=5000, seed=None):
        """Try swap moves: remove 1 edge, add 1 different edge (net 0), 
        but the goal is to enable future additions. Also try remove 1 add 2 (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen = set(chosen)
        current_size = len(chosen)
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = [i for i in range(total_edges) if i not in chosen]
            if not unchosen:
                break
            
            # Pick a random unchosen edge
            idx = rng.choice(unchosen)
            
            if is_addable(idx, chosen):
                chosen.add(idx)
                current_size += 1
            else:
                # Find blocking triples
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if i1 in chosen and i2 in chosen and i3 in chosen:
                        blocking.append((i1, i2, i3))
                
                if not blocking:
                    chosen.add(idx)
                    current_size += 1
                else:
                    # Try removing one edge from each blocking triple
                    # and check if we can add idx
                    for triple in blocking:
                        for rem_idx in triple:
                            chosen.discard(rem_idx)
                            if is_addable(idx, chosen):
                                chosen.add(idx)
                                # Net: 0
                                break
                            else:
                                chosen.add(rem_idx)
        
        return chosen
    
    def two_opt_search(chosen, max_iters=3000, seed=None):
        """Try 2-Opt: remove 2 edges, add 2 edges (net 0), or remove 1 add 2 (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen = set(chosen)
        current_size = len(chosen)
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = [i for i in range(total_edges) if i not in chosen]
            if len(unchosen) < 2:
                break
            
            # Pick 2 random unchosen edges
            idx1, idx2 = rng.sample(unchosen, 2)
            
            # Check if both can be added directly
            if is_addable(idx1, chosen) and is_addable(idx2, chosen):
                chosen.add(idx1)
                chosen.add(idx2)
                current_size += 2
            else:
                # Try removing 1 edge to enable adding both
                # Get all blocking edges for idx1 and idx2
                block1 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                    if i1 in chosen and i2 in chosen and i3 in chosen:
                        block1.update([i1, i2, i3])
                
                block2 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                    if i1 in chosen and i2 in chosen and i3 in chosen:
                        block2.update([i1, i2, i3])
                
                # Try removing one edge from the union of blocking sets
                candidates = list(block1 | block2)
                rng.shuffle(candidates)
                
                for rem_idx in candidates[:3]:  # Try at most 3 removals
                    chosen.discard(rem_idx)
                    if is_addable(idx1, chosen) and is_addable(idx2, chosen):
                        chosen.add(idx1)
                        chosen.add(idx2)
                        current_size += 1  # Net +1
                        break
                    else:
                        chosen.add(rem_idx)
        
        return chosen
    
    best_solution = None
    best_size = 0
    
    # Phase 1: Multiple randomized greedy restarts
    num_restarts = 0
    while time.time() - start_time < time_limit - 35:
        seed = num_restarts
        solution = greedy_randomized(seed)
        size = len(solution)
        if size > best_size:
            best_size = size
            best_solution = solution
        num_restarts += 1
    
    # Phase 2: Local search on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 15:
        improved = local_search(best_solution, max_iters=15000, seed=42)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Phase 3: Swap search
    if best_solution is not None and time.time() - start_time < time_limit - 10:
        improved = swap_search(best_solution, max_iters=8000, seed=123)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Phase 4: 2-Opt search
    if best_solution is not None and time.time() - start_time < time_limit - 8:
        improved = two_opt_search(best_solution, max_iters=5000, seed=456)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Phase 5: More local search iterations if time permits
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            improved = local_search(best_solution, max_iters=8000, seed=random.randint(0, 100000))
            if len(improved) > best_size:
                best_size = len(improved)
                best_solution = improved
    
    if best_solution is None:
        # Fallback: deterministic greedy
        best_solution = greedy_randomized(0)
    
    # Convert to output format
    result = []
    for idx in sorted(best_solution):
        u, v, i = all_edges[idx]
        result.append([u, v])
    
    return result
