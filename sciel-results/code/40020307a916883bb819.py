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
    
    # For each edge, precompute the "forbidden pairs" - the other 3 edges that would complete a C4
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
            ok = True
            for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                if i1 in chosen and i2 in chosen and i3 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(idx)
        
        return chosen
    
    def local_search(chosen, max_iters=10000, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen = set(chosen)
        current_size = len(chosen)
        
        # Build list of unchosen edges
        unchosen = [i for i in range(total_edges) if i not in chosen]
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            if not unchosen:
                break
            
            # Try a random unchosen edge
            idx = rng.choice(unchosen)
            
            # Find which chosen edges conflict with idx
            blocking = []
            for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                if i1 in chosen and i2 in chosen and i3 in chosen:
                    blocking.append((i1, i2, i3))
            
            if not blocking:
                # Can add directly
                chosen.add(idx)
                current_size += 1
                unchosen.remove(idx)
            else:
                # Try removing one edge from a blocking triple and see if we can add idx
                # Pick a random blocking triple
                triple = rng.choice(blocking)
                # Try removing each edge in the triple
                for rem_idx in triple:
                    # Check if removing rem_idx and adding idx is valid
                    chosen.discard(rem_idx)
                    # Check if idx is now addable
                    can_add = True
                    for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                        if i1 in chosen and i2 in chosen and i3 in chosen:
                            can_add = False
                            break
                    if can_add:
                        chosen.add(idx)
                        # Net change: 0 (removed 1, added 1)
                        unchosen.remove(idx)
                        break
                    else:
                        # Restore
                        chosen.add(rem_idx)
        
        return chosen
    
    def two_opt_search(chosen, max_iters=5000, seed=None):
        """Try 2-Opt style moves: remove 2 edges, add 2 edges"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen = set(chosen)
        current_size = len(chosen)
        
        unchosen = [i for i in range(total_edges) if i not in chosen]
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            if not unchosen:
                break
            
            # Pick 2 random unchosen edges
            if len(unchosen) < 2:
                break
            idx1, idx2 = rng.sample(unchosen, 2)
            
            # Find blocking triples for each
            blocking1 = []
            for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                if i1 in chosen and i2 in chosen and i3 in chosen:
                    blocking1.append((i1, i2, i3))
            
            blocking2 = []
            for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                if i1 in chosen and i2 in chosen and i3 in chosen:
                    blocking2.append((i1, i2, i3))
            
            if not blocking1 and not blocking2:
                # Can add both directly
                chosen.add(idx1)
                chosen.add(idx2)
                current_size += 2
                unchosen.remove(idx1)
                unchosen.remove(idx2)
            elif not blocking1:
                # idx1 can be added, idx2 needs help
                for triple in blocking2:
                    for rem_idx in triple:
                        chosen.discard(rem_idx)
                        can_add2 = True
                        for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                            if i1 in chosen and i2 in chosen and i3 in chosen:
                                can_add2 = False
                                break
                        if can_add2:
                            # Also check idx1 can still be added
                            can_add1 = True
                            for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                                if i1 in chosen and i2 in chosen and i3 in chosen:
                                    can_add1 = False
                                    break
                            if can_add1:
                                chosen.add(idx1)
                                chosen.add(idx2)
                                current_size += 1  # net +1 (removed 1, added 2)
                                unchosen.remove(idx1)
                                unchosen.remove(idx2)
                                break
                            else:
                                chosen.add(rem_idx)
                        else:
                            chosen.add(rem_idx)
                    else:
                        continue
                    break
            elif not blocking2:
                # idx2 can be added, idx1 needs help
                for triple in blocking1:
                    for rem_idx in triple:
                        chosen.discard(rem_idx)
                        can_add1 = True
                        for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                            if i1 in chosen and i2 in chosen and i3 in chosen:
                                can_add1 = False
                                break
                        if can_add1:
                            # Also check idx2 can still be added
                            can_add2 = True
                            for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                                if i1 in chosen and i2 in chosen and i3 in chosen:
                                    can_add2 = False
                                    break
                            if can_add2:
                                chosen.add(idx1)
                                chosen.add(idx2)
                                current_size += 1  # net +1
                                unchosen.remove(idx1)
                                unchosen.remove(idx2)
                                break
                            else:
                                chosen.add(rem_idx)
                        else:
                            chosen.add(rem_idx)
                    else:
                        continue
                    break
            else:
                # Both need help - try removing one edge from each blocking set
                # This is more complex, skip for now
                pass
        
        return chosen
    
    best_solution = None
    best_size = 0
    
    # Phase 1: Multiple randomized greedy restarts
    num_restarts = 0
    while time.time() - start_time < time_limit - 40:
        seed = num_restarts
        solution = greedy_randomized(seed)
        size = len(solution)
        if size > best_size:
            best_size = size
            best_solution = solution
        num_restarts += 1
    
    # Phase 2: Local search on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 15:
        improved = local_search(best_solution, max_iters=20000, seed=42)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Phase 3: 2-Opt search
    if best_solution is not None and time.time() - start_time < time_limit - 10:
        improved = two_opt_search(best_solution, max_iters=10000, seed=123)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Phase 4: More local search iterations if time permits
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            improved = local_search(best_solution, max_iters=10000, seed=random.randint(0, 100000))
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
