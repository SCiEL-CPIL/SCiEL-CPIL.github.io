import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 118.0  # Safety margin: killed at 150s
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute popcount for fast Hamming distance
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(MAX_V):
        popcount[i] = bin(i).count('1')
    
    # Precompute compatibility masks: compatible_mask[v] has bit u set if dist(v,u) >= 2
    # We need to check: new vertex v is compatible with all used vertices except the one it's adjacent to in the path
    compatible_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 0
        for u in range(MAX_V):
            if popcount[v ^ u] >= 2:
                mask |= (1 << u)
        compatible_mask[v] = mask
    
    best_path = []
    best_len = 0
    
    # Precompute for each vertex, the list of neighbors sorted by some heuristic? 
    # Actually, let's use a more sophisticated approach: 
    # 1. Use a "frontier" based search where we maintain the set of possible next vertices
    # 2. Use constraint propagation: if a vertex has only one possible extension, we must take it
    
    def greedy_extend_constrained(path, used_mask, randomize=True):
        """
        Extend path using constraint propagation:
        - At each step, find all valid candidates
        - If any candidate has only 1 future option (degree 1 in the induced subgraph of unused vertices), 
          it's a "forced" move - but actually we want to avoid getting stuck, so we prefer low degree
        - Use Warnsdorff's rule with constraint propagation
        """
        if time.time() > deadline:
            return path, used_mask
        
        # For constraint propagation, we need to track "free" vertices:
        # A vertex is "free" if it has at least one valid extension
        # When we add a vertex, some neighbors become blocked
        
        # Let's use a simpler but effective approach:
        # 1. Find all valid candidates
        # 2. For each candidate, compute its "future degree" (number of valid extensions it would have)
        # 3. Prefer candidates with lower future degree (Warnsdorff)
        # 4. If a candidate has future degree 0, it's a dead end - avoid it unless no other choice
        
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            
            # Find valid candidates: unused vertices adjacent to cur
            candidates = []
            for nb in neighbors[cur]:
                if (used_mask >> nb) & 1:
                    continue
                # Check compatibility with all other used vertices
                mask_others = used_mask ^ (1 << cur)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    # Compute future degree: how many valid extensions nb would have
                    new_used = used_mask | (1 << nb)
                    deg = 0
                    for nn in neighbors[nb]:
                        if (new_used >> nn) & 1:
                            continue
                        # Check if nn is compatible with all vertices in new_used except nb
                        mask_others_nb = new_used ^ (1 << nb)
                        if (mask_others_nb & ~compatible_mask[nn]) == 0:
                            deg += 1
                    candidates.append((deg, nb))
            
            if not candidates:
                return path, used_mask
            
            # Warnsdorff: prefer lower degree
            candidates.sort(key=lambda x: x[0])
            
            # Pick with some randomness
            if randomize and len(candidates) > 1:
                # Pick from top few
                k = min(4, len(candidates))
                chosen = random.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            
            # Add chosen
            path.append(chosen)
            used_mask |= (1 << chosen)
        
    def backtrack_search(path, used_mask, max_depth, depth=0):
        """
        Backtracking search with pruning.
        Try to extend the path, and if stuck, backtrack.
        Use time limit to bound the search.
        """
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return path, used_mask
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if depth >= max_depth:
            return path, used_mask
        
        cur = path[-1]
        
        # Find valid candidates
        candidates = []
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            mask_others = used_mask ^ (1 << cur)
            if (mask_others & ~compatible_mask[nb]) == 0:
                # Compute future degree for pruning
                new_used = used_mask | (1 << nb)
                deg = 0
                for nn in neighbors[nb]:
                    if (new_used >> nn) & 1:
                        continue
                    mask_others_nb = new_used ^ (1 << nb)
                    if (mask_others_nb & ~compatible_mask[nn]) == 0:
                        deg += 1
                candidates.append((deg, nb))
        
        if not candidates:
            return path, used_mask
        
        # Sort by future degree (Warnsdorff)
        candidates.sort(key=lambda x: x[0])
        
        # Try each candidate
        for deg, nb in candidates:
            if time.time() > deadline:
                break
            
            new_path = path + [nb]
            new_used = used_mask | (1 << nb)
            
            # Recurse
            result_path, result_used = backtrack_search(new_path, new_used, max_depth, depth + 1)
            
            # If we found a better solution, update
            if len(result_path) > len(path):
                path = result_path
                used_mask = result_used
        
        return path, used_mask
    
    def local_search_improve(path, used_mask, max_iters=20):
        """
        Improve a path by trying to extend at either end.
        Also try to "shift" the path: remove a vertex from one end and add to the other.
        """
        if time.time() > deadline:
            return path, used_mask
        
        best = path[:]
        best_mask = used_mask
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(best)
            
            # Try extending the path at the front
            first = best[0]
            for nb in neighbors[first]:
                if (best_mask >> nb) & 1:
                    continue
                mask_others = best_mask ^ (1 << first)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    new_path = [nb] + best
                    new_mask = best_mask | (1 << nb)
                    best = new_path
                    best_mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            
            # Try extending the path at the back
            last = best[-1]
            for nb in neighbors[last]:
                if (best_mask >> nb) & 1:
                    continue
                mask_others = best_mask ^ (1 << last)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    new_path = best + [nb]
                    new_mask = best_mask | (1 << nb)
                    best = new_path
                    best_mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            
            # Try to shift: remove first vertex, see if we can add to the back
            if len(best) > 2:
                old_first = best[0]
                candidate_path = best[1:]
                candidate_mask = best_mask ^ (1 << old_first)
                last = candidate_path[-1]
                for nb in neighbors[last]:
                    if (candidate_mask >> nb) & 1:
                        continue
                    mask_others = candidate_mask ^ (1 << last)
                    if (mask_others & ~compatible_mask[nb]) == 0:
                        new_path = candidate_path + [nb]
                        new_mask = candidate_mask | (1 << nb)
                        if len(new_path) > len(best):
                            best = new_path
                            best_mask = new_mask
                            improved = True
                        break
                if improved:
                    continue
            
            # Try to shift: remove last vertex, see if we can add to the front
            if len(best) > 2:
                old_last = best[-1]
                candidate_path = best[:-1]
                candidate_mask = best_mask ^ (1 << old_last)
                first = candidate_path[0]
                for nb in neighbors[first]:
                    if (candidate_mask >> nb) & 1:
                        continue
                    mask_others = candidate_mask ^ (1 << first)
                    if (mask_others & ~compatible_mask[nb]) == 0:
                        new_path = [nb] + candidate_path
                        new_mask = candidate_mask | (1 << nb)
                        if len(new_path) > len(best):
                            best = new_path
                            best_mask = new_mask
                            improved = True
                        break
                if improved:
                    continue
            
            break
        
        return best, best_mask
    
    # Main search: multiple random restarts with different strategies
    random.seed(42)
    
    # Strategy 1: Pure greedy with Warnsdorff
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    # Phase 1: Quick greedy restarts to find a good initial solution
    num_quick_restarts = 200
    for i in range(num_quick_restarts):
        if time.time() > deadline - 30:
            break
        
        start = start_vertices[i % MAX_V]
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = greedy_extend_constrained(path, used_mask, randomize=True)
        path, used_mask = local_search_improve(path, used_mask, max_iters=5)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Backtracking search from the best solution found
    if best_path and time.time() < deadline - 15:
        # Use the best path as a starting point for backtracking
        # Try to extend it further with backtracking
        max_bt_depth = 10
        bt_path, bt_used = backtrack_search(best_path[:], 
                                             int(''.join(str((1 << v) & -1) for v in best_path), 2) if False else sum(1 << v for v in best_path),
                                             max_bt_depth)
        
        # Actually, let's just do greedy extension from the best path with less randomization
        if time.time() < deadline - 10:
            bt_path, bt_used = greedy_extend_constrained(best_path[:], 
                                                         sum(1 << v for v in best_path),
                                                         randomize=False)
            bt_path, bt_used = local_search_improve(bt_path, bt_used, max_iters=10)
            
            if len(bt_path) > best_len:
                best_len = len(bt_path)
                best_path = bt_path[:]
    
    # Phase 3: More aggressive search with different random seeds
    if time.time() < deadline - 5:
        random.seed(99999)
        for i in range(100):
            if time.time() > deadline - 2:
                break
            
            start = random.randint(0, MAX_V - 1)
            path = [start]
            used_mask = 1 << start
            
            # Use less randomization for more deterministic extension
            path, used_mask = greedy_extend_constrained(path, used_mask, randomize=(i % 2 == 0))
            path, used_mask = local_search_improve(path, used_mask, max_iters=10)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    # Final verification and return
    if not best_path:
        return [0]
    
    return best_path
