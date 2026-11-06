import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # Safety margin: killed at 150s
    
    # Precompute neighbors using bit operations
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute popcount for fast Hamming distance
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(MAX_V):
        popcount[i] = bin(i).count('1')
    
    # Precompute for each vertex v, a bitmask of vertices u such that dist(v, u) >= 2.
    compatible_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 0
        for u in range(MAX_V):
            if popcount[v ^ u] >= 2:
                mask |= (1 << u)
        compatible_mask[v] = mask
    
    best_path = []
    best_len = 0
    
    def backtrack_extend(path, used_mask, depth_limit=None):
        """
        Constraint-propagation backtracking solver.
        Uses MRV (Minimum Remaining Values) heuristic:
        - At each step, the current end has a small set of candidates.
        - We pick the candidate that leaves the fewest options for the next step.
        - If we get stuck, we backtrack and try a different candidate.
        """
        nonlocal best_path, best_len, deadline
        
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
                candidates.append(nb)
        
        if not candidates:
            # Dead end - update best if this path is longer
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            return path, used_mask
        
        # MRV heuristic: for each candidate, count how many valid extensions it would have
        # Pick the candidate with the fewest extensions (most constrained)
        scored_candidates = []
        for nb in candidates:
            new_used = used_mask | (1 << nb)
            deg = 0
            for nn in neighbors[nb]:
                if (new_used >> nn) & 1:
                    continue
                mask_others_nb = new_used ^ (1 << nb)
                if (mask_others_nb & ~compatible_mask[nn]) == 0:
                    deg += 1
            scored_candidates.append((deg, nb))
        
        # Sort by degree (ascending) - most constrained first
        scored_candidates.sort(key=lambda x: x[0])
        
        # Try candidates in order of increasing degree
        for deg, nb in scored_candidates:
            if time.time() > deadline:
                return path, used_mask
            
            # Extend
            path.append(nb)
            used_mask |= (1 << nb)
            
            # Recurse
            path, used_mask = backtrack_extend(path, used_mask)
            
            # Backtrack
            path.pop()
            used_mask ^= (1 << nb)
            
            # If we found a better solution, we can continue searching
            # But if we've used most of our time, stop
        
        return path, used_mask
    
    def greedy_extend(path, used_mask, randomize=True):
        """Greedily extend a path using Warnsdorff-like heuristic with bitmasks."""
        if time.time() > deadline:
            return path, used_mask
        
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            
            candidates = []
            for nb in neighbors[cur]:
                if (used_mask >> nb) & 1:
                    continue
                mask_others = used_mask ^ (1 << cur)
                if (mask_others & ~compatible_mask[nb]) == 0:
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
            
            candidates.sort(key=lambda x: x[0])
            
            if randomize and len(candidates) > 1:
                k = min(3, len(candidates))
                chosen = random.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
        
    def local_search_improve(path, used_mask, max_iters=10):
        """Try to improve a path by local search: try to extend at ends."""
        if time.time() > deadline:
            return path, used_mask
        
        best = path[:]
        best_mask = used_mask
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(best)
            
            # Try extending the path at either end
            first = best[0]
            for nb in neighbors[first]:
                if (best_mask >> nb) & 1:
                    continue
                mask_others = best_mask ^ (1 << first)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    best = [nb] + best
                    best_mask = best_mask | (1 << nb)
                    improved = True
                    break
            if improved:
                continue
            
            last = best[-1]
            for nb in neighbors[last]:
                if (best_mask >> nb) & 1:
                    continue
                mask_others = best_mask ^ (1 << last)
                if (mask_others & ~compatible_mask[nb]) == 0:
                    best = best + [nb]
                    best_mask = best_mask | (1 << nb)
                    improved = True
                    break
            if improved:
                continue
            
            break
        
        return best, best_mask
    
    # Main search: constraint-propagation backtracking with multiple restarts
    random.seed(12345)
    
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    # Phase 1: Quick greedy restarts to find a good initial solution
    for start_idx in range(min(50, len(start_vertices))):
        if time.time() > deadline - 60:
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = greedy_extend(path, used_mask, randomize=True)
        path, used_mask = local_search_improve(path, used_mask, max_iters=10)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Deep backtracking search from the best starting point
    if best_path and len(best_path) > 0:
        # Use the best path found so far as a seed for backtracking
        # Try backtracking from the start of the best path
        start = best_path[0]
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = backtrack_extend(path, used_mask)
    
    # Phase 3: More backtracking from different starts if time remains
    for start_idx in range(len(start_vertices)):
        if time.time() > deadline - 5:
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = backtrack_extend(path, used_mask)
    
    # Final check
    if not best_path:
        return [0]
    
    return best_path
