import time
import random
import sys

N = 9
MAX_V = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    # Precompute neighbors for all vertices (bitmask for O(1) lookup)
    neighbors = [[] for _ in range(MAX_V)]
    neighbor_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 0
        for b in range(N):
            nb = v ^ (1 << b)
            neighbors[v].append(nb)
            mask |= (1 << nb)
        neighbor_mask[v] = mask
    
    best_path = []
    best_len = 0
    
    sys.setrecursionlimit(10000)
    
    def backtrack(path, used, block_count, depth_limit=200):
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if len(path) >= depth_limit:
            return
            
        cur = path[-1]
        
        # Generate candidates with Warnsdorff heuristic (optimized bitmask check)
        candidates = []
        cur_mask = neighbor_mask[cur]
        for n in neighbors[cur]:
            if (cur_mask >> n) & 1:
                if n in used or block_count[n] != 1:
                    continue
                # Count how many valid extensions n has
                deg = 0
                n_mask = neighbor_mask[n]
                for nn in neighbors[n]:
                    if (n_mask >> nn) & 1:
                        if nn != cur and nn not in used and block_count[nn] == 0:
                            deg += 1
                candidates.append((deg, n))
                
        if not candidates:
            return
            
        # Sort by degree (Warnsdorff: prefer lower degree to avoid getting stuck)
        candidates.sort(key=lambda x: x[0])
        
        # Add some randomness to avoid deterministic traps
        if len(candidates) > 1 and random.random() < 0.3:
            k = min(3, len(candidates))
            idx = random.randint(0, k-1)
            chosen = candidates[idx][1]
            for i, (d, n) in enumerate(candidates):
                if n == chosen:
                    candidates[idx], candidates[i] = candidates[i], candidates[idx]
                    break
        
        for deg, n in candidates:
            if time.time() > deadline:
                return
            
            path.append(n)
            used.add(n)
            
            for nn in neighbors[n]:
                block_count[nn] += 1
                
            backtrack(path, used, block_count, depth_limit)
            
            path.pop()
            used.remove(n)
            
            for nn in neighbors[n]:
                block_count[nn] -= 1
                
            if time.time() > deadline:
                return
    
    def local_improve(path):
        """Try to extend the path by finding a valid next vertex."""
        if time.time() > deadline:
            return path
        
        cur = path[-1]
        used = set(path)
        
        block_count = [0] * MAX_V
        for v in path:
            for n in neighbors[v]:
                if n not in used:
                    block_count[n] += 1
        
        for n in neighbors[cur]:
            if n not in used and block_count[n] == 1:
                path.append(n)
                return local_improve(path)
        
        return path
    
    def greedy_extend(path, used, block_count, max_steps=500):
        """Greedy extension using Warnsdorff heuristic without backtracking."""
        nonlocal best_path, best_len, deadline
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
                
            cur = path[-1]
            
            candidates = []
            cur_mask = neighbor_mask[cur]
            for n in neighbors[cur]:
                if (cur_mask >> n) & 1:
                    if n in used or block_count[n] != 1:
                        continue
                    deg = 0
                    n_mask = neighbor_mask[n]
                    for nn in neighbors[n]:
                        if (n_mask >> nn) & 1:
                            if nn != cur and nn not in used and block_count[nn] == 0:
                                deg += 1
                    candidates.append((deg, n))
                    
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0])
            
            # With some probability, pick a random candidate for diversity
            if len(candidates) > 1 and random.random() < 0.2:
                idx = random.randint(0, min(2, len(candidates)-1))
                _, n = candidates[idx]
            else:
                _, n = candidates[0]
            
            path.append(n)
            used.add(n)
            
            for nn in neighbors[n]:
                block_count[nn] += 1
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
        return path
    
    # Strategy: Run backtracking with multiple random restarts
    random.seed(42)
    
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    # Phase 1: Quick greedy extensions from many starting points
    for start_idx in range(min(50, len(start_vertices))):
        if time.time() > deadline - 30:
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used = {start}
        block_count = [0] * MAX_V
        
        for n in neighbors[start]:
            block_count[n] += 1
        
        greedy_extend(path, used, block_count, max_steps=300)
    
    # Phase 2: Backtracking from promising starting points
    for start_idx in range(min(20, len(start_vertices))):
        if time.time() > deadline - 10:
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used = {start}
        block_count = [0] * MAX_V
        
        for n in neighbors[start]:
            block_count[n] += 1
        
        backtrack(path, used, block_count, depth_limit=200)
    
    # Phase 3: Local improvement on best path (both ends, multiple passes)
    if best_path and time.time() < deadline - 10:
        improved = local_improve(best_path[:])
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
    
    # Phase 4: Try to extend from both ends of the best path
    if best_path and time.time() < deadline - 5:
        # Try extending from the beginning
        reversed_path = list(reversed(best_path))
        improved_rev = local_improve(reversed_path)
        if len(improved_rev) > best_len:
            best_path = list(reversed(improved_rev))
            best_len = len(best_path)
    
    # Phase 5: More greedy restarts with different random seeds for diversity
    if best_path and time.time() < deadline - 3:
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            random.seed(1000 + attempt)
            start = random.randint(0, MAX_V - 1)
            path = [start]
            used = {start}
            block_count = [0] * MAX_V
            for n in neighbors[start]:
                block_count[n] += 1
            greedy_extend(path, used, block_count, max_steps=200)
    
    # Phase 6: Final local polish
    if best_path and time.time() < deadline - 1:
        improved = local_improve(best_path[:])
        if len(improved) > best_len:
            best_path = improved
            best_len = len(best_path)
        reversed_path = list(reversed(best_path))
        improved_rev = local_improve(reversed_path)
        if len(improved_rev) > best_len:
            best_path = list(reversed(improved_rev))
            best_len = len(best_path)
    
    if not best_path:
        return [0]
        
    return best_path
