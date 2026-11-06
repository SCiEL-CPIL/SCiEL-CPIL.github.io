import time
import random
import sys
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # 5s safety margin before 120s limit
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
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
        
        # Generate candidates with Warnsdorff heuristic
        candidates = []
        for n in neighbors[cur]:
            if n not in used and block_count[n] == 1:
                # Count how many valid extensions n has (its "degree" in remaining graph)
                deg = 0
                for nn in neighbors[n]:
                    if nn not in used and nn != cur:
                        if block_count[nn] == 0:
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
            
            # Add n
            path.append(n)
            used.add(n)
            
            # Update block counts for neighbors of n
            for nn in neighbors[n]:
                block_count[nn] += 1
                
            backtrack(path, used, block_count, depth_limit)
            
            # Remove n
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
        
        # Compute block counts
        block_count = [0] * MAX_V
        for v in path:
            for n in neighbors[v]:
                if n not in used:
                    block_count[n] += 1
        
        # Try to find a valid extension
        for n in neighbors[cur]:
            if n not in used and block_count[n] == 1:
                path.append(n)
                return local_improve(path)
        
        return path
    
    def try_extend_both_ends(path):
        """Try to extend the path from both ends."""
        if time.time() > deadline:
            return path
        
        used = set(path)
        
        # Compute block counts
        block_count = [0] * MAX_V
        for v in path:
            for n in neighbors[v]:
                if n not in used:
                    block_count[n] += 1
        
        # Try to extend from the end
        extended = False
        cur = path[-1]
        for n in neighbors[cur]:
            if n not in used and block_count[n] == 1:
                path.append(n)
                used.add(n)
                for nn in neighbors[n]:
                    if nn not in used:
                        block_count[nn] += 1
                extended = True
                break
        
        if not extended:
            # Try to extend from the beginning
            cur = path[0]
            for n in neighbors[cur]:
                if n not in used and block_count[n] == 1:
                    path.insert(0, n)
                    used.add(n)
                    for nn in neighbors[n]:
                        if nn not in used:
                            block_count[nn] += 1
                    extended = True
                    break
        
        if extended:
            return try_extend_both_ends(path)
        
        return path
    
    # Strategy: Run backtracking with multiple random restarts
    # Use different starting vertices and random seeds
    
    random.seed(42)
    
    # Try multiple starting points
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    max_starts = 20  # Limit number of restarts
    
    for start_idx in range(min(max_starts, len(start_vertices))):
        if time.time() > deadline:
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used = {start}
        block_count = [0] * MAX_V
        
        # Initialize block counts for start vertex
        for n in neighbors[start]:
            block_count[n] += 1
        
        # Run backtracking
        backtrack(path, used, block_count)
        
        # Try to locally improve the best path found from both ends
        if best_path and time.time() < deadline - 5:
            improved = try_extend_both_ends(best_path[:])
            if len(improved) > best_len:
                best_path = improved
                best_len = len(best_path)
    
    # Ensure we return a valid path
    if not best_path:
        return [0]
        
    return best_path
