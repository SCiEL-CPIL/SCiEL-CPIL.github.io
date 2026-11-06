import time
import random
import sys
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    # Precompute neighbors
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
                deg = 0
                for nn in neighbors[n]:
                    if nn not in used and nn != cur:
                        if block_count[nn] == 0:
                            deg += 1
                candidates.append((deg, n))
                
        if not candidates:
            return
        
        # Sort by degree (Warnsdorff)
        candidates.sort(key=lambda x: x[0])
        
        # Add randomness
        if len(candidates) > 1 and rng.random() < 0.3:
            k = min(3, len(candidates))
            idx = rng.randint(0, k-1)
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
    
    def greedy_extend(path, used, bit_order):
        while time.time() < deadline:
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path
    
    # Run backtracking with multiple random restarts
    start_vertices = list(range(MAX_V))
    rng.shuffle(start_vertices)
    
    max_starts = 20
    
    for start_idx in range(min(max_starts, len(start_vertices))):
        if time.time() > deadline:
            break
        
        start = start_vertices[start_idx]
        
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        
        path = [start]
        used = {start}
        block_count = [0] * MAX_V
        
        for n in neighbors[start]:
            block_count[n] += 1
        
        backtrack(path, used, block_count)
        
        if best_path and time.time() < deadline - 5:
            improved = local_improve(best_path[:])
            if len(improved) > best_len:
                best_path = improved
                best_len = len(best_path)
    
    # Greedy restarts
    if time.time() < deadline - 10:
        for _ in range(1000):
            if time.time() >= deadline - 5:
                break
            
            bit_order = list(range(N))
            rng.shuffle(bit_order)
            
            start = rng.randint(0, MAX_V - 1)
            path = [start]
            used = {start}
            path = greedy_extend(path, used, bit_order)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    if not best_path:
        return [0]
        
    return best_path
