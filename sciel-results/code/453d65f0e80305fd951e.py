import time
import random
import sys

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    sys.setrecursionlimit(10000)
    
    def backtrack(path, used_mask, depth_limit=200):
        nonlocal best_path, best_len, deadline
        
        # Check time
        if time.time() > deadline:
            return
            
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Stop if we hit the depth limit
        if len(path) >= depth_limit:
            return
            
        cur = path[-1]
        
        # Generate candidates
        # A candidate n is valid if:
        # 1. It is not used
        # 2. It is not adjacent to any vertex in the path except the current one (cur)
        #    This is equivalent to: n is not in the "block" set.
        #    The "block" set contains all neighbors of used vertices.
        #    However, maintaining a separate block count array is expensive.
        #    Instead, we can check: for a neighbor n of cur, is n valid?
        #    n is valid if n is not used AND n is not adjacent to any other used vertex.
        #    Wait, the condition for an induced path is that non-consecutive vertices differ in at least 2 bits.
        #    This means n cannot be adjacent to any vertex in path[:-1].
        
        # Efficient check:
        # n is a neighbor of cur.
        # We need to ensure n is not a neighbor of any v in path[:-1].
        # Let's precompute a bitmask of "blocked" vertices? No, that's MAX_V size.
        # Let's just check the neighbors of n against the path.
        # Since N=9, the number of neighbors is small.
        # But checking against the whole path is O(L).
        
        # Better approach:
        # Maintain a bitmask `used_mask`.
        # A vertex n is a valid extension if:
        # 1. (used_mask >> n) & 1 == 0
        # 2. No neighbor of n (except cur) is in used_mask.
        #    i.e., for all m in neighbors[n], if m != cur and (used_mask >> m) & 1: invalid.
        
        candidates = []
        
        for n in neighbors[cur]:
            # Check if n is used
            if (used_mask >> n) & 1:
                continue
            
            # Check if n is adjacent to any other used vertex
            valid = True
            for m in neighbors[n]:
                if m == cur:
                    continue
                if (used_mask >> m) & 1:
                    valid = False
                    break
            
            if valid:
                # Heuristic: count how many valid extensions n has
                # This is the number of neighbors of n that are not used and not adjacent to other used vertices.
                # This is expensive to compute exactly.
                # Approximation: count neighbors of n that are not used.
                deg = 0
                for m in neighbors[n]:
                    if m != cur and not ((used_mask >> m) & 1):
                        deg += 1
                candidates.append((deg, n))
                
        if not candidates:
            return
            
        # Sort by degree (Warnsdorff: prefer lower degree)
        candidates.sort(key=lambda x: x[0])
        
        # Add randomness
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
            new_mask = used_mask | (1 << n)
            
            backtrack(path, new_mask, depth_limit)
            
            path.pop()
            
            if time.time() > deadline:
                return
    
    def greedy_extend(path, used_mask):
        """Greedy extension using Warnsdorff heuristic without backtracking."""
        nonlocal best_path, best_len, deadline
        
        while True:
            if time.time() > deadline:
                break
                
            cur = path[-1]
            
            candidates = []
            for n in neighbors[cur]:
                if (used_mask >> n) & 1:
                    continue
                
                valid = True
                for m in neighbors[n]:
                    if m == cur:
                        continue
                    if (used_mask >> m) & 1:
                        valid = False
                        break
                
                if valid:
                    deg = 0
                    for m in neighbors[n]:
                        if m != cur and not ((used_mask >> m) & 1):
                            deg += 1
                    candidates.append((deg, n))
                    
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0])
            
            # Pick the best one
            _, n = candidates[0]
            
            path.append(n)
            used_mask |= (1 << n)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
        return path

    # Strategy: Run backtracking with multiple random restarts
    random.seed(42)
    
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    # Phase 1: Quick greedy extensions from many starting points to find a good initial path
    for start_idx in range(min(100, len(start_vertices))):
        if time.time() > deadline - 60:  # Reserve 60s for backtracking
            break
        
        start = start_vertices[start_idx]
        
        path = [start]
        used_mask = 1 << start
        
        greedy_extend(path, used_mask)
        
        if time.time() > deadline - 60:
            break

    # Phase 2: Backtracking from the best path found so far and some random starts
    # We will try to extend the best path found in Phase 1 using backtracking
    if best_path:
        # Use the best path as a prefix for backtracking?
        # Or just start backtracking from the end of the best path?
        # Let's try backtracking from the end of the best path.
        # This is risky if the best path is a dead end.
        # Better: Run independent backtracking searches from random starts, 
        # but stop early if we find a path longer than the current best.
        
        # Let's run backtracking from a few promising starts.
        # The most promising starts are those that led to the longest greedy paths.
        # We didn't track which start led to the best path.
        # Let's just run backtracking from a few random starts.
        
        # To make backtracking faster, we can limit the depth or use a time slice per start.
        # Let's allocate the remaining time to backtracking.
        
        remaining_starts = start_vertices[100:150]
        
        for start in remaining_starts:
            if time.time() > deadline - 5:
                break
            
            path = [start]
            used_mask = 1 << start
            
            # Run backtracking with a time check inside
            backtrack(path, used_mask, depth_limit=200)
            
            if time.time() > deadline - 5:
                break

    # Phase 3: Local improvement on best path
    # Try to extend the best path found by backtracking
    if best_path and time.time() < deadline - 2:
        # Create a copy for greedy extension
        improved_path = best_path[:]
        used_mask = 0
        for v in improved_path:
            used_mask |= (1 << v)
        
        greedy_extend(improved_path, used_mask)
        
        if len(improved_path) > best_len:
            best_path = improved_path
            best_len = best_len # already updated in greedy_extend

    if not best_path:
        return [0]
        
    return best_path
