import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_path = []
    best_len = 0
    
    # State: path (list), used (set), forbidden (set)
    # forbidden: vertices that cannot be the next vertex because they are adjacent to path[i] for i < len(path)-1
    
    def dfs(path, used, forbidden):
        nonlocal best_len, best_path
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        if time.time() > deadline:
            return
        
        cur = path[-1]
        prev = path[-2] if len(path) > 1 else None
        
        # Generate candidates: neighbors of cur
        # A candidate c is valid if:
        # 1. c not in used
        # 2. c not in forbidden
        candidates = []
        for c in NEIGHBORS[cur]:
            if c in used:
                continue
            if c in forbidden:
                continue
            candidates.append(c)
        
        if not candidates:
            return
        
        # Score candidates using Warnsdorff's rule
        # Score = number of valid moves from c
        # If we move to c, the new forbidden set will be forbidden | NEIGHBORS[cur]
        # So a move n from c is valid if:
        # n not in used, n not in forbidden, n not in NEIGHBORS[cur]
        
        new_forbidden_base = forbidden | set(NEIGHBORS[cur])
        
        scored_candidates = []
        for c in candidates:
            count = 0
            for n in NEIGHBORS[c]:
                if n in used:
                    continue
                if n in new_forbidden_base:
                    continue
                count += 1
            scored_candidates.append((count, c))
        
        # Sort by score (fewest onward moves first)
        scored_candidates.sort(key=lambda x: x[0])
        
        # To add randomness, we can shuffle candidates with the same score
        # Or just pick the first one. Let's pick randomly from the top few or all min-score ones.
        min_score = scored_candidates[0][0]
        best_moves = [c for score, c in scored_candidates if score == min_score]
        
        random.shuffle(best_moves)
        
        for c in best_moves:
            if time.time() > deadline:
                return
            
            # Make move
            path.append(c)
            used.add(c)
            
            # New forbidden set: old forbidden + neighbors of cur
            # Note: cur becomes a non-consecutive vertex
            new_forbidden = forbidden | set(NEIGHBORS[cur])
            
            dfs(path, used, new_forbidden)
            
            # Undo move
            path.pop()
            used.remove(c)
            
            if time.time() > deadline:
                return

    # Get a decent initial seed using a simple greedy approach
    def greedy_extend(start_v):
        path = [start_v]
        used = {start_v}
        forbidden = set()
        
        while True:
            cur = path[-1]
            candidates = []
            for c in NEIGHBORS[cur]:
                if c in used or c in forbidden:
                    continue
                candidates.append(c)
            
            if not candidates:
                break
            
            # Pick random candidate for variety in seed
            next_v = random.choice(candidates)
            path.append(next_v)
            used.add(next_v)
            forbidden |= set(NEIGHBORS[cur])
            
        return path

    # Initial seed
    p = greedy_extend(0)
    best_len = len(p)
    best_path = p
    
    # Run multiple DFS searches from different starts
    num_restarts = 0
    while time.time() < deadline:
        start = random.randint(0, SIZE - 1)
        
        # Start DFS
        path = [start]
        used = {start}
        forbidden = set()
        
        dfs(path, used, forbidden)
        
        num_restarts += 1
        
        # Optional: If we have time, try to improve the best path with local search?
        # For now, just keep restarting. The efficient DFS should find good paths.
        
    return best_path
