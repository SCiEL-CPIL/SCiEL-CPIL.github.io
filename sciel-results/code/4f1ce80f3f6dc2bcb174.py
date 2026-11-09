"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import deque

N = 9
TIME_LIMIT = 115  # seconds, leave buffer before 120s

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using a randomized greedy 
    approach with multiple restarts.
    
    The approach models the problem as a constraint satisfaction problem where 
    we extend a path by choosing vertices that satisfy:
    1. Adjacency to the current head.
    2. Non-adjacency to all non-consecutive vertices in the path.
    
    We use a heuristic to choose the next vertex that maximizes the number of 
    valid options for the *next* step (lookahead). We perform many such 
    attempts (restarts) to escape local optima.
    """
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def greedy_search(start_v, start_time):
        """
        Perform a greedy search starting from start_v.
        Returns the path found.
        """
        path = [start_v]
        used = {start_v}
        blocked = set()
        
        while True:
            cur = path[-1]
            
            # Generate candidates
            candidates = []
            for nb in neighbors[cur]:
                if nb in used:
                    continue
                if nb in blocked:
                    continue
                candidates.append(nb)
            
            if not candidates:
                break
            
            # Score candidates
            # Heuristic: Count how many valid moves are available from this candidate
            # for the *next* step.
            # Next step blocked set = blocked U neighbors(path[-2])
            
            prev_prev = path[-2] if len(path) >= 2 else None
            
            def score(v):
                avail = 0
                for nb in neighbors[v]:
                    if nb in used or nb == v:
                        continue
                    if nb in blocked:
                        continue
                    if prev_prev is not None and nb in neighbors[prev_prev]:
                        continue
                    avail += 1
                return avail

            # Sort by score, pick randomly from top 3 to add variation
            candidates.sort(key=score, reverse=True)
            k = min(3, len(candidates))
            v = random.choice(candidates[:k])
            
            # Update state
            path.append(v)
            used.add(v)
            
            # Update blocked set for the NEXT step
            # new_blocked = blocked U neighbors(cur)
            blocked.update(neighbors[cur])
            
            # Time check
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
        return path

    # Main loop: Randomized restarts
    start_time = time.time()
    random.seed(42)
    
    # First, try a deterministic greedy from 0 to get a baseline
    p = greedy_search(0, start_time)
    if len(p) > best_len:
        best_len = len(p)
        best_path = p[:]
        
    # Now, perform random restarts
    while time.time() - start_time < TIME_LIMIT - 1:
        start_v = random.randint(0, (1 << N) - 1)
        p = greedy_search(start_v, start_time)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
    return best_path
# EVOLVE-BLOCK-END
