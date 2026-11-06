import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    n = 256
    
    # Perfect squares up to 2*256 = 512
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Step 1: Remove numbers where 2*a is a perfect square (diagonal constraint)
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
    
    cand_set = set(candidates)
    
    # Build conflict graph
    # conflict[a] is a set of b in candidates such that a+b is a square
    conflict = {a: set() for a in candidates}
    
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Helper to check if a set is valid (should be true by construction, but for safety)
    def is_valid(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if S_list[i] + S_list[j] in squares:
                    return False
        for a in S:
            if 2*a in squares:
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Local Search Function
    def local_search(initial_set):
        S = set(initial_set)
        # Track time locally
        local_start = time.time()
        
        improved = True
        while improved and (time.time() - local_start) < 2.0: # Limit each LS run to 2s
            improved = False
            
            # Phase 1: Try to add any node that has no conflicts with S
            added = False
            # Randomize order to escape local minima
            to_check = [v for v in candidates if v not in S]
            random.shuffle(to_check)
            for v in to_check:
                if not (conflict[v] & S):
                    S.add(v)
                    improved = True
                    added = True
                    break
            
            if added:
                continue # Restart the loop to check for more adds
            
            # Phase 2: Try 1-1 swaps
            # Find a node u in S and v not in S such that v's only neighbor in S is u.
            # Then we can swap u for v.
            # To avoid cycling, we might track recent states or just randomize.
            
            # Pre-compute neighbors in S for all v not in S? 
            # It's O(N * |S|). N=256, |S|~100. 25k ops. Fast.
            
            swaps = []
            for v in candidates:
                if v in S:
                    continue
                # Count neighbors in S
                neighbors_in_S = [u for u in conflict[v] if u in S]
                if len(neighbors_in_S) == 1:
                    u = neighbors_in_S[0]
                    swaps.append((u, v))
            
            if swaps:
                # Pick a random swap
                u, v = random.choice(swaps)
                S.remove(u)
                S.add(v)
                improved = True
                continue
        
        return S

    # Run multiple restarts
    while time.time() < deadline:
        # Initialize with a random greedy set
        S = set()
        order = candidates[:]
        random.shuffle(order)
        for v in order:
            if not (conflict[v] & S):
                S.add(v)
        
        # Local search
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            # Optional: if we find a very good one, we might want to spend more time polishing it?
            # For now, just keep updating best.

    result = sorted(best_set)
    
    # Ensure nonempty
    if not result:
        # Fallback: find any valid single element
        for a in range(1, n + 1):
            if 2 * a not in squares:
                result = [a]
                break
    
    return result
