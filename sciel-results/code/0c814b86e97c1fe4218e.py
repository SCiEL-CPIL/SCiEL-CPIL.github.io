"""
Snake-in-the-box in Q_9 using Heuristic-Guided Backtracking (Warnsdorff's Rule)
with Randomized Restarts.

This approach treats the problem as finding a long induced path.
We use a DFS strategy where the next vertex is chosen based on Warnsdorff's rule:
prefer vertices with the fewest onward moves (among valid moves).
This helps avoid getting trapped in dead ends early.
We perform multiple randomized restarts to explore different parts of the search space.
"""
import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 145  # seconds, safety margin before 150s kill

# Precompute bit masks for fast XOR
BIT_MASKS = [1 << i for i in range(N)]

def _get_neighbors(v):
    """Yield all neighbors of v in Q_N."""
    for b in BIT_MASKS:
        yield v ^ b

def _construct_snake_warnsdorff(start_vertex, time_budget, randomize_order=False):
    """
    Construct a snake starting from start_vertex using Warnsdorff's heuristic.
    Returns the path found.
    """
    start_time = time.time()
    path = [start_vertex]
    used = {start_vertex}
    
    # blocked[v] is True if v is adjacent to any vertex in path[:-1] 
    # (i.e., adding v to the end would create a chord with a non-consecutive vertex)
    # Actually, it's easier to just check on the fly if we keep the path small, 
    # but for N=9, maintaining a 'forbidden' set is faster.
    
    # Let's maintain a set of 'valid_next' candidates? 
    # No, let's just compute the score for each neighbor on the fly.
    # Score = number of unvisited neighbors that are NOT adjacent to path[:-1].
    
    def is_valid_extension(v, current_path):
        """Check if v can be appended to current_path."""
        if v in used:
            return False
        # Check if v is adjacent to any non-consecutive vertex in current_path
        # Non-consecutive vertices are current_path[:-1]
        for i in range(len(current_path) - 1):
            if (v ^ current_path[i]) & ((v ^ current_path[i]) - 1) == 0: # Hamming distance 1
                return False
        return True

    # To speed up is_valid_extension, we can maintain a set of 'blocked' vertices.
    # A vertex u is 'blocked' for extension if it is adjacent to any vertex in path[:-1].
    # When we append a new vertex w to path, the new 'blocked' set will include neighbors of w 
    # minus the last vertex (which becomes consecutive).
    # However, the 'blocked' set depends on the *entire* path history for the chord constraint.
    # Specifically, if we have path P, and we add w, then for any future vertex x, 
    # x must not be adjacent to any p in P[:-1] (the new non-consecutive set).
    
    # Let's use a simpler but slightly slower check initially, and optimize if needed.
    # Given N=9, path length ~100-200, checking O(N * PathLen) per step is ~1800 ops.
    # 150s / 1800 ops = 83k steps. That's too slow for pure Python if we do many restarts.
    
    # Optimization: Precompute adjacency lists as bitmasks?
    # adj_mask[v] is a bitmask of all neighbors of v.
    # Then v is adjacent to p if adj_mask[v] has bit p set.
    
    global _ADJ_MASK
    if not hasattr(_construct_snake_warnsdorff, '_ADJ_MASK'):
        _ADJ_MASK = [0] * (1 << N)
        for v in range(1 << N):
            mask = 0
            for b in range(N):
                mask |= 1 << (v ^ (1 << b))
            _ADJ_MASK[v] = mask
            _construct_snake_warnsdorff._ADJ_MASK = _ADJ_MASK

    adj_mask = _construct_snake_warnsdorff._ADJ_MASK
    
    # State: path, used_mask (bitmask of used vertices)
    # To check if v is adjacent to any non-consecutive vertex:
    # Non-consecutive vertices are path[:-1].
    # We can maintain a bitmask `non_consec_mask` which has bits set for all vertices in path[:-1].
    # Then v is valid if:
    # 1. v is not in used_mask
    # 2. (adj_mask[v] & non_consec_mask) == 0  (v is not adjacent to any non-consecutive vertex)
    
    used_mask = 1 << start_vertex
    non_consec_mask = 0 # Initially empty
    
    while True:
        if time.time() - start_time > time_budget:
            break
            
        cur = path[-1]
        candidates = []
        
        # Find all valid neighbors of cur
        for b in BIT_MASKS:
            v = cur ^ b
            if used_mask & (1 << v):
                continue
            # Check chord constraint: v must not be adjacent to any vertex in non_consec_mask
            if adj_mask[v] & non_consec_mask:
                continue
            candidates.append(v)
            
        if not candidates:
            break
            
        if randomize_order:
            random.shuffle(candidates)
            
        # Apply Warnsdorff's rule: choose candidate with fewest onward moves
        # Onward moves = number of neighbors of v that are valid extensions from v.
        # A neighbor w of v is a valid extension if:
        # 1. w is not in used_mask | (1 << v)
        # 2. w is not adjacent to (non_consec_mask | (1 << cur)) 
        #    Note: cur becomes a non-consecutive vertex for w.
        #    So the new non_consec_mask for w would be non_consec_mask | (1 << cur).
        
        best_v = None
        best_score = float('inf')
        
        for v in candidates:
            # Calculate score for v
            # New used_mask would be used_mask | (1 << v)
            # New non_consec_mask would be non_consec_mask | (1 << cur)
            
            new_used_mask = used_mask | (1 << v)
            new_non_consec_mask = non_consec_mask | (1 << cur)
            
            score = 0
            for b in BIT_MASKS:
                w = v ^ b
                if new_used_mask & (1 << w):
                    continue
                # Check if w is adjacent to new_non_consec_mask
                if adj_mask[w] & new_non_consec_mask:
                    continue
                score += 1
                
            if score < best_score:
                best_score = score
                best_v = v
                
        # If multiple have same score, pick one randomly to add diversity?
        # Or just pick the first one found with min score.
        # Let's add a small random tie-breaker.
        if best_score < float('inf'):
            # Collect all with best_score
            best_candidates = []
            for v in candidates:
                new_used_mask = used_mask | (1 << v)
                new_non_consec_mask = non_consec_mask | (1 << cur)
                score = 0
                for b in BIT_MASKS:
                    w = v ^ b
                    if new_used_mask & (1 << w):
                        continue
                    if adj_mask[w] & new_non_consec_mask:
                        continue
                    score += 1
                if score == best_score:
                    best_candidates.append(v)
            best_v = random.choice(best_candidates)
            
        # Commit the move
        path.append(best_v)
        used_mask |= (1 << best_v)
        non_consec_mask |= (1 << cur)
        
    return path

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    """
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Try different starting vertices and randomization levels
    # We will run many short searches
    
    while time.time() - start_time < TIME_LIMIT - 2:
        # Choose a random start vertex
        start_v = random.randint(0, (1 << N) - 1)
        
        # Allocate a small time budget for each attempt
        # If we have lots of time left, we can afford longer attempts
        time_left = TIME_LIMIT - (time.time() - start_time)
        # Budget: 1-5 seconds per attempt
        budget = min(5.0, time_left * 0.1) 
        if budget < 0.5:
            budget = 0.5
            
        path = _construct_snake_warnsdorff(start_v, budget, randomize_order=True)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            # If we hit a very long path, maybe we can try to extend it further?
            # The Warnsdorff search usually goes until stuck.
            
    # Final safety check
    if not best_path:
        # Fallback to a simple greedy if something went wrong
        best_path = [0]
        used = {0}
        cur = 0
        for _ in range(100):
            extended = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Quick check
                valid = True
                for p in best_path[:-1]:
                    if bin(v ^ p).count('1') == 1:
                        valid = False
                        break
                if valid:
                    best_path.append(v)
                    used.add(v)
                    cur = v
                    extended = True
                    break
            if not extended:
                break
                
    return best_path
