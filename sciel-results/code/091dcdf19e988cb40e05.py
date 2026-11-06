import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
ALL_BITS = [1 << b for b in range(N)]
DEADLINE_FRACTION = 0.85  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _popcount(x):
    return x.bit_count()

def _is_valid_move(path, v):
    """
    Check if adding vertex v to the end of path is valid.
    1. v not in path.
    2. v is not adjacent to any path[i] for i < len(path) - 1.
    """
    if v in path_set:
        return False
    # Check against all non-consecutive vertices
    # The last vertex is path[-1], which is the current position.
    # We need to check against path[0] ... path[-2]
    for i in range(len(path) - 1):
        if _popcount(v ^ path[i]) == 1:
            return False
    return True

# Global state for speed in recursive calls
path_set = set()

def _count_safe_moves(v, path, path_set):
    """
    Count how many valid moves exist from v, assuming we are at the end of the path.
    This is used for Warnsdorff's rule.
    """
    count = 0
    for b in range(N):
        nv = v ^ (1 << b)
        if nv in path_set:
            continue
        # Quick check: nv must not be adjacent to any path[i] for i < len(path)-1
        # Note: nv is a neighbor of v (path[-1]), so it is not adjacent to path[-1] in terms of 
        # the "induced path" constraint for non-consecutive? No, the constraint is on the 
        # vertices themselves.
        # If we move to nv, nv becomes the new last vertex.
        # The constraint is: for all i < len(path)-1, dist(nv, path[i]) != 1.
        # Since dist(nv, v) == 1 and v is path[-1], we only need to check i < len(path)-1.
        
        valid = True
        for i in range(len(path) - 1):
            if _popcount(nv ^ path[i]) == 1:
                valid = False
                break
        if valid:
            count += 1
    return count

def _search_warnsdorff(start, deadline, best_path):
    """
    Perform a DFS search using Warnsdorff's rule for move ordering.
    """
    path = [start]
    path_set = {start}
    
    def dfs():
        nonlocal path, path_set
        
        if time.time() > deadline:
            return
        
        cur = path[-1]
        
        # Generate all valid neighbors
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in path_set:
                continue
            # Check induced constraint
            valid = True
            for i in range(len(path) - 1):
                if _popcount(v ^ path[i]) == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            # Dead end
            if len(path) > len(best_path[0]):
                best_path[0] = list(path)
            return
        
        # Sort candidates by Warnsdorff's rule: fewest onward moves first
        # To add randomness, we can shuffle candidates with the same score
        scored_candidates = []
        for v in candidates:
            # Count safe moves from v
            # Optimization: We can approximate or just count directly.
            # Direct count is O(N * L). L is path length. N=9.
            # This is fast enough.
            score = _count_safe_moves(v, path, path_set)
            scored_candidates.append((score, v))
        
        # Group by score to allow randomization among ties
        # Or just sort and add random noise?
        # Let's sort by score, then shuffle those with equal scores.
        scored_candidates.sort(key=lambda x: x[0])
        
        # To introduce randomness, we can pick randomly from the top K or just shuffle the whole list?
        # Standard Warnsdorff picks the min. Randomized Warnsdorff picks randomly among min-score nodes.
        min_score = scored_candidates[0][0]
        best_moves = [v for score, v in scored_candidates if score == min_score]
        
        # Randomize among best moves
        random.shuffle(best_moves)
        
        for v in best_moves:
            if time.time() > deadline:
                return
            
            path.append(v)
            path_set.add(v)
            
            dfs()
            
            path.pop()
            path_set.remove(v)
            
            if time.time() > deadline:
                return

    dfs()

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION  # 102 seconds
    
    # Initial best path: empty or a small greedy one
    # Let's start with a greedy path from 0 to have a baseline
    def _greedy_start(start):
        path = [start]
        used = {start}
        while True:
            cur = path[-1]
            best_v = None
            min_score = 100
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for i in range(len(path) - 1):
                    if _popcount(v ^ path[i]) == 1:
                        valid = False
                        break
                if valid:
                    score = _count_safe_moves(v, path, used)
                    if score < min_score:
                        min_score = score
                        best_v = v
            if best_v is None:
                break
            path.append(best_v)
            used.add(best_v)
        return path

    # Get a decent initial seed
    initial_seed = _greedy_start(0)
    best_path = [initial_seed]
    
    # Run multiple randomized searches
    # We will try different starting vertices and rely on the randomized Warnsdorff in the DFS
    # Since the search space is huge, random restarts are key.
    
    # To speed up, we can limit the depth of backtracking or just rely on the fact that 
    # Warnsdorff is very effective.
    
    num_restarts = 0
    while time.time() < deadline:
        # Random start vertex
        start = random.randint(0, (1 << N) - 1)
        
        # Call the search
        _search_warnsdorff(start, deadline, best_path)
        
        num_restarts += 1
        # If we found a very long path, we might stop early? 
        # No, keep searching to see if we can beat it.
        
        # Safety check: if we've been running for a while and not improving, 
        # maybe we should try a different strategy?
        # For now, just keep restarting.

    return best_path[0]

# EVOLVE-BLOCK-END
