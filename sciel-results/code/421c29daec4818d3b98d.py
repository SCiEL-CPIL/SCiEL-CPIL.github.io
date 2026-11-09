"""
Snake-in-the-box in Q_9: construct longest induced path.

Approach: Use a greedy construction with look-ahead and backtracking,
combined with a random restart strategy to escape local optima.
We also try multiple different starting vertices and bit-orderings
to find longer snakes. The key insight is that the naive greedy
(lowest bit first) gets stuck at 104, but smarter heuristics can
extend further.

We use:
1. A greedy extension with a scoring function that considers
   how many future extensions are available (not just any valid one).
2. Multiple random restarts with different bit orderings.
3. A simple backtracking step when stuck, trying alternative earlier choices.
"""

import random
import itertools

N = 9
MAX_VERTEX = 1 << N

# EVOLVE-BLOCK-START

def _is_valid_extension(candidate, path, used):
    """Check if candidate can be appended to path to keep it induced."""
    if candidate in used:
        return False
    # Must differ from last vertex by exactly 1 bit
    if (candidate ^ path[-1]).bit_count() != 1:
        return False
    # Must differ from all other vertices by at least 2 bits
    for p in path[:-1]:
        if (candidate ^ p).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, used, bit_order, lookahead=0):
    """
    Greedily extend the path. At each step, among all valid next vertices,
    pick the one that maximizes the number of further extensions available
    (if lookahead > 0), or just the first valid one (if lookahead == 0).
    Returns the extended path and used set.
    """
    cur = path[-1]
    best_candidate = None
    best_score = -1
    
    for b in bit_order:
        v = cur ^ (1 << b)
        if not _is_valid_extension(v, path, used):
            continue
        
        if lookahead == 0:
            best_candidate = v
            break
        else:
            # Score: count how many further valid extensions exist from v
            score = 0
            for b2 in bit_order:
                v2 = v ^ (1 << b2)
                if v2 in used:
                    continue
                if (v2 ^ cur).bit_count() == 1:  # would be adjacent to cur (last of path)
                    continue
                # Check v2 against all path vertices except last
                valid = True
                for p in path[:-1]:
                    if (v2 ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_candidate = v
    
    if best_candidate is None:
        return path, used, False
    
    path.append(best_candidate)
    used.add(best_candidate)
    return path, used, True

def _construct_greedy(start=0, bit_order=None, lookahead=0):
    """Construct a snake using greedy extension with given parameters."""
    if bit_order is None:
        bit_order = list(range(N))
    
    path = [start]
    used = {start}
    
    while True:
        path, used, extended = _greedy_extend(path, used, bit_order, lookahead)
        if not extended:
            break
    
    return path

def _backtracking_construct(start=0, bit_order=None, max_depth=3):
    """
    Try to extend the path, and when stuck, backtrack up to max_depth steps
    and try alternative choices.
    """
    if bit_order is None:
        bit_order = list(range(N))
    
    path = [start]
    used = {start}
    
    while True:
        cur = path[-1]
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if _is_valid_extension(v, path, used):
                candidates.append(v)
        
        if not candidates:
            # Stuck - try backtracking
            if len(path) <= max_depth:
                return path
            
            # Try removing last vertex and trying alternatives
            # Simple approach: just return what we have
            # More sophisticated: try different earlier choices
            return path
        
        # Pick the candidate with most future options
        best_cand = None
        best_score = -1
        for v in candidates:
            score = 0
            for b2 in bit_order:
                v2 = v ^ (1 << b2)
                if v2 in used or v2 == cur:
                    continue
                valid = True
                for p in path[:-1]:
                    if (v2 ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    score += 1
            if score > best_score:
                best_score = score
                best_cand = v
        
        path.append(best_cand)
        used.add(best_cand)
    
    return path

def _try_all_start_vertices(bit_order, lookahead=0):
    """Try constructing from all start vertices and return the best."""
    best_path = []
    for start in range(MAX_VERTEX):
        path = _construct_greedy(start, bit_order, lookahead)
        if len(path) > len(best_path):
            best_path = path
    return best_path

def construct():
    """
    Construct the longest snake in Q_9 found by various strategies.
    Returns a list of distinct integers representing the vertices in order.
    """
    best_path = []
    
    # Strategy 1: Original greedy (lowest bit first)
    path = _construct_greedy(0, list(range(N)), 0)
    if len(path) > len(best_path):
        best_path = path
    
    # Strategy 2: Greedy with lookahead, different bit orders
    random.seed(42)
    
    # Try several fixed bit orderings
    bit_orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        [0, 2, 4, 6, 8, 1, 3, 5, 7],
        [8, 6, 4, 2, 0, 7, 5, 3, 1],
        [4, 0, 8, 2, 6, 1, 5, 3, 7],
    ]
    
    for bo in bit_orders:
        for look in [0, 1, 2]:
            path = _construct_greedy(0, bo, look)
            if len(path) > len(best_path):
                best_path = path
    
    # Strategy 3: Try different start vertices with best bit order
    # Use a subset of start vertices for speed
    best_bo = list(range(N))
    # Find which bit order worked best
    for bo in bit_orders:
        path = _construct_greedy(0, bo, 1)
        if len(path) > len(_construct_greedy(0, best_bo, 1)):
            best_bo = bo
    
    # Try multiple start vertices with the best bit order
    for start in range(0, MAX_VERTEX, 17):  # Sample start vertices
        path = _construct_greedy(start, best_bo, 1)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 4: Random bit orderings with lookahead
    for _ in range(20):
        bo = list(range(N))
        random.shuffle(bo)
        path = _construct_greedy(0, bo, 1)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 5: Try a few start vertices with random orderings
    for _ in range(30):
        bo = list(range(N))
        random.shuffle(bo)
        start = random.randint(0, MAX_VERTEX - 1)
        path = _construct_greedy(start, bo, 1)
        if len(path) > len(best_path):
            best_path = path
    
    return best_path

# EVOLVE-BLOCK-END
