"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Precompute adjacency list
_N = P['n']
_ADJ = [set() for _ in range(_N)]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)

# Convert to frozensets for faster hashing in memoization
_ADJ_FROZEN = [frozenset(s) for s in _ADJ]

def _dfs_longest(start, deadline):
    """Find longest path starting from 'start' using DFS with pruning."""
    best_path = [start]
    
    # Use a recursive approach with memoization on (current, visited_set)
    # But visited_set is too large for memoization. Instead, use iterative deepening
    # or a simple DFS that tries all branches but prunes when current length + remaining < best.
    
    # For n=100, simple DFS might be too slow without good pruning.
    # Let's use a randomized restart approach with greedy + local search.
    
    # Actually, for this specific graph, let's try a more robust heuristic.
    # The graph is sparse (avg degree ~3.6). Longest path is likely near n.
    
    # Strategy: 
    # 1. Try to extend a path from both ends greedily.
    # 2. Use random restarts with different initial seeds.
    # 3. Use a "dead-end" avoidance heuristic.
    
    # Let's implement a simple iterative improvement:
    # Start with a greedy path, then try to swap or extend.
    
    # Actually, let's stick to a robust DFS with time limit for now, 
    # but optimized for the specific structure.
    
    # Given the time limit, a full exhaustive search is impossible.
    # We will use a randomized local search approach.
    
    # Let's implement a function that tries to extend a path from both ends.
    
    def _try_extend(path, seen):
        """Try to extend path from both ends greedily."""
        # Extend from end
        while True:
            candidates = _ADJ[path[-1]] - seen
            if not candidates:
                break
            # Choose candidate that has most unvisited neighbors (greedy)
            best_v = min(candidates, key=lambda v: (len(_ADJ[v] - seen), v))
            path.append(best_v)
            seen.add(best_v)
        
        # Extend from start
        while True:
            candidates = _ADJ[path[0]] - seen
            if not candidates:
                break
            best_v = min(candidates, key=lambda v: (len(_ADJ[v] - seen), v))
            path.insert(0, best_v)
            seen.add(best_v)
        
        return path, seen

    # Randomized restarts
    best_len = 0
    best_path = None
    
    rng = random.Random(748)
    
    for _ in range(1000):  # Up to 1000 restarts
        if time.time() > deadline:
            break
        
        # Random start
        start = rng.randint(0, _N - 1)
        path = [start]
        seen = {start}
        
        # Greedy extension from both ends
        path, seen = _try_extend(path, seen)
        
        # Try local improvement: swap internal nodes? Too complex for now.
        # Just try to extend again with different tie-breaking?
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
            # If we found a Hamiltonian path, we're done
            if best_len == _N:
                break
    
    return best_path if best_path else [0]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # First, try a deterministic greedy approach to get a baseline
    best = []
    for start in range(_N):
        path = [start]
        seen = {start}
        while _ADJ[path[-1]] - seen:
            v = min(_ADJ[path[-1]] - seen, key=lambda w: (len(_ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
        
        # Try to extend from the start as well
        while _ADJ[path[0]] - seen:
            v = min(_ADJ[path[0]] - seen, key=lambda w: (len(_ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            
        if len(path) > len(best):
            best = path
    
    # Now try randomized search to improve
    if len(best) < _N:
        improved = _dfs_longest(0, deadline)
        if len(improved) > len(best):
            best = improved
    
    return best
# EVOLVE-BLOCK-END
