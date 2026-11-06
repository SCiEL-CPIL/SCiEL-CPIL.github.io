"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
import numpy as np

# Precompute neighbor masks for O(1) forbidden set updates
# neighbors_mask[v] is an integer with bits set for all neighbors of v
neighbors_mask = [0] * TOTAL_VERTICES
neighbors_list = [[] for _ in range(TOTAL_VERTICES)]
for v in range(TOTAL_VERTICES):
    mask = 0
    for b in range(N):
        u = v ^ (1 << b)
        mask |= (1 << u)
        neighbors_list[v].append(u)
    neighbors_mask[v] = mask

def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _backtracking_search(deadline, max_depth=None):
    """
    Constraint Satisfaction Problem (CSP) solver using Backtracking with 
    Forward Checking and Most-Constrained-Value (MCV) heuristic.
    """
    rng = random.Random(42)
    
    # State: path, used_mask, forbidden_mask
    # We'll use a stack-based DFS to avoid recursion limits
    
    best_path = []
    best_len = 0
    
    # Start with a random vertex
    start_vertex = rng.randint(0, TOTAL_VERTICES - 1)
    
    # Stack entries: (path, used_mask, forbidden_mask, last_vertex, candidates)
    # candidates is a list of valid next vertices for the last vertex
    
    def get_valid_candidates(last_vertex, used_mask, forbidden_mask):
        """Get valid next vertices for last_vertex."""
        candidates = []
        for v in neighbors_list[last_vertex]:
            bit = 1 << v
            if (used_mask & bit) == 0 and (forbidden_mask & bit) == 0:
                candidates.append(v)
        return candidates
    
    # MCV heuristic: sort candidates by number of further valid moves (ascending)
    # This is the "most constrained" value - vertices with fewer options should be tried first
    def score_candidate(v, used_mask, forbidden_mask, last_vertex):
        """Score a candidate by how constrained it is (fewer options = higher priority)."""
        new_used = used_mask | (1 << v)
        new_forbidden = forbidden_mask | neighbors_mask[last_vertex]
        
        # Count how many valid moves v would have
        count = 0
        for w in neighbors_list[v]:
            bit = 1 << w
            if (new_used & bit) == 0 and (new_forbidden & bit) == 0:
                count += 1
        return count
    
    # Initialize stack
    initial_candidates = get_valid_candidates(start_vertex, 1 << start_vertex, 0)
    if not initial_candidates:
        return [start_vertex]
    
    # Sort initial candidates by MCV
    initial_candidates.sort(key=lambda v: score_candidate(v, 1 << start_vertex, 0, start_vertex))
    
    stack = [([start_vertex], 1 << start_vertex, 0, start_vertex, initial_candidates)]
    
    while stack and time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        path, used_mask, forbidden_mask, last_vertex, candidates = stack.pop()
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if not candidates:
            continue
            
        if max_depth and len(path) >= max_depth:
            continue
        
        # Try candidates in MCV order (most constrained first)
        # We need to re-sort each time because the state changes
        scored_candidates = []
        for v in candidates:
            s = score_candidate(v, used_mask, forbidden_mask, last_vertex)
            scored_candidates.append((s, v))
        
        scored_candidates.sort(key=lambda x: x[0])  # Ascending: most constrained first
        
        # Push to stack in reverse order so most constrained is popped first
        for s, v in reversed(scored_candidates):
            new_path = path + [v]
            new_used = used_mask | (1 << v)
            new_forbidden = forbidden_mask | neighbors_mask[last_vertex]
            
            # Get valid candidates for the new state
            new_candidates = get_valid_candidates(v, new_used, new_forbidden)
            
            stack.append((new_path, new_used, new_forbidden, v, new_candidates))
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_path = []
    best_len = 0
    
    # Try multiple starting vertices and different random seeds
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Change the random seed for diversity
        random.seed(trial * 1000 + 42)
        
        path = _backtracking_search(deadline, max_depth=200)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        # Also try a simpler greedy approach as fallback
        if time.time() > deadline - 10:
            break
    
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback: simple greedy from vertex 0
    path, used = [0], 1
    forbidden = 0
    while True:
        cur = path[-1]
        nxt = None
        for v in neighbors_list[cur]:
            bit = 1 << v
            if (used & bit) == 0 and (forbidden & bit) == 0:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used |= (1 << nxt)
        forbidden |= neighbors_mask[cur]

# EVOLVE-BLOCK-END
