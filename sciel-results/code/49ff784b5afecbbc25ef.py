"""
Snake-in-the-box Q_9: Backtracking search with Warnsdorff heuristic and bitmask state.
"""
import time
import random
import sys

N = 9
TOTAL_VERTICES = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(TOTAL_VERTICES)]
for v in range(TOTAL_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ BIT_MASKS[b])

# Precompute neighbor masks for fast "is neighbor" checks
# NEIGHBOR_MASK[v] is a bitmask where bit i is set if i is a neighbor of v
NEIGHBOR_MASK = [0] * TOTAL_VERTICES
for v in range(TOTAL_VERTICES):
    mask = 0
    for nb in NEIGHBORS[v]:
        mask |= (1 << nb)
    NEIGHBOR_MASK[v] = mask

def construct():
    """
    Construct the longest induced path (snake) in the hypercube Q_N in order.
    Returns a list of distinct integers.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_path = [0]
    best_len = 1
    
    # Symmetry breaking: The first step can be assumed to flip bit 0.
    # So we start with path [0, 1].
    # Vertex 0 is used. Vertex 1 is used.
    # Vertex 1 is the current head.
    # Vertices adjacent to 0 (except 1) are blocked for future non-consecutive adjacency?
    # Actually, the rule is: if we add v, all neighbors of v (except prev) become blocked.
    # Initially:
    # Path: [0, 1]
    # Used: {0, 1}
    # Blocked: Neighbors of 1 (excluding 0) AND Neighbors of 0 (excluding 1)?
    # Let's trace the constraint carefully.
    # Constraint: For any i < j, if |i-j| > 1, dist(path[i], path[j]) > 1.
    # This is equivalent to: path[j] cannot be a neighbor of any path[i] where i < j-1.
    
    # Let's maintain:
    # used_mask: bits set for vertices in path.
    # blocked_mask: bits set for vertices that CANNOT be the next vertex because they are adjacent to some path[i] where i < len(path)-1.
    
    # Initial state:
    # Path: [0, 1]
    # Current head: 1
    # Used: 0, 1
    # Blocked: 
    # When we added 1 (from 0), the neighbors of 1 (except 0) become blocked for *future* steps?
    # No, the constraint is that the *next* vertex cannot be adjacent to any *previous* vertex except the current one.
    # So, if we are at head $h$, and consider next vertex $v$:
    # $v$ must not be in `used`.
    # $v$ must not be adjacent to any $p_i$ where $i < \text{len}(path)-1$.
    # The set of vertices adjacent to any $p_i$ ($i < \text{len}(path)-1$) is exactly the `blocked` set we maintain.
    
    # Initialization for [0, 1]:
    # Previous vertex was 0. Current is 1.
    # Vertices adjacent to 0 (except 1) are now "past" and cannot be adjacent to future vertices.
    # So, neighbors of 0 (excluding 1) are added to `blocked`.
    # Also, when we move to 1, do we block neighbors of 1?
    # No, neighbors of 1 are valid candidates for the NEXT step, provided they don't conflict with earlier vertices.
    # But once we move from 1 to $v$, $v$ becomes the head. Then neighbors of 1 (excluding $v$) will be blocked for the step *after* that.
    # So, `blocked` should contain neighbors of all $p_i$ where $i < \text{len}(path)-1$.
    
    # Start:
    # p_0 = 0, p_1 = 1.
    # Head is 1.
    # Indices < len(path)-1 = 1. So index 0 (vertex 0) is "past".
    # Neighbors of 0 (excluding 1) are blocked.
    
    initial_used = (1 << 0) | (1 << 1)
    initial_blocked = 0
    for nb in NEIGHBORS[0]:
        if nb != 1:
            initial_blocked |= (1 << nb)
            
    initial_path = [0, 1]
    
    # We will run multiple DFS searches with different random seeds for tie-breaking
    # to explore different parts of the search space.
    
    sys.setrecursionlimit(10000)
    
    def dfs(path, cur, used, blocked, depth, rng):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
            
        if depth > best_len:
            best_len = depth
            best_path = list(path)
            
        # Generate candidates
        # A vertex v is a candidate if:
        # 1. v is a neighbor of cur
        # 2. v is not in used
        # 3. v is not in blocked
        
        candidates = []
        for v in NEIGHBORS[cur]:
            bit_v = 1 << v
            if (used & bit_v) == 0 and (blocked & bit_v) == 0:
                candidates.append(v)
                
        if not candidates:
            return
            
        # Heuristic: Warnsdorff's Rule
        # Score each candidate by the number of its valid neighbors (future options).
        # We want to pick the one with the fewest options to avoid getting stuck.
        
        def get_score(v):
            # New state if we go to v:
            # new_cur = v
            # new_used = used | (1<<v)
            # new_blocked = blocked | (neighbors of cur excluding v)
            # Wait, the blocked set update:
            # The vertex `cur` becomes "past". So its neighbors (excluding v) become blocked.
            # So new_blocked = blocked | (NEIGHBOR_MASK[cur] & ~(1<<v) & ~used)
            # Actually, we just OR with the mask of neighbors of cur, excluding v.
            # We don't need to mask with ~used because those are already in used or blocked?
            # If a neighbor of cur is in used, it's already handled.
            # If a neighbor of cur is in blocked, it's already blocked.
            # So new_blocked = blocked | (NEIGHBOR_MASK[cur] & ~(1<<v))
            
            nb_mask = NEIGHBOR_MASK[cur] & ~(1 << v)
            new_blocked = blocked | nb_mask
            new_used = used | (1 << v)
            
            # Count how many neighbors of v are available in the new state
            # Available: not in new_used, not in new_blocked
            avail_mask = NEIGHBOR_MASK[v] & ~new_used & ~new_blocked
            return avail_mask.bit_count()
        
        # Calculate scores
        scored_candidates = []
        for v in candidates:
            s = get_score(v)
            scored_candidates.append((s, v))
            
        # Sort by score (ascending)
        scored_candidates.sort(key=lambda x: x[0])
        
        # Randomization: To break symmetries and explore different paths,
        # we shuffle candidates that have the same score, or pick randomly from the top few if time allows.
        # A simple approach: shuffle the list, then sort stably by score.
        # Or: group by score, shuffle within groups.
        # Given small N, simple shuffle then sort is fine, but might be biased if scores are unique.
        # Better: shuffle candidates first, then sort by score (stable sort preserves shuffle order for ties).
        
        # We already have scored_candidates. Let's re-shuffle the original candidates and re-score? No, too slow.
        # Let's just shuffle the `candidates` list before scoring.
        # Re-implementing the loop for clarity and randomness:
        
        candidates_shuffled = candidates[:]
        rng.shuffle(candidates_shuffled)
        
        scored_candidates = []
        for v in candidates_shuffled:
            s = get_score(v)
            scored_candidates.append((s, v))
        
        # Stable sort by score
        scored_candidates.sort(key=lambda x: x[0])
        
        for s, v in scored_candidates:
            if time.time() > deadline:
                return
            
            path.append(v)
            
            # Update state
            new_used = used | (1 << v)
            # cur becomes past, so its neighbors (except v) are blocked
            new_blocked = blocked | (NEIGHBOR_MASK[cur] & ~(1 << v))
            
            dfs(path, v, new_used, new_blocked, depth + 1, rng)
            
            path.pop()
            
            if time.time() > deadline:
                return
            
            # Optional: If we found a very long path, we might stop early or continue to find even longer ones.
            # We continue to try to improve best_len.
            
    # Run multiple independent searches
    # We use different random seeds to explore different branches.
    num_searches = 50
    for i in range(num_searches):
        if time.time() > deadline:
            break
            
        rng = random.Random(i)
        # Start from the symmetry-broken state
        current_path = list(initial_path)
        
        dfs(current_path, 1, initial_used, initial_blocked, 2, rng)
        
        # If we have a lot of time left, we can try restarting from different symmetry breaks?
        # The problem has high symmetry (hypercube automorphism group).
        # Fixing first step to bit 0 is a valid symmetry break.
        # We don't need to try other first steps.
        
    # Final validation
    if not _is_valid_snake(best_path):
        # Fallback to a known valid short snake if something went wrong
        best_path = [0, 1, 3, 2] # 000->001->011->010 (valid in Q3, extends to Q9)
        
    return best_path

def _is_valid_snake(path):
    """Check that path is a valid induced path."""
    n = len(path)
    if n < 2:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        v = path[i]
        for nb in (v ^ m for m in BIT_MASKS):
            if nb in used:
                if i > 0 and path[i - 1] == nb:
                    continue
                if i < n - 1 and path[i + 1] == nb:
                    continue
                return False
    return True
