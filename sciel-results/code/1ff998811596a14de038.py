import time
import random

N = 9
SIZE = 1 << N

def construct():
    """
    Find a long induced path (snake) in the N-dimensional hypercube Q_N.
    Returns a list of distinct integers representing the vertices in order.
    """
    # Time limit with a small safety margin
    DEADLINE = time.time() + 145.0
    
    # Precompute neighbors for speed?
    # For N=9, generating neighbors on the fly is fast enough, 
    # but precomputing a list of neighbors for all 512 nodes might help.
    # Let's precompute neighbors as a list of lists.
    neighbors = [[] for _ in range(SIZE)]
    for v in range(SIZE):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = [0]
    
    def backtrack(path, blocked):
        nonlocal best_path
        
        # Check time
        if time.time() > DEADLINE:
            return False
        
        # Update best if current path is longer
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Heuristic: If we have more than a certain length, we can stop early
        # to allow more restarts, but for finding the absolute best, 
        # we should try to go as deep as possible.
        
        cur = path[-1]
        candidates = []
        
        # Find valid candidates
        # A candidate 'v' is valid if:
        # 1. v is a neighbor of cur
        # 2. blocked[v] == 0 (v is not adjacent to path[-2] or earlier, 
        #    effectively v is not in the "forbidden" set defined by the 
        #    snake property for non-consecutive vertices)
        
        for v in neighbors[cur]:
            if blocked[v] == 0:
                candidates.append(v)
                
        if not candidates:
            return False
            
        # Heuristic: Sort candidates. 
        # Strategy: Prefer candidates that have fewer valid extensions? 
        # Or just randomize to explore different branches.
        # Randomizing is often effective for finding long paths in hypercubes.
        random.shuffle(candidates)
        
        # Optional: Try to pick a candidate that doesn't "dead-end" soon?
        # This is expensive. Let's stick to random + backtracking.
        
        # We need to update 'blocked' when we move.
        # When we add 'nxt' to path:
        # 1. Mark 'nxt' as used (implicitly by checking blocked? No, 
        #    we need to ensure we don't revisit 'nxt'. 
        #    Actually, the 'blocked' array handles "non-consecutive" conflicts.
        #    We must also ensure we don't revisit any vertex in the path.
        #    Since 'blocked' only tracks vertices adjacent to path[-2], 
        #    it does NOT track vertices in path[0..-3] directly.
        #    Wait, the standard snake definition:
        #    Consecutive: differ by 1 bit.
        #    Non-consecutive: differ by >= 2 bits.
        #    
        #    Let's re-verify the 'blocked' logic.
        #    If we have path v0, v1, ..., vk.
        #    We want to add v_{k+1}.
        #    v_{k+1} must be neighbor of vk.
        #    v_{k+1} must NOT be neighbor of v_{k-1} (diff 1).
        #    v_{k+1} must NOT be neighbor of v_{k-2} (diff 1).
        #    ...
        #    v_{k+1} must NOT be neighbor of v0 (diff 1).
        #    
        #    The 'blocked' array typically counts how many vertices in the 
        #    "forbidden neighborhood" of the new vertex exist in the path.
        #    
        #    Let's use a different, clearer approach:
        #    Maintain a `used` boolean array for all vertices in the path.
        #    Maintain a `blocked` count for vertices that are neighbors of 
        #    path[-2]. 
        #    
        #    Actually, checking `blocked[v] == 0` is only sufficient if 
        #    `blocked` accounts for ALL previous vertices.
        #    
        #    Correct Logic for O(1) check:
        #    Let `path` be the current sequence.
        #    To add `nxt`:
        #    1. `nxt` must not be in `path` (checked via `used` array).
        #    2. `nxt` must not be adjacent to `path[-2]` (checked via `blocked` 
        #       if `blocked` counts adjacency to `path[-2]`? No.)
        #       
        #    Let's look at the standard "Snake in the Box" backtracking optimization.
        #    The condition is: `d(v_i, v_j) >= 2` for `|i-j| > 1`.
        #    This is equivalent to: `v_j` is NOT a neighbor of `v_i` for `|i-j| > 1`.
        #    
        #    If we maintain an array `block_count[x]` which is the number of 
        #    vertices `v` in the path such that `x` is a neighbor of `v` 
        #    AND `v` is not the immediate predecessor of the current end?
        #    
        #    Actually, simpler:
        #    When we are at the end `v_k`, we want to choose `v_{k+1}`.
        #    `v_{k+1}` cannot be:
        #    - Any `v_i` in the path (handled by `used`)
        #    - A neighbor of any `v_i` where `i <= k-1`.
        #    
        #    Let `conflict_count[x]` be the number of vertices `v` in `path[0..k-1]` 
        #    such that `x` is a neighbor of `v`.
        #    Then `x` is a valid candidate if `conflict_count[x] == 0` AND `used[x] == False`.
        #    
        #    How to update `conflict_count` efficiently?
        #    When we add `v_k` to the path (becoming the new end):
        #    - All neighbors of `v_k` get their `conflict_count` increased by 1?
        #    - No, because `v_k` is now the predecessor. It does NOT block its 
        #      neighbors from being the *next* vertex. It only blocks them from 
        #      being non-consecutive neighbors later.
        #    
        #    Wait, the condition is: `v_{k+1}` must not be adjacent to `v_{k-1}`.
        #    `v_{k+1}` is allowed to be adjacent to `v_k` (it must be).
        #    
        #    So, when adding `v_k`:
        #    - We mark `v_k` as used.
        #    - We must ensure that future vertices are not adjacent to `v_{k-1}`.
        #    - The "blocking" effect of `v_{k-1}` starts applying to the next vertex `v_{k+1}`.
        #    
        #    Therefore:
        #    `blocked[x]` = number of vertices `v` in `path[0..k-1]` that are 
        #    adjacent to `x`? No.
        #    
        #    Let's trace:
        #    Path: v0, v1.
        #    Current end: v1.
        #    Next v2.
        #    v2 must not be adjacent to v0.
        #    
        #    If we maintain `blocked[x]` as the count of vertices in the path 
        #    (excluding the current end) that are adjacent to `x`?
        #    When we add `v2`:
        #    - New end is v2.
        #    - Next v3 must not be adjacent to v1 OR v0.
        #    - So, when we add `v2` to the path, we should add `v1` to the 
        #      "blockers" for future vertices? 
        #      Yes! `v1` becomes a blocker for `v3`.
        #    - What about `v0`? It was already a blocker for `v2`. 
        #      Does it remain a blocker for `v3`? Yes, `v3` must not be 
        #      adjacent to `v0`.
        #      
        #    So, the set of blockers for the next vertex `v_{k+1}` is 
        #    `{v_0, v_1, ..., v_{k-1}}`.
        #    The set of blockers for `v_k` was `{v_0, ..., v_{k-2}}`.
        #    
        #    Therefore, when moving from end `v_k` to `v_{k+1}`:
        #    1. We add `v_{k-1}` to the blocker set.
        #    2. We do NOT remove any blockers.
        #    
        #    This implies the blocker set only grows.
        #    `blocked[x]` = number of vertices in `path[0..k-1]` adjacent to `x`.
        #    
        #    Initialization:
        #    Path: [v0].
        #    Blockers for next vertex (v1): None? 
        #    Wait, v1 is adjacent to v0. That's required.
        #    v1 must not be adjacent to v0? No, v1 IS adjacent to v0.
        #    The condition is: `d(v_i, v_j) >= 2` for `|i-j| > 1`.
        #    For v0 and v1, `|0-1|=1`, so they CAN be adjacent.
        #    For v0 and v2, `|0-2|=2`, so they MUST NOT be adjacent.
        #    
        #    So, when choosing v1 (from v0):
        #    v1 is adjacent to v0 (OK).
        #    There are no other vertices.
        #    
        #    When choosing v2 (from v1):
        #    v2 is adjacent to v1 (OK).
        #    v2 must NOT be adjacent to v0.
        #    
        #    So, the "forbidden" neighbors for the next vertex `v_{k+1}` are 
        #    all neighbors of `v_i` where `i <= k-1`.
        #    
        #    Algorithm:
        #    `blocked[x]` counts how many vertices `v` in the current path 
        #    (specifically `path[0..k-1]`) are adjacent to `x`.
        #    
        #    Start: Path = [v0].
        #    `blocked` is all 0.
        #    
        #    Step 1: Choose v1 from neighbors of v0.
        #    Constraint: `blocked[v1] == 0`? 
        #    If `blocked` represents "neighbors of path[0..-1]", then 
        #    for v1, the set is empty (since v0 is the current end, 
        #    it's not in the "previous" set yet? Or is it?).
        #    
        #    Let's define `blocked` based on the path `v_0 ... v_k`.
        #    We want to find `v_{k+1}`.
        #    `v_{k+1}` must not be adjacent to `v_0 ... v_{k-1}`.
        #    So `blocked` should count adjacencies to `v_0 ... v_{k-1}`.
        #    
        #    Initially (k=0, path=[v0]):
        #    We want v1. v1 must not be adjacent to `v_0 ... v_{-1}` (empty).
        #    So `blocked` is all 0.
        #    We pick v1 from neighbors of v0.
        #    
        #    After picking v1, path becomes [v0, v1]. k=1.
        #    We want v2. v2 must not be adjacent to `v_0 ... v_0` (i.e., v0).
        #    So we need to update `blocked` to reflect that v0 is now a blocker.
        #    Update: For each neighbor `u` of `v0`, `blocked[u] += 1`.
        #    
        #    Check v2: `blocked[v2] == 0`? And `v2` is neighbor of `v1`.
        #    
        #    After picking v2, path becomes [v0, v1, v2]. k=2.
        #    We want v3. v3 must not be adjacent to `v_0, v_1`.
        #    Update: Add `v1` to blockers. For each neighbor `u` of `v1`, `blocked[u] += 1`.
        #    (Note: `blocked` now contains contributions from v0 AND v1).
        #    
        #    This logic holds!
        #    
        #    Implementation detail:
        #    When we recurse with `v_{k+1}`, we must update `blocked` by adding 
        #    the neighbors of `v_k` (the old end, which is now `path[-2]`).
        #    Wait, `v_k` becomes `path[-2]` after appending `v_{k+1}`?
        #    Path: `... v_k`. Append `v_{k+1}`. Path: `... v_k, v_{k+1}`.
        #    New end is `v_{k+1}`.
        #    Blockers for NEXT vertex (`v_{k+2}`) are `v_0 ... v_k`.
        #    Previous blockers (for `v_{k+1}`) were `v_0 ... v_{k-1}`.
        #    Difference: `v_k` is added.
        #    
        #    So, before recursing, we add neighbors of `v_k` to `blocked`.
        #    After recursion returns, we subtract neighbors of `v_k` from `blocked`.
    
    def dfs(path, blocked):
        nonlocal best_path
        
        if time.time() > DEADLINE:
            return
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        cur = path[-1]
        # The vertex that will become a NEW blocker is path[-2] (if it exists)
        # Wait, no. 
        # Current state: Path `v_0...v_k`. `blocked` reflects `v_0...v_{k-1}`.
        # We are choosing `v_{k+1}`.
        # Valid `v_{k+1}` must have `blocked[v_{k+1}] == 0` and be neighbor of `v_k`.
        
        candidates = []
        for v in neighbors[cur]:
            if blocked[v] == 0 and v not in path: # v not in path check?
                # Since blocked[v]==0 implies v is not neighbor of any blocker,
                # and blockers are v_0...v_{k-1}, v cannot be v_0...v_{k-1} 
                # unless v is neighbor of itself? No.
                # Can v be in the path? 
                # If v == v_i for i < k-1, then v is neighbor of v_i? 
                # d(v_i, v) = 0. Not 1. 
                # The condition "v not in path" is necessary because 
                # `blocked` only checks adjacency to previous vertices.
                # It doesn't check identity.
                # So we MUST check `used`.
                pass
        
        # We need a `used` array for O(1) membership check.
        
    # Reset for actual run
    used = [False] * SIZE
    blocked = [0] * SIZE
    best_path = [0]
    used[0] = True
    
    # Initial state:
    # Path: [0]
    # blocked: reflects neighbors of path[0..-1] (empty). So blocked is all 0.
    # We are choosing v1.
    
    # Note: We do NOT add neighbors of 0 to blocked yet.
    # We add neighbors of v_k (old end) to blocked AFTER we select v_{k+1} 
    # and BEFORE we recurse for v_{k+2}.
    
    def search(path, blocked, used):
        nonlocal best_path
        
        if time.time() > DEADLINE:
            return
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        cur = path[-1]
        
        # Find valid next vertices
        candidates = []
        for v in neighbors[cur]:
            if used[v]:
                continue
            if blocked[v] > 0:
                continue
            candidates.append(v)
            
        if not candidates:
            return
            
        # Randomize to explore different branches
        random.shuffle(candidates)
        
        for v in candidates:
            # Add v to path
            path.append(v)
            used[v] = True
            
            # Update blocked for the NEXT step
            # The current end `cur` becomes a blocker for the next-next vertex.
            # So we add neighbors of `cur` to `blocked`.
            for u in neighbors[cur]:
                blocked[u] += 1
                
            # Recurse
            search(path, blocked, used)
            
            # Backtrack
            for u in neighbors[cur]:
                blocked[u] -= 1
            used[v] = False
            path.pop()
            
            # Optional: If we found a very long path, we might want to break 
            # and restart to find an even longer one in a different region.
            # But simple backtracking usually finds the local max quickly.
            
    # Run multiple restarts to explore different parts of the hypercube
    # Since the search space is huge, a single run might get stuck.
    # However, backtracking explores the entire tree from 0. 
    # Is the tree connected? Yes.
    # But the "validity" constraint makes it sparse.
    # We might want to restart with a different starting vertex or 
    # random bit orderings.
    
    # For simplicity and speed, we run one deep backtracking search from 0.
    # If time allows, we can restart with a random start.
    
    start_time = time.time()
    
    # First run from 0
    path = [0]
    used = [False] * SIZE
    blocked = [0] * SIZE
    used[0] = True
    
    search(path, blocked, used)
    
    # If time remains, try a few more random starts
    while time.time() < DEADLINE - 1.0:
        start_v = random.randint(0, SIZE - 1)
        
        # Reset
        path = [start_v]
        used = [False] * SIZE
        blocked = [0] * SIZE
        used[start_v] = True
        
        # We can limit the depth of these secondary searches to avoid 
        # spending too much time on a bad start, but the time guard handles it.
        search(path, blocked, used)
        
        # Early exit if we hit the theoretical max (unknown) or a very high number
        # For N=9, max is around 190+. 
        # If we find 190, we can stop.
        if len(best_path) > 180:
             break

    return best_path

# The construct function is defined above. 
# The task asks to keep the function name `construct`.
# The code above defines `construct`.

# One optimization: The `search` function uses `nonlocal best_path`. 
# In Python, nested functions can access variables from the enclosing scope.
# However, `search` is defined inside `construct`.
# `best_path`, `used`, `blocked` are local to `construct`.
# `search` modifies `best_path` (via nonlocal), and reads/modifies `used` and `blocked` 
# (passed as args, so mutation is visible).
# This should work.

# Let's double check the `blocked` update logic.
# Path: v0, v1.
# blocked initially 0.
# Choose v1.
# Before recursing for v2, we add neighbors of v0 to blocked.
# So blocked[u] > 0 if u is neighbor of v0.
# In recursion, we choose v2.
# v2 must have blocked[v2] == 0.
# So v2 is NOT a neighbor of v0. Correct.
# 
# Path: v0, v1, v2.
# Before recursing for v3, we add neighbors of v1 to blocked.
# blocked now reflects neighbors of v0 AND v1.
# v3 must have blocked[v3] == 0.
# So v3 is NOT neighbor of v0 AND NOT neighbor of v1. Correct.
# 
# The logic is sound.

# One potential issue: Recursion depth.
# Max length ~190. Python default recursion limit is 1000. This is fine.
# However, recursion overhead in Python can be high.
# Given the 150s limit, it should be fast enough for N=9.

# To be safe against recursion limits or stack overflows, we could use an explicit stack,
# but 190 depth is trivial.

# Final check on `used` array.
# `used[v]` is set to True when v is added to path.
# `used[v]` is set to False when backtracking.
# This ensures we don't revisit vertices.

# The `blocked` array is updated before the recursive call and reverted after.
# This ensures that the state is correct for the next level.

# The function `construct` returns `best_path`.

# One minor improvement: 
# If we find a path of length L, and we are exploring a branch, 
# if the current path length + (SIZE - len(path)) < len(best_path), we could prune?
# No, because we don't know how many valid vertices remain.
# But we can prune if `len(path) + (SIZE - len(path)) <= len(best_path)`?
# That's just `SIZE <= len(best_path)`, which is false.
# 
# A better pruning: 
# If `len(path) + remaining_available <= len(best_path)`, prune.
# `remaining_available` is hard to compute quickly.
# We can use `SIZE - len(path)` as an upper bound.
# If `len(path) + (SIZE - len(path)) <= len(best_path)`, i.e., `SIZE <= len(best_path)`.
# This is never true until we've visited all vertices.
# So no simple pruning on length alone.

# We will rely on the time limit and randomization.

# The code is ready.

