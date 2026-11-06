import time
import random

N = 9

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_path = []
    best_len = 0
    
    # Precompute neighbors for all vertices?
    # 2^9 = 512 vertices.
    # neighbors[v] = list of 9 neighbors.
    neighbors = {}
    for v in range(2**N):
        neighbors[v] = [v ^ (1 << b) for b in range(N)]
    
    def dfs(path, current, prev, forbidden):
        nonlocal best_path, best_len, start_time, deadline
        
        # Time check
        if time.time() - start_time > deadline:
            return
        
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
        # Generate candidates
        candidates = []
        for w in neighbors[current]:
            if w not in forbidden:
                candidates.append(w)
                
        if not candidates:
            return
            
        # Heuristic: Warnsdorff
        # For each candidate w, count valid next moves
        # Next moves from w are neighbors x of w such that x not in new_forbidden
        # new_forbidden = forbidden | {w} | set(neighbors(prev))
        
        # To speed up, we can precompute the set of neighbors of prev
        prev_neighbors_set = set(neighbors[prev])
        
        def score(w):
            # Count x in neighbors[w] such that x not in forbidden and x not in prev_neighbors_set and x != w
            # Note: x != w is always true for neighbors
            # Note: x in forbidden is checked
            # Note: x in prev_neighbors_set is checked
            count = 0
            for x in neighbors[w]:
                if x in forbidden:
                    continue
                if x in prev_neighbors_set:
                    continue
                count += 1
            return count
            
        # Sort candidates by score (ascending)
        candidates.sort(key=score)
        
        for w in candidates:
            if time.time() - start_time > deadline:
                return
            
            # Construct new forbidden set
            # new_forbidden = forbidden | {w} | prev_neighbors_set
            # Since forbidden is a set, we can do:
            new_forbidden = forbidden | {w} | prev_neighbors_set
            
            # Recurse
            path.append(w)
            dfs(path, w, current, new_forbidden)
            path.pop()
            
            if time.time() - start_time > deadline:
                return

    # Start DFS
    # Initial state:
    # path = [0]
    # current = 0
    # prev = None?
    # forbidden = set()
    # Wait, the update rule:
    # F' = F | {w} | neighbors(prev)
    # For the first step:
    # path=[0], current=0, prev=None.
    # Candidates w are neighbors of 0 not in forbidden (empty).
    # If we pick w, new_forbidden = empty | {w} | neighbors(None)?
    # neighbors(None) is undefined.
    # Let's adjust the state.
    # State: path, current, forbidden.
    # forbidden contains all vertices that cannot be the NEXT vertex.
    # Initially, forbidden is empty.
    # When we add w to path [0]:
    # Next current is w.
    # The constraint for the step AFTER w is: no adjacency to 0.
    # So new_forbidden should contain neighbors of 0.
    # And w itself is in the path, so it's forbidden for future steps?
    # Yes, vertices in the path are forbidden.
    # So new_forbidden = {w} | neighbors(0).
    
    # Let's re-verify the general step.
    # Path: v0...vk. Current vk.
    # Forbidden F contains:
    # - All vi for i <= k
    # - All neighbors of vi for i < k-1
    # 
    # Wait, if F contains all vi, then we don't need to check "in path" separately.
    # 
    # Transition vk -> vk+1 = w.
    # New Path: v0...vk w.
    # New Current: w.
    # New Forbidden F' must contain:
    # - All vi for i <= k+1 (so F plus w)
    # - All neighbors of vi for i < k (so F's neighbors part plus neighbors(vk-1))
    # 
    # So F' = F | {w} | neighbors(vk-1).
    # 
    # For the first step (k=0, current=v0=0, prev=None):
    # F = empty.
    # We pick w (neighbor of 0).
    # F' = empty | {w} | neighbors(None)?
    # The formula "neighbors(vk-1)" fails for k=0.
    # For k=0, the condition for the next step (finding v2) is no adjacency to v0.
    # So F' should contain neighbors(v0).
    # So for k=0, we add neighbors(v0).
    # 
    # Generalized:
    # If prev is None (first step):
    #   new_forbidden = {w} | set(neighbors(current))
    # Else:
    #   new_forbidden = forbidden | {w} | set(neighbors(prev))
    
    def dfs_start(path, current, prev, forbidden):
        nonlocal best_path, best_len, start_time, deadline
        
        if time.time() - start_time > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
        candidates = []
        for w in neighbors[current]:
            if w not in forbidden:
                candidates.append(w)
                
        if not candidates:
            return
            
        # Heuristic
        if prev is None:
            # Next forbidden will be {w} | neighbors(current)
            # Score w: count x in neighbors[w] not in {w} and not in neighbors(current)
            curr_neighbors_set = set(neighbors[current])
            def score(w):
                count = 0
                for x in neighbors[w]:
                    if x == w: continue
                    if x in curr_neighbors_set: continue
                    count += 1
                return count
            sort_key = score
        else:
            prev_neighbors_set = set(neighbors[prev])
            def score(w):
                count = 0
                for x in neighbors[w]:
                    if x in forbidden: continue
                    if x in prev_neighbors_set: continue
                    count += 1
                return count
            sort_key = score
            
        candidates.sort(key=sort_key)
        
        for w in candidates:
            if time.time() - start_time > deadline:
                return
            
            if prev is None:
                new_forbidden = {w} | set(neighbors[current])
            else:
                new_forbidden = forbidden | {w} | set(neighbors[prev])
            
            path.append(w)
            dfs_start(path, w, current, new_forbidden)
            path.pop()
            
            if time.time() - start_time > deadline:
                return

    dfs_start([0], 0, None, set())
    
    return best_path
# EVOLVE-BLOCK-END
