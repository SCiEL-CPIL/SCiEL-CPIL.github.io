import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We store them as tuples of indices (0-based for internal, 1-based for output)
    # Let's use 0-based indices internally: 0 to N-1
    aps = []
    ap_index = {} # map element to list of AP indices it belongs to
    
    idx = 0
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(ap)
            for elem in ap:
                if elem not in ap_index:
                    ap_index[elem] = []
                ap_index[elem].append(idx)
            idx += 1
    
    num_aps = len(aps)
    
    # State
    in_set = [False] * N
    block_count = [0] * N # For elements NOT in set, count of APs where they are the 4th missing one
    
    # Helper to update block counts
    # When adding element x:
    # For each AP containing x, if the other 3 are in_set, then for the 4th element y (if not in_set), block_count[y] += 1
    # Actually, if x is added, it might complete an AP for some y? No, y is the one being blocked.
    # If we add x, we might block y if x, y, z, w form an AP and z, w are already in set.
    # So for each AP containing x:
    #   Let y, z, w be the other elements.
    #   If z and w are in_set, and y is not in_set, then block_count[y] += 1.
    #   If z and y are in_set, and w is not in_set, then block_count[w] += 1.
    #   If y and w are in_set, and z is not in_set, then block_count[z] += 1.
    #   (Note: if multiple are not in set, no block is created yet, because we need 3 in set to block the 4th)
    
    def add_element(x):
        in_set[x] = True
        for ap_idx in ap_index[x]:
            ap = aps[ap_idx]
            others = [e for e in ap if e != x]
            # Check how many of the others are in set
            # If exactly 2 are in set, the 3rd (not in set) gets blocked
            in_set_others = [e for e in others if in_set[e]]
            if len(in_set_others) == 2:
                # The one not in set is blocked
                for e in others:
                    if not in_set[e]:
                        block_count[e] += 1
                        break
    
    def remove_element(x):
        in_set[x] = False
        for ap_idx in ap_index[x]:
            ap = aps[ap_idx]
            others = [e for e in ap if e != x]
            in_set_others = [e for e in others if in_set[e]]
            if len(in_set_others) == 2:
                # The one not in set was blocked by x (and the other 2)
                # So its block_count decreases
                for e in others:
                    if not in_set[e]:
                        block_count[e] -= 1
                        break

    # Initialize with a greedy approach to get a decent start
    # Greedy: add smallest possible
    # But let's just start empty and let the local search do its thing, 
    # or use a random start?
    # A greedy start is usually better for local search.
    
    # Let's do a simple greedy initialization
    for i in range(N):
        if block_count[i] == 0:
            add_element(i)
            
    current_size = sum(in_set)
    best_size = current_size
    best_set = [i for i in range(N) if in_set[i]]
    
    # Local Search
    # We will try to improve the set.
    # Strategy: 
    # 1. If there are unblocked elements, add the best one.
    # 2. If no unblocked elements, remove the element that unblocks the most "critical" elements.
    
    # Precompute for each x, the list of APs it's in. Already done in ap_index.
    
    # To speed up "find best x to remove", we can maintain a score for each x in set.
    # score[x] = number of y not in set such that block_count[y] == 1 and x is in a blocking AP for y.
    # This is expensive to update dynamically. 
    # Given N=512, we can just recalculate it when we are stuck.
    
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        # Check for unblocked elements
        unblocked = [i for i in range(N) if not in_set[i] and block_count[i] == 0]
        
        if unblocked:
            # Pick one to add. 
            # Heuristic: Pick the one that blocks the fewest future elements?
            # Or just pick the first one?
            # Let's pick the one with smallest index for determinism, or random?
            # Random might help escape local minima.
            x_to_add = random.choice(unblocked)
            add_element(x_to_add)
            current_size += 1
            stuck_count = 0
            
            if current_size > best_size:
                best_size = current_size
                best_set = [i for i in range(N) if in_set[i]]
                
        else:
            # Stuck. Need to remove an element.
            # Calculate unblock potential for each x in set
            # potential[x] = count of y not in set such that block_count[y] == 1 and x blocks y
            
            # Optimization: Only consider x that are in APs with y where block_count[y]==1
            # Let's just iterate through all x in set and count.
            # N is small, so this is fine.
            
            in_set_indices = [i for i in range(N) if in_set[i]]
            if not in_set_indices:
                break # Should not happen
            
            max_potential = -1
            best_x_to_remove = None
            
            # To speed up, we can precompute for each y with block_count[y]==1, 
            # which x in set block it.
            # Let's do that.
            
            # Map y -> list of x in set that block it
            # We only care about y with block_count[y] == 1.
            
            # Actually, if block_count[y] > 1, removing one x won't unblock it.
            # So we only look at y with block_count[y] == 1.
            
            # For each such y, find the x in set that is in the same AP as y and the other 2 in set.
            
            potential_counts = {x: 0 for x in in_set_indices}
            
            for y in range(N):
                if not in_set[y] and block_count[y] == 1:
                    # Find the AP that blocks y
                    # It must be an AP containing y and 3 elements from set.
                    # Since block_count[y] is 1, there is exactly one such AP.
                    # We can find it by checking APs containing y.
                    for ap_idx in ap_index[y]:
                        ap = aps[ap_idx]
                        others = [e for e in ap if e != y]
                        if all(in_set[e] for e in others):
                            # This is the blocking AP
                            # Any x in others blocks y.
                            # Since we remove only one x, removing any of them will unblock y.
                            # So we increment potential for each x in others.
                            for x in others:
                                potential_counts[x] += 1
                            break # Only one such AP exists because block_count[y]==1
            
            # Find x with max potential
            for x in in_set_indices:
                if potential_counts[x] > max_potential:
                    max_potential = potential_counts[x]
                    best_x_to_remove = x
            
            if max_potential == 0:
                # Removing any x doesn't immediately unblock any y with count 1.
                # Maybe we should remove a random x to explore?
                # Or stop if we've been stuck for a while.
                stuck_count += 1
                if stuck_count > 100:
                    break
                # Pick random x to remove
                best_x_to_remove = random.choice(in_set_indices)
            else:
                stuck_count = 0
                
            remove_element(best_x_to_remove)
            current_size -= 1
            
            # Update best? No, size decreased. But we might increase it later.
            # We keep best_set as the maximum size seen so far.

    return sorted([x + 1 for x in best_set])
