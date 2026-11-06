import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We will try to find a valid coloring for increasing n
    # Start with n=12, if solved quickly, try 13, etc.
    # Or just pick a target like 16 and try hard.
    # Given the seed is 11, getting 12 is easy. 16 is harder but doable.
    # Let's aim for 16.
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    target_n = 16
    
    # Helper to check validity
    def is_valid(n, colors):
        # colors is a list of length n*(n-1)/2
        # Reconstruct adjacency or just check C4s
        # Efficient check:
        # For each color c, build adjacency lists
        # For each pair u, v, count common neighbors in c
        # If any count > 1, invalid.
        
        # This is O(n^3) roughly, but for n=16 it's fast.
        # 16^3 = 4096. Very fast.
        pass # We will use incremental search instead of full check every time
        
    def solve_for_n(n, time_limit):
        # Initialize random coloring
        # 0..4 are colors
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        
        # Data structures for incremental update
        # adj[u][v] = color
        adj = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1) 
                # Wait, standard lexicographic index:
                # i=0: j=1..n-1 (n-1 edges)
                # i=1: j=2..n-1 (n-2 edges)
                # index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
                # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
                idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                
        # common[u][v][c] = number of common neighbors of u and v in color c
        # This is n x n x 5. For n=16, 16*16*5 = 1280 ints. Fine.
        common = [[[0] * 5 for _ in range(n)] for _ in range(n)]
        
        # Initialize common
        # For each color c, for each pair u, v, count w such that adj[u][w]==c and adj[v][w]==c
        # O(n^3 * 5) is 16^3 * 5 = 20480. Fast.
        for u in range(n):
            for v in range(u + 1, n):
                for c in range(5):
                    count = 0
                    for w in range(n):
                        if w != u and w != v:
                            if adj[u][w] == c and adj[v][w] == c:
                                count += 1
                    common[u][v][c] = count
                    common[v][u][c] = count
                    
        # Total violations E = sum_{u<v} sum_{c} C(common[u][v][c], 2)
        def get_E():
            e = 0
            for u in range(n):
                for v in range(u + 1, n):
                    for c in range(5):
                        cnt = common[u][v][c]
                        if cnt >= 2:
                            e += cnt * (cnt - 1) // 2
            return e
            
        E = get_E()
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        steps = 0
        
        while time.time() < time_limit:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            c_old = adj[u][v]
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
                
            # Calculate delta E
            # Removing (u, v) from c_old
            # Adding (u, v) to c_new
            
            delta = 0
            
            # Effect of removing (u, v) from c_old
            # For each w != u, v:
            # If adj[u][w] == c_old and adj[v][w] == c_old:
            #   common[u][v][c_old] decreases by 1? NO.
            #   common[u][w][c_old] decreases by 1 (v was a common neighbor)
            #   common[v][w][c_old] decreases by 1 (u was a common neighbor)
            #   common[u][v][c_old] is UNCHANGED (as reasoned before)
            
            # Wait, let's re-verify the "common[u][v] unchanged" logic.
            # common[u][v][c] = |N_c(u) \cap N_c(v)|.
            # N_c(u) contains v. N_c(v) contains u.
            # Intersection: does it contain u? No, u not in N_c(u).
            # Does it contain v? No, v not in N_c(v).
            # So removing the edge (u, v) removes v from N_c(u) and u from N_c(v).
            # It does NOT remove any element from the intersection, because u and v were not in the intersection to begin with.
            # So common[u][v][c_old] is indeed unchanged.
            
            # However, for any w:
            # common[u][w][c_old]:
            #   Old: |N_c(u) \cap N_c(w)|.
            #   New: N_c(u) loses v.
            #   If v in N_c(w), then intersection loses v.
            #   So common[u][w][c_old] decreases by 1 if adj[v][w] == c_old.
            
            # Similarly common[v][w][c_old] decreases by 1 if adj[u][w] == c_old.
            
            # So we need to iterate w from 0 to n-1, w!=u, w!=v.
            
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Update for c_old (removal)
                if adj[u][w] == c_old and adj[v][w] == c_old:
                    # common[u][v][c_old] unchanged
                    # common[u][w][c_old] dec by 1
                    # common[v][w][c_old] dec by 1
                    pass # We need to calculate delta E based on changes in common counts
                    
                # This logic is getting complex to do "on the fly" without updating the structure first.
                # Let's calculate delta E by simulating the change on the common counts for affected pairs.
                
                # Affected pairs for c_old:
                # (u, w) and (v, w) for all w.
                # (u, v) is unaffected in count, but let's be sure.
                
                # Let's just update the common array and recalculate E locally?
                # No, E is global. We need delta E.
                # Delta E comes from changes in common[u][w][c] and common[v][w][c].
                
                # Let's compute the change in "cost" for a specific pair (a, b, c) when common[a][b][c] changes by d.
                # Cost(a, b, c) = C(common[a][b][c], 2)
                # Change = C(k+d, 2) - C(k, 2) = (k+d)(k+d-1)/2 - k(k-1)/2
                # = (k^2 + 2kd + d^2 - k - d) - (k^2 - k) / 2
                # = (2kd + d^2 - d) / 2
                # = d(k - 1 + d/2)
                
                def cost_diff(k, d):
                    return (k + d) * (k + d - 1) // 2 - k * (k - 1) // 2

                # We will apply changes to a temporary copy of affected common values?
                # Or just accumulate delta.
                
                # Changes for c_old (removal of edge u-v):
                # For each w != u, v:
                #   if adj[u][w] == c_old and adj[v][w] == c_old:
                #       common[u][w][c_old] -= 1
                #       common[v][w][c_old] -= 1
                #       # Note: common[u][v][c_old] is NOT changed.
                
                #   if adj[u][w] == c_old and adj[v][w] != c_old:
                #       # v is not in N_c(w), so v was not a common neighbor of u and w.
                #       # So common[u][w][c_old] does NOT change.
                #       # Wait. N_c(u) loses v.
                #       # common[u][w] = |N_c(u) \cap N_c(w)|.
                #       # v was in N_c(u). Was v in N_c(w)?
                #       # If adj[v][w] != c_old, then v is NOT in N_c(w).
                #       # So v was NOT in the intersection.
                #       # So removing v from N_c(u) does NOT change the intersection size.
                #       # Correct.
                
                #   So only if BOTH adj[u][w]==c_old AND adj[v][w]==c_old do we have a change?
                #   Let's re-read.
                #   N_c(u) loses v.
                #   Intersection N_c(u) \cap N_c(w).
                #   If v in N_c(w), then v was in intersection. Now it's out. Count -1.
                #   If v not in N_c(w), then v was not in intersection. Count unchanged.
                #   So YES, condition is adj[v][w] == c_old.
                #   But we also need u to be a neighbor of w? No.
                #   The edge (u, w) color determines if u is in N_c(w) or w in N_c(u).
                #   The change is in N_c(u).
                #   So we check if v \in N_c(w). i.e. adj[v][w] == c_old.
                #   Does it matter if adj[u][w] == c_old?
                #   No. The common neighbors are vertices z.
                #   z=v is a common neighbor if v \in N_c(u) AND v \in N_c(w).
                #   We know v \in N_c(u) (before removal).
                #   So we just check if v \in N_c(w).
                
                # So for removal from c_old:
                # For each w != u, v:
                #   if adj[v][w] == c_old:
                #       common[u][w][c_old] -= 1
                #   if adj[u][w] == c_old:
                #       common[v][w][c_old] -= 1
                
                # Similarly for addition to c_new:
                # For each w != u, v:
                #   if adj[v][w] == c_new:
                #       common[u][w][c_new] += 1
                #   if adj[u][w] == c_new:
                #       common[v][w][c_new] += 1

                # We need to calculate delta E from these changes.
                # We must be careful not to double count or miss interactions.
                # The changes are independent for each pair (u, w) and (v, w).
                
                # Let's collect all changes:
                # changes = list of (pair, color, delta_count)
                # pair is (a, b) with a < b.
                
                changes = []
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # Removal from c_old
                    if adj[v][w] == c_old:
                        # common[u][w][c_old] decreases by 1
                        a, b = (u, w) if u < w else (w, u)
                        changes.append((a, b, c_old, -1))
                        
                    if adj[u][w] == c_old:
                        # common[v][w][c_old] decreases by 1
                        a, b = (v, w) if v < w else (w, v)
                        changes.append((a, b, c_old, -1))
                        
                    # Addition to c_new
                    if adj[v][w] == c_new:
                        # common[u][w][c_new] increases by 1
                        a, b = (u, w) if u < w else (w, u)
                        changes.append((a, b, c_new, 1))
                        
                    if adj[u][w] == c_new:
                        # common[v][w][c_new] increases by 1
                        a, b = (v, w) if v < w else (w, v)
                        changes.append((a, b, c_new, 1))
                
                # Now apply changes to calculate delta E and update common array
                # Note: Multiple changes might affect the same (a, b, c).
                # We should aggregate them.
                # Since n is small, we can just apply them one by one and update common.
                # But we need delta E.
                # We can calculate delta E by summing cost_diff for each change.
                # However, if two changes affect the same (a, b, c), we must apply them sequentially or combine.
                # Combining is safer.
                
                # Let's use a dictionary to aggregate changes
                agg_changes = {}
                for a, b, c, d in changes:
                    key = (a, b, c)
                    agg_changes[key] = agg_changes.get(key, 0) + d
                    
                delta_e = 0
                for (a, b, c), d in agg_changes.items():
                    k = common[a][b][c]
                    delta_e += cost_diff(k, d)
                    common[a][b][c] += d
                    common[b][a][c] += d # Keep symmetric? 
                    # Wait, my common array is indexed [u][v]. 
                    # I initialized common[u][v] for u<v? 
                    # In init: 
                    # for u in range(n): for v in range(u+1, n):
                    # common[u][v][c] = count
                    # common[v][u][c] = count
                    # So yes, symmetric.
                    
                E += delta_e
                
                # If E < 0, accept.
                # Else accept with prob exp(-delta_e / T)
                
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    # Accept move
                    adj[u][v] = c_new
                    adj[v][u] = c_new
                    idx = u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
                    colors[idx] = c_new
                else:
                    # Reject move, revert changes
                    for (a, b, c), d in agg_changes.items():
                        common[a][b][c] -= d
                        common[b][a][c] -= d
                    E -= delta_e # Revert E? 
                    # Wait, E was updated by delta_e. So E_new = E_old + delta_e.
                    # If we reject, E should be E_old.
                    # So E -= delta_e is correct.
                    
            T *= cooling
            steps += 1
            
            if E == 0:
                return colors, n
                
        return None, n

    # Try increasing n
    current_limit = 140.0
    for target in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > start_time + 130:
            break
            
        # Allocate time based on difficulty
        # n=12 is easy, n=16 is hard.
        # Let's give 5s for 12, 10s for 13, 15s for 14, 20s for 15, 40s for 16.
        time_alloc = 5
        if target == 13: time_alloc = 10
        if target == 14: time_alloc = 15
        if target == 15: time_alloc = 25
        if target == 16: time_alloc = 40
        if target > 16: time_alloc = 20 # Just a shot in the dark
        
        end_time = time.time() + time_alloc
        colors, n_found = solve_for_n(target, end_time)
        
        if colors is not None:
            best_n = n_found
            best_colors = colors
            # If we found a valid one, we can try to extend or just keep it.
            # The task asks to maximize n.
            # So we continue to the next target.
            
    return {'n': best_n, 'colors': best_colors}
