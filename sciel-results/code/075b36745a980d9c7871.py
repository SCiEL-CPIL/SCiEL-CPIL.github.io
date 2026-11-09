import time
import random

# EVOLVE-BLOCK-START
P = {'n': 12, 'm': 40}
N = 1 << P['n']
M = P['m']

# Precompute comparability lists and similarity check helper
# comparables[i] is a list of j such that i != j and (i subset j or j subset i)
comparables = [[] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        # Check if i subset j or j subset i
        if (i & j) == i or (i & j) == j:
            comparables[i].append(j)

def get_list_from_mask(mask):
    return [b for b in range(P['n']) if (mask >> b) & 1]

def count_pairs(family):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(family)
    for i in range(m):
        for j in range(i + 1, m):
            if (family[i] & family[j]) == family[i] or (family[i] & family[j]) == family[j]:
                count += 1
    return count

def greedy_construct(tie_breaker='mask', seed=None):
    """
    Greedy construction of the family.
    tie_breaker: 'mask' (smallest mask), 'size' (smallest size), 'rand' (random among max)
    """
    selected = []
    selected_set = set()
    gains = [0] * N
    
    rng = random.Random(seed)
    
    for _ in range(M):
        best_gain = -1
        candidates = []
        
        # Find max gain
        for i in range(N):
            if i in selected_set:
                continue
            if gains[i] > best_gain:
                best_gain = gains[i]
                candidates = [i]
            elif gains[i] == best_gain:
                candidates.append(i)
        
        if not candidates:
            break
            
        # Tie breaking
        if tie_breaker == 'mask':
            next_mask = min(candidates)
        elif tie_breaker == 'size':
            # Prefer smaller popcount? Or larger? Let's try smaller first.
            next_mask = min(candidates, key=lambda x: bin(x).count('1'))
        elif tie_breaker == 'size_desc':
            next_mask = min(candidates, key=lambda x: -bin(x).count('1'))
        else: # rand
            next_mask = rng.choice(candidates)
            
        selected.append(next_mask)
        selected_set.add(next_mask)
        
        # Update gains
        for j in comparables[next_mask]:
            if j not in selected_set:
                gains[j] += 1
            # Note: if j is already selected, we don't need to update its gain for future selection,
            # but gains are only used for *unselected* items.
            
    return selected

def local_search(family, max_iterations=1000):
    """
    1-swap local search.
    family: list of masks
    """
    selected_set = set(family)
    # Initialize deg array
    # deg[i] = number of elements in family comparable to i
    deg = [0] * N
    for s in family:
        for t in comparables[s]:
            if t in selected_set:
                deg[t] += 1
    # Note: deg[s] for s in family is the number of other family members comparable to s.
    # My definition above: deg[i] counts neighbors in family.
    # Let's verify:
    # If s, t in family and s~t, then t is in comparables[s] and s is in comparables[t].
    # The loop `for s in family: for t in comparables[s]: if t in selected_set: deg[t]+=1`
    # increments deg[t] for each s in family comparable to t.
    # So deg[t] is indeed the number of elements in family comparable to t.
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(len(family)):
            s = family[i]
            base = deg[s]
            best_t = -1
            best_gain = base # We need strictly greater to improve
            
            # Find best T not in selected
            # We want to maximize (deg[T] - sim(s, T))
            # Let val(T) = deg[T] - (1 if s~T else 0)
            # If val(T) > base, swap is beneficial.
            
            # Optimization: We can pre-filter T?
            # Just scan all N.
            for t in range(N):
                if t in selected_set:
                    continue
                # Calculate potential gain
                sim = 1 if ((s & t) == s or (s & t) == t) else 0
                val = deg[t] - sim
                if val > best_gain:
                    best_gain = val
                    best_t = t
            
            if best_t != -1:
                # Perform swap s -> best_t
                # Update deg
                # Remove s:
                for t in comparables[s]:
                    if t in selected_set:
                        deg[t] -= 1
                # Note: deg[s] itself? 
                # deg[s] was base. 
                # The elements t in comparables[s] that are in selected_set lose 1 neighbor (s).
                # What about s itself? deg[s] is not used for selection anymore, 
                # but if we keep the array consistent:
                # s is removed from selected_set.
                
                selected_set.remove(s)
                
                # Add best_t:
                for t in comparables[best_t]:
                    if t in selected_set:
                        deg[t] += 1
                # Also, best_t itself gets degree from others?
                # The loop above updates deg[t] for t in comparables[best_t].
                # This includes all neighbors of best_t that are in the set.
                # Does it update deg[best_t]? No, best_t is not in comparables[best_t].
                # But deg[best_t] should reflect its new neighbors.
                # The neighbors of best_t in the new set are the old neighbors in selected_set.
                # The loop `for t in comparables[best_t]: if t in selected_set: deg[t] += 1` 
                # updates the degrees of the *neighbors*, not best_t itself.
                # We need deg[best_t] to be correct for future swaps.
                # deg[best_t] = number of elements in selected_set comparable to best_t.
                # This is exactly the count of `t in comparables[best_t]` that are in selected_set.
                # So we can just compute it or maintain it.
                # Since we update deg[t] for neighbors, deg[best_t] is NOT updated by this loop.
                # We need to set deg[best_t] correctly.
                # deg[best_t] = sum(1 for x in selected_set if x in comparables[best_t])
                # Or simply: deg[best_t] = 0
                # for x in comparables[best_t]:
                #    if x in selected_set: deg[best_t] += 1
                
                # Let's just recalculate deg[best_t] and deg[s] is no longer needed.
                # Actually, we can just recalculate deg for all selected items if it's cheap?
                # No, let's be precise.
                
                # Correct update logic:
                # 1. Remove s from selected_set.
                # 2. For every x in selected_set, if x ~ s, deg[x] decreases by 1.
                #    (This is what the first loop does, but it checks `if t in selected_set` 
                #     BEFORE removing s? No, I removed s first in the snippet above? 
                #     Let's reorder: 
                #     1. Remove s from selected_set.
                #     2. For t in comparables[s]: if t in selected_set: deg[t] -= 1.
                #     3. Add best_t to selected_set.
                #     4. For t in comparables[best_t]: if t in selected_set: deg[t] += 1.
                #     5. Calculate deg[best_t] = sum(1 for t in comparables[best_t] if t in selected_set).
                #     6. Replace s with best_t in family list.
                
                # Let's rewrite the swap block properly.
                pass # Placeholder for actual implementation below

    return family

def construct():
    start_time = time.time()
    deadline = 50.0 # 50s margin
    
    best_family = []
    best_score = -1
    
    # Try a few different tie-breakers and random seeds
    strategies = [
        ('mask', None),
        ('size', None),
        ('size_desc', None),
        ('rand', 1),
        ('rand', 42),
        ('rand', 123),
    ]
    
    for tie_breaker, seed in strategies:
        if time.time() > deadline:
            break
            
        family = greedy_construct(tie_breaker, seed)
        
        # Local Search
        # Implementing the local search correctly is crucial.
        # Let's implement a simple version first.
        
        selected_set = set(family)
        deg = [0] * N
        for s in family:
            for t in comparables[s]:
                if t in selected_set:
                    deg[t] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(family)):
                s = family[i]
                base = deg[s]
                best_t = -1
                best_val = base
                
                for t in range(N):
                    if t in selected_set:
                        continue
                    sim = 1 if ((s & t) == s or (s & t) == t) else 0
                    val = deg[t] - sim
                    if val > best_val:
                        best_val = val
                        best_t = t
                
                if best_t != -1:
                    # Perform Swap
                    # 1. Remove s
                    selected_set.remove(s)
                    for t in comparables[s]:
                        if t in selected_set:
                            deg[t] -= 1
                            
                    # 2. Add best_t
                    selected_set.add(best_t)
                    for t in comparables[best_t]:
                        if t in selected_set:
                            deg[t] += 1
                            
                    # 3. Update deg[best_t]
                    deg[best_t] = 0
                    for t in comparables[best_t]:
                        if t in selected_set:
                            deg[best_t] += 1
                            
                    # 4. Update family list
                    family[i] = best_t
                    improved = True
                    break # Restart scan from beginning? Or continue?
                    # Restarting is safer to avoid missing better swaps that become available.
                    # But breaking inner loop and restarting outer is O(m) overhead.
                    # Given m=40, it's fine.
            
        score = count_pairs(family)
        if score > best_score:
            best_score = score
            best_family = family[:]
            
    # Convert best_family to list of lists
    if not best_family:
        # Fallback
        return [[b for b in range(P['n']) if (mask>>b)&1] for mask in range(M)]
        
    return [get_list_from_mask(mask) for mask in best_family]
# EVOLVE-BLOCK-END
