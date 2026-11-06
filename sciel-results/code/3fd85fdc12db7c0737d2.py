import math
import time
from functools import lru_cache

# Precomputation
N = 511
weights = [0.0] * (N + 1)
for i in range(1, N + 1):
    weights[i] = 1.0 / (512 - i)

# Build conflict and compatibility masks
# conflict_mask[i] has bit j set if gcd(i, j) > 1
# compat_mask[i] has bit j set if gcd(i, j) == 1
conflict_mask = [0] * (N + 1)
compat_mask = [0] * (N + 1)

all_vertices_mask = 0
for i in range(1, N + 1):
    all_vertices_mask |= (1 << i)

for i in range(1, N + 1):
    for j in range(1, N + 1):
        if math.gcd(i, j) > 1:
            conflict_mask[i] |= (1 << j)
        else:
            compat_mask[i] |= (1 << j)

# Precompute list of vertices sorted by weight descending for selection heuristic
# We will pick the highest weight vertex from the available set.
# To do this efficiently, we can just iterate or maintain a sorted list.
# Since N is small, iterating to find max weight in avail_mask is fine.

def get_best_vertex(avail_mask):
    best_v = -1
    best_w = -1.0
    # Iterate through set bits
    m = avail_mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        if weights[v] > best_w:
            best_w = weights[v]
            best_v = v
        m ^= lsb
    return best_v

def sum_weights(avail_mask):
    total = 0.0
    m = avail_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        total += weights[v]
        m ^= lsb
    return total

# Global variables for B&B
global_best = 0.0
global_best_set = [] # To track the actual set if needed, but score is enough for now
# We need to return the set, so we should track the path.
# Alternatively, run B&B to find max score, then reconstruct?
# Reconstructing is hard with B&B.
# Better: store the best set found.

best_solution = []

def search(avail_mask, current_score, current_set):
    global global_best, best_solution
    
    if current_score > global_best:
        global_best = current_score
        best_solution = current_set.copy()
        
    if avail_mask == 0:
        return
        
    # Upper bound: sum of all remaining weights
    # This is a relaxation.
    ub = current_score + sum_weights(avail_mask)
    if ub <= global_best:
        return
        
    # Select vertex to branch on
    u = get_best_vertex(avail_mask)
    if u == -1:
        return
        
    # Branch 1: Include u
    # New available: vertices in avail_mask that are coprime to u, excluding u itself
    # compat_mask[u] includes u itself (gcd(u,u)=u, wait. gcd(u,u)=u. If u>1, gcd>1. If u=1, gcd=1.
    # My compat_mask definition: gcd(i,j)==1.
    # If i=j=1, gcd(1,1)=1. So 1 is in compat_mask[1].
    # If i=j=2, gcd(2,2)=2. So 2 is NOT in compat_mask[2].
    # So for u>1, u is not in compat_mask[u].
    # For u=1, u is in compat_mask[1].
    # We should always remove u from the available set to prevent re-selection.
    
    new_avail_include = avail_mask & compat_mask[u] & ~(1 << u)
    search(new_avail_include, current_score + weights[u], current_set + [u])
    
    # Branch 2: Exclude u
    new_avail_exclude = avail_mask & ~(1 << u)
    search(new_avail_exclude, current_score, current_set)

def construct():
    global global_best, best_solution
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Initialize
    global_best = 0.0
    best_solution = []
    
    # Start search with all vertices available
    # To help find a good solution early, we can do a greedy pass first
    # Greedy: pick highest weight, remove conflicts, repeat.
    greedy_set = []
    avail = all_vertices_mask
    while avail:
        u = get_best_vertex(avail)
        if u == -1: break
        greedy_set.append(u)
        global_best = max(global_best, sum(weights[x] for x in greedy_set))
        if global_best > 0: # Update best solution if greedy is better
             # Actually we want to keep the best found so far
             pass
        avail = avail & compat_mask[u] & ~(1 << u)
        
    # Set initial best from greedy
    if greedy_set:
        global_best = sum(weights[x] for x in greedy_set)
        best_solution = greedy_set.copy()
        
    # Now run B&B
    # We need to be careful with recursion depth. Max depth 511. Python default is 1000. OK.
    # But B&B can be slow.
    
    # To improve B&B, we can order the branching.
    # The current search explores "Include" first.
    
    # Check time limit periodically?
    # Python doesn't have easy timeout in recursion.
    # We can check time in the search function.
    
    import time as _time
    def timed_search(avail_mask, current_score, current_set):
        if _time.time() > deadline:
            return
        search(avail_mask, current_score, current_set)
        
    # Recursively call with time check wrapper?
    # Better to just call search and rely on it finishing or being fast.
    # Given N=511, it might be slow if pruning is weak.
    
    # Let's try to run it.
    search(all_vertices_mask, 0.0, [])
    
    # If it didn't finish or found nothing better, return greedy?
    # search updates global_best.
    
    return best_solution

# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    N = 511
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / (512 - i)

    conflict_mask = [0] * (N + 1)
    compat_mask = [0] * (N + 1)

    all_vertices_mask = 0
    for i in range(1, N + 1):
        all_vertices_mask |= (1 << i)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if math.gcd(i, j) > 1:
                conflict_mask[i] |= (1 << j)
            else:
                compat_mask[i] |= (1 << j)

    def get_best_vertex(avail_mask):
        best_v = -1
        best_w = -1.0
        m = avail_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if weights[v] > best_w:
                best_w = weights[v]
                best_v = v
            m ^= lsb
        return best_v

    def sum_weights(avail_mask):
        total = 0.0
        m = avail_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            total += weights[v]
            m ^= lsb
        return total

    global_best = 0.0
    best_solution = []
    
    # Greedy Initialization to set a good lower bound
    greedy_set = []
    avail = all_vertices_mask
    while avail:
        u = get_best_vertex(avail)
        if u == -1: break
        greedy_set.append(u)
        avail = avail & compat_mask[u] & ~(1 << u)
    
    if greedy_set:
        global_best = sum(weights[x] for x in greedy_set)
        best_solution = greedy_set.copy()
        
    # Branch and Bound
    # To handle time limits in recursion, we check time at each node.
    def search(avail_mask, current_score, current_set):
        if time.time() > deadline:
            return
            
        if current_score > global_best:
            global_best = current_score
            best_solution = current_set.copy()
            
        if avail_mask == 0:
            return
            
        ub = current_score + sum_weights(avail_mask)
        if ub <= global_best:
            return
            
        u = get_best_vertex(avail_mask)
        if u == -1:
            return
            
        # Branch Include
        new_avail_include = avail_mask & compat_mask[u] & ~(1 << u)
        search(new_avail_include, current_score + weights[u], current_set + [u])
        
        # Check time before Exclude branch
        if time.time() > deadline:
            return
            
        # Branch Exclude
        new_avail_exclude = avail_mask & ~(1 << u)
        search(new_avail_exclude, current_score, current_set)

    search(all_vertices_mask, 0.0, [])
    
    return best_solution
# EVOLVE-BLOCK-END
