import time
import random

N = 512
# Precompute 4-APs
# Represent APs as tuples of 4 indices (0-indexed for bitmasks)
aps = []
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a = start - 1
        b = a + d
        c = b + d
        dd = c + d
        aps.append((a, b, c, dd))

# For each vertex, store the indices of APs it participates in
vertex_aps = [[] for _ in range(N)]
for i, ap in enumerate(aps):
    for v in ap:
        vertex_aps[v].append(i)

# Precompute bitmasks for APs
ap_masks = []
for a, b, c, d in aps:
    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
    ap_masks.append(mask)

# Helper to check if a set S (bitmask) can add vertex v
# Returns True if adding v does not complete a 4-AP with existing S
def can_add(S, v):
    for ap_idx in vertex_aps[v]:
        mask = ap_masks[ap_idx]
        # Check if the other 3 bits are in S
        # mask without v
        mask_wo_v = mask ^ (1 << v)
        if (S & mask_wo_v) == mask_wo_v:
            return False
    return True

# Helper to get all blocked vertices given S
# A vertex v is blocked if adding it would complete a 4-AP with S
def get_blocked(S):
    blocked = 0
    for i, mask in enumerate(ap_masks):
        # Count bits in S that are in this AP
        # If 3 are in S, the 4th is blocked
        # We can check each AP
        # Optimization: Only check if S intersects mask in 3 bits?
        # Python int bit_count is fast
        # But we need to identify which bit is missing
        inter = S & mask
        if inter.bit_count() == 3:
            # The missing bit is mask ^ inter
            missing = mask ^ inter
            blocked |= missing
    return blocked

def greedy_initial():
    S = 0
    # Order by degree (number of APs) descending
    order = sorted(range(N), key=lambda x: len(vertex_aps[x]), reverse=True)
    for v in order:
        if can_add(S, v):
            S |= (1 << v)
    return S

def local_search(S, max_iterations=10000):
    current_size = S.bit_count()
    best_S = S
    best_size = current_size
    for _ in range(max_iterations):
        # Try to add a random vertex
        # Or try swaps
        # Let's try single moves: remove one, add another?
        # Or just try to add a vertex from the "almost blocked" set?
        
        # Simple random restart of local search logic
        # Pick a random vertex not in S, try to add it. If fails, pick a random vertex in S to remove, then try to add?
        
        # Let's try: Pick random v not in S. If can_add, add it.
        # If not, pick random u in S, remove u, then try to add v?
        
        v = random.randint(0, N - 1)
        if not (S & (1 << v)):
            if can_add(S, v):
                S |= (1 << v)
                if S.bit_count() > best_size:
                    best_size = S.bit_count()
                    best_S = S
        else:
            # Try to swap?
            # Remove v, try to add a random other w
            S_temp = S ^ (1 << v)
            w = random.randint(0, N - 1)
            if not (S_temp & (1 << w)):
                if can_add(S_temp, w):
                    S_new = S_temp | (1 << w)
                    if S_new.bit_count() > best_size:
                        best_size = S_new.bit_count()
                        best_S = S_new
                    # Accept move if size is same or better? 
                    # For 4-AP free, size usually doesn't increase with 1-for-1 swap unless we fix a conflict.
                    # But here we are just swapping.
                    # Let's just update best.
                    S = S_new
    return best_S

def branch_and_bound(S, A, best_size, best_S, time_limit, start_time):
    # S: current selected set (int)
    # A: available candidates (int)
    # best_size: current best known size
    # best_S: current best set
    
    if time.time() - start_time > time_limit:
        return best_size, best_S
    
    # Prune
    if S.bit_count() + A.bit_count() <= best_size:
        return best_size, best_S
    
    # If A is empty
    if A == 0:
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S
        return best_size, best_S
    
    # Get blocked vertices
    blocked = get_blocked(S)
    A &= ~blocked
    
    if A == 0:
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S
        return best_size, best_S
    
    # Pick a vertex to branch on
    # Heuristic: Pick the vertex in A with the highest "degree" in the remaining conflict structure?
    # For simplicity, pick the first set bit, or the one with most APs in A?
    # Let's pick the vertex v in A that participates in the most APs with other vertices in A.
    # This is expensive. Let's just pick the first one for now, or random.
    # Better: Pick the one with highest global degree.
    
    # Get list of vertices in A
    # To avoid iterating 512 bits every time, we can just pick a random one or the lowest set bit.
    # Lowest set bit is fast.
    v = (A & -A).bit_length() - 1
    
    # Branch 1: Include v
    if can_add(S, v):
        new_S = S | (1 << v)
        new_A = A ^ (1 << v)
        best_size, best_S = branch_and_bound(new_S, new_A, best_size, best_S, time_limit, start_time)
    
    # Branch 2: Exclude v
    new_A = A ^ (1 << v)
    best_size, best_S = branch_and_bound(S, new_A, best_size, best_S, time_limit, start_time)
    
    return best_size, best_S

def construct():
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    # 1. Greedy Initial
    S0 = greedy_initial()
    best_size = S0.bit_count()
    best_S = S0
    
    # 2. Local Search on Greedy
    S_ls = local_search(S0, max_iterations=20000)
    if S_ls.bit_count() > best_size:
        best_size = S_ls.bit_count()
        best_S = S_ls
        
    # 3. Branch and Bound
    # Use the best found so far as the bound
    # A: all vertices not in best_S
    A = ((1 << N) - 1) ^ best_S
    
    # Run B&B with time limit
    best_size, best_S = branch_and_bound(best_S, A, best_size, best_S, time_limit, start_time)
    
    # 4. Final Local Search on B&B result
    if time.time() - start_time < time_limit:
        S_final = local_search(best_S, max_iterations=10000)
        if S_final.bit_count() > best_size:
            best_size = S_final.bit_count()
            best_S = S_final
    
    # Convert bitmask to sorted list of 1-indexed values
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
