import itertools
import random
import time

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute squared distances
# dist[i][j]
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

# Precompute isosceles pairs for each point p
# iso_pairs[p] is a list of (a, b) such that (p, a, b) is isosceles
# We need a < b to avoid duplicates in the pair list for a specific p?
# Actually, for a fixed p, the pair (a,b) is unordered.
# Triangle (p,a,b) is isosceles if:
# dist(p,a) == dist(p,b)
# dist(p,a) == dist(a,b)
# dist(p,b) == dist(a,b)

iso_pairs = [[] for _ in range(NUM_POINTS)]

for p in range(NUM_POINTS):
    # We need to find all pairs (a, b) with a != p, b != p, a != b
    # such that triangle (p,a,b) is isosceles.
    # To avoid storing duplicates (a,b) and (b,a), we can just store the tuple.
    # In can_add, we check if a in S and b in S. Order doesn't matter for membership.
    
    # Optimization:
    # 1. dist(p,a) == dist(p,b): Group points by distance from p.
    #    For each distance d, let P_d be the set of points at distance d from p.
    #    Any pair (a,b) from P_d forms an isosceles triangle with p.
    # 2. dist(p,a) == dist(a,b): For each a != p, find b such that dist(a,b) == dist(p,a).
    #    This is harder to precompute efficiently for all p.
    #    However, NUM_POINTS is only 256.
    #    Total pairs is ~ 32k.
    #    For each p, checking all pairs (a,b) is 256^2 ~ 65k operations.
    #    256 * 65k ~ 16M operations. This is fine for startup.

    for a in range(NUM_POINTS):
        if a == p: continue
        for b in range(a+1, NUM_POINTS):
            if b == p: continue
            d_pa = dist[p][a]
            d_pb = dist[p][b]
            d_ab = dist[a][b]
            
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                iso_pairs[p].append((a, b))

def can_add(p, s):
    """
    s is an integer bitmask representing the set.
    Returns True if p can be added to S without forming an isosceles triangle.
    """
    for a, b in iso_pairs[p]:
        if ((s >> a) & 1) and ((s >> b) & 1):
            return False
    return True

def is_maximal(s):
    """
    s is an integer bitmask representing the set.
    Returns True if S is maximal (no other point can be added).
    """
    for p in range(NUM_POINTS):
        if not ((s >> p) & 1):
            if can_add(p, s):
                return False
    return True

def greedy_construction(order):
    s = 0
    for p in order:
        if can_add(p, s):
            s |= (1 << p)
    return s

def local_search(s, time_limit=10.0):
    """
    Attempts to reduce the size of the set s while maintaining maximality.
    Uses removal and swap moves.
    """
    start_time = time.time()
    current_s = s
    current_size = bin(current_s).count('1')
    
    # To avoid infinite loops on swaps that don't reduce size, 
    # we track visited states or limit swaps.
    # However, removals strictly reduce size, so they are always good.
    # Swaps maintain size but might enable future removals.
    
    # We will prioritize removals.
    # If no removal is possible, we try swaps to escape local minima.
    
    max_swaps_per_removal = 50 # Limit swaps between removals to prevent cycling
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove a point
        removed = False
        # Iterate over set bits
        temp_s = current_s
        # Randomize order of removal attempts?
        # We can extract indices and shuffle them.
        indices = [i for i in range(NUM_POINTS) if (current_s >> i) & 1]
        random.shuffle(indices)
        
        for i in indices:
            new_s = current_s & ~(1 << i)
            if is_maximal(new_s):
                current_s = new_s
                current_size -= 1
                removed = True
                break
        
        if removed:
            continue # Restart the loop to try more removals
        
        # 2. If no removal, try swaps
        # Pick a random point u in S and v not in S
        # Check if (S - {u}) U {v} is valid.
        # If valid, check if maximal.
        # If maximal, accept.
        
        swapped = False
        for _ in range(max_swaps_per_removal):
            # Pick u from S
            indices_s = [i for i in range(NUM_POINTS) if (current_s >> i) & 1]
            if not indices_s: break
            u = random.choice(indices_s)
            
            # Pick v from not S
            # Create list of indices not in S
            # This is expensive if done every time. 
            # But S is small, so not-S is large.
            # We can just pick a random v and check if it's in S.
            # Or precompute not_s_mask?
            # Let's just pick random v and check.
            v = random.randint(0, NUM_POINTS - 1)
            if (current_s >> v) & 1:
                continue # v is in S, skip
            
            new_s = (current_s & ~(1 << u)) | (1 << v)
            
            # Check validity of new_s
            # Validity: no isosceles triangles.
            # We can check this by verifying that for every p in new_s, 
            # can_add(p, new_s without p) is true? 
            # No, can_add checks if p can be added to a set.
            # To check if a set is valid, we need to ensure no 3 points form isosceles.
            # This is equivalent to: for all p in S, can_add(p, S \ {p}) is True?
            # Actually, can_add(p, S) checks if p conflicts with pairs in S.
            # If we construct S greedily, it is valid.
            # If we do a swap, we must ensure the new set is valid.
            
            # Efficient validity check for a candidate set T (bitmask):
            # For each p in T, check if can_add(p, T without p) is True.
            # This is O(|T| * |iso_pairs[p]|).
            
            valid = True
            temp_t = new_s
            # Iterate through points in new_s
            t_indices = [i for i in range(NUM_POINTS) if (new_s >> i) & 1]
            for p in t_indices:
                # Check if p conflicts with other points in T
                # can_add(p, T without p)
                t_without_p = new_s & ~(1 << p)
                if not can_add(p, t_without_p):
                    valid = False
                    break
            
            if valid:
                if is_maximal(new_s):
                    current_s = new_s
                    swapped = True
                    break
        
        if not swapped and not removed:
            # Stuck in local minimum? 
            # We could try a 2-swap or just break and let the outer loop restart.
            # Given the time limit, breaking and restarting greedy is often better.
            break

    return current_s

def construct():
    start_time = time.time()
    deadline = start_time + 250.0 # Safety margin
    
    best_s = 0
    best_size = float('inf')
    
    restarts = 0
    while time.time() < deadline - 20.0:
        restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        s = greedy_construction(order)
        
        # Local search to minimize size
        s = local_search(s, time_limit=10.0)
        
        size = bin(s).count('1')
        if size < best_size:
            best_size = size
            best_s = s
            
        # If we find a very small set, we might spend more time polishing it?
        # For now, just keep searching.
        
    if best_s == 0:
        # Fallback: deterministic greedy
        s = greedy_construction(list(range(NUM_POINTS)))
        best_s = s
        best_size = bin(s).count('1')

    # Convert bitmask to list of points
    result = []
    for i in range(NUM_POINTS):
        if (best_s >> i) & 1:
            result.append(POINTS[i])
            
    return result
