import random
import time
import math

N = 16
num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# Precompute distances squared
dist = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    x1, y1 = coords[i]
    for j in range(i + 1, num_points):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

# Precompute isosceles triangles
# iso_pairs[i][j] is a set of k such that (i, j, k) is isosceles
# We need this to check if adding k to {i, j} creates an isosceles triangle.
# And for maximality: for every v not in S, there exist i, j in S such that v in iso_pairs[i][j].

# Optimization: Instead of storing sets of k for every pair (i,j), 
# we can precompute for each point v, the set of pairs (i,j) that block it.
# blocked_by[v] = set of (i, j) with i < j such that {v, i, j} is isosceles.
# This is useful for the maximality check: v is blocked if any pair in blocked_by[v] is in S.

blocked_by = [set() for _ in range(num_points)]
# Also, for the "no isosceles" constraint check during search, 
# we need to know if adding v to S creates an isosceles triangle.
# This happens if there exist i, j in S such that v in iso_pairs[i][j].
# Let conflict[v] = set of (i, j) such that {v, i, j} is isosceles.
# This is the same as blocked_by[v].
# So, v can be added to S iff S contains no pair from conflict[v].

# Let's build conflict[v] as a list of (i, j) pairs.
for v in range(num_points):
    for i in range(num_points):
        if i == v: continue
        for j in range(i + 1, num_points):
            if j == v: continue
            # Check if {v, i, j} is isosceles
            d_v_i = dist[v][i]
            d_v_j = dist[v][j]
            d_i_j = dist[i][j]
            if d_v_i == d_v_j or d_v_i == d_i_j or d_v_j == d_i_j:
                blocked_by[v].add((i, j))

# To speed up "is S isosceles-free?" check:
# For a set S, it is isosceles-free if for all v in S, S contains no pair from blocked_by[v].
# This is still O(|S|^3) naive. 
# Better: Precompute for each pair (i, j) the set of points k that conflict with it.
# conflict_pair[i][j] = set of k such that {i, j, k} is isosceles.
# Then S is isosceles-free iff for all pairs (i, j) in S, conflict_pair[i][j] is disjoint from S.
# Or: for all k in S, conflict_pair[k][i] is disjoint from S\{k} for all i in S.

# Let's use the pair-based conflict for checking validity of a set.
# But for local search, we need to evaluate the cost of flipping a bit.
# State: bitmask of selected points.
# Objective: Minimize popcount(S) subject to:
# 1. S is isosceles-free.
# 2. S is maximal (every v not in S is blocked by some pair in S).

# Penalty Function for Simulated Annealing:
# Cost = alpha * |S| + beta * (# isosceles triangles in S) + gamma * (# unblocked points not in S)
# We want to find a state with Cost 0 (valid and maximal) and minimal |S|.
# If we can't find Cost 0, we return the best valid one found or the one with lowest penalty.

# Precompute conflict_pair for fast triangle count?
# Actually, for SA, we need to update the penalty incrementally when flipping a bit.
# Flipping bit v:
# - If adding v:
#   - Increase |S| by 1.
#   - New triangles formed: pairs (i, j) in S such that v in conflict_pair[i][j].
#     This is equivalent to counting how many pairs in S are in blocked_by[v].
#   - Maximal constraint: v is now in S, so it no longer needs to be blocked.
#     But adding v might block other points? No, blocking is about pairs in S.
#     Adding v creates new pairs (v, u) for u in S.
#     These new pairs might block points w not in S.
#     So the number of unblocked points might decrease.
#     Specifically, w is now blocked if (w, v, u) is isosceles for some u in S.
#     i.e., (v, u) is in blocked_by[w].
#     So we need to check which w not in S are newly blocked by pairs involving v.
#     
# This incremental update is complex. 
# Given N=16, num_points=256.
# A full evaluation of the penalty for a state S takes O(|S|^2 * avg_degree) or similar.
# With SA, we might do 100k-1M steps. 
# If each step takes 1ms, 1M steps = 1000s. Too slow.
# We need a faster evaluation or fewer steps.

# Alternative: Use the "Greedy + Local Search" approach but with a different neighborhood.
# The current program uses "remove point if maximal".
# This is a "descent" on size, but it gets stuck if removing a point breaks maximality.
# To escape, we need to allow temporary non-maximality or non-isosceles-free states?
# No, we must end with a valid set.
# 
# Let's try a different search: 
# 1. Start with a random valid maximal set (from greedy).
# 2. Perform "2-opt" like moves: Remove point p, add point q.
#    - Check if S \ {p} U {q} is isosceles-free.
#    - Check if S \ {p} U {q} is maximal.
#    - If yes, and |S| doesn't increase (it stays same), we might not improve size.
#    - We want to DECREASE size. So we need to remove p without adding q? 
#      That's the current strategy.
#    - To decrease size, we must find a point p such that S\{p} is still maximal.
#    - If no such p exists, we are in a local minimum.
#    - To escape, we can try to "swap" p with q, then remove q? 
#      Or remove p and q, add r? 
#
# Let's implement a "Swap and Shrink" local search.
# 1. Try to remove p. If maximal, do it.
# 2. If stuck, try to swap p with q (keep size same). 
#    This changes the structure of S, potentially allowing future removals.
# 3. After a swap, try to remove points again.
#
# This is a standard local search for minimum maximal independent set type problems.

def is_isosceles_free(S_set):
    """Check if S_set contains no isosceles triangles."""
    s_list = list(S_set)
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i + 1, len(s_list)):
            b = s_list[j]
            # Check if any k in S_set is in conflict_pair[a][b]
            # We don't have conflict_pair precomputed as sets of k.
            # We have blocked_by[k] which is sets of pairs.
            # So for each k in S_set, check if (a,b) in blocked_by[k].
            # This is O(|S|^3) if done naively.
            # Optimization: 
            # For fixed a, b, we need to know if S_set intersects conflict_pair[a][b].
            # Let's precompute conflict_pair[a][b] as a bitmask or set of points.
            pass
    # Let's rebuild precomputation for speed.
    # conflict_pair[i][j] = set of k such that {i,j,k} is isosceles.
    return True

# Let's precompute conflict_pair properly.
conflict_pair = [[set() for _ in range(num_points)] for __ in range(num_points)]
for i in range(num_points):
    for j in range(i + 1, num_points):
        for k in range(num_points):
            if k == i or k == j:
                continue
            d1 = dist[i][j]
            d2 = dist[i][k]
            d3 = dist[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                conflict_pair[i][j].add(k)
                conflict_pair[j][i].add(k)

# Now is_isosceles_free can be:
# For all pairs (i, j) in S, conflict_pair[i][j] is disjoint from S.
# We can optimize this by checking for each k in S, if there is any pair (i, j) in S 
# such that k in conflict_pair[i][j].
# This is still slow.
# 
# Better: Use bitmasks for S.
# S_mask is an integer.
# For a pair (i, j), conflict_mask[i][j] is a bitmask of points k.
# S is isosceles-free iff for all i, j in S, (S_mask & conflict_mask[i][j]) == 0.
# 
# Let's precompute conflict_mask.
conflict_mask = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    for j in range(i + 1, num_points):
        mask = 0
        for k in conflict_pair[i][j]:
            mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def is_isosceles_free_mask(S_mask, s_list):
    """
    S_mask: integer bitmask of selected points.
    s_list: list of indices of selected points (for iteration).
    """
    for idx_a, a in enumerate(s_list):
        for b in s_list[idx_a + 1:]:
            if (S_mask & conflict_mask[a][b]) != 0:
                return False
    return True

def is_maximal_mask(S_mask, s_list):
    """
    Check if every point v NOT in S_mask is blocked by some pair in S.
    v is blocked if there exist i, j in S such that v in conflict_pair[i][j].
    i.e., S_mask has at least one pair (i, j) from blocked_by[v].
    """
    # For each v not in S:
    # Check if any pair in blocked_by[v] is fully contained in S.
    # blocked_by[v] is a set of (i, j).
    # We can precompute for each v, the list of pairs.
    # To check if a pair (i, j) is in S, we check (S_mask & (1<<i)) and (S_mask & (1<<j)).
    
    # Optimization: 
    # If S is small, this is fast.
    # If S is large, this is slow.
    # But we are minimizing S, so S should get smaller.
    
    for v in range(num_points):
        if (S_mask >> v) & 1:
            continue
        # v is not in S. Check if it is blocked.
        blocked = False
        for (i, j) in blocked_by[v]:
            if ((S_mask >> i) & 1) and ((S_mask >> j) & 1):
                blocked = True
                break
        if not blocked:
            return False
    return True

def generate_initial_set(seed):
    """Generate a random maximal no-isosceles set using greedy."""
    random.seed(seed)
    order = list(range(num_points))
    random.shuffle(order)
    
    S_mask = 0
    s_list = []
    
    for p in order:
        # Check if p can be added
        valid = True
        # p conflicts with (i, j) if p in conflict_pair[i][j]
        # So we need to check if any pair (i, j) in s_list has p in conflict_pair[i][j].
        # Equivalently, for each i in s_list, check if p in conflict_pair[i][j] for any j in s_list.
        # Or simpler: For each i in s_list, if (1 << p) is in conflict_mask[i][j] for some j in s_list.
        # Let's just check pairs.
        for idx_a, a in enumerate(s_list):
            for b in s_list[idx_a + 1:]:
                if (conflict_mask[a][b] >> p) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_mask |= (1 << p)
            s_list.append(p)
            
    # Repair to maximal
    changed = True
    while changed:
        changed = False
        for p in range(num_points):
            if (S_mask >> p) & 1:
                continue
            # Try to add p
            valid = True
            for idx_a, a in enumerate(s_list):
                for b in s_list[idx_a + 1:]:
                    if (conflict_mask[a][b] >> p) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_mask |= (1 << p)
                s_list.append(p)
                changed = True
                
    return S_mask, s_list

def local_search(S_mask, s_list, deadline):
    """
    Local search to minimize |S|.
    1. Try to remove points.
    2. If stuck, try to swap points to change structure.
    """
    best_mask = S_mask
    best_size = len(s_list)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove points
        random.shuffle(s_list)
        for p in s_list:
            if not ((S_mask >> p) & 1):
                continue
            new_mask = S_mask ^ (1 << p)
            new_list = [x for x in s_list if x != p]
            
            # Check validity of new set
            if is_isosceles_free_mask(new_mask, new_list) and is_maximal_mask(new_mask, new_list):
                S_mask = new_mask
                s_list = new_list
                improved = True
                if len(s_list) < best_size:
                    best_size = len(s_list)
                    best_mask = S_mask
                break # Restart removal loop with new set
        
        if not improved:
            # Phase 2: Try to swap points
            # Pick a random point p in S and a random point q not in S.
            # Swap p with q.
            # Check if new set is valid.
            # If valid, accept. This might allow future removals.
            
            if not s_list:
                break
                
            p = random.choice(s_list)
            # Pick q not in S
            # To avoid expensive sampling, we can pick a random index and check.
            # Or maintain a list of outside points.
            # For simplicity, pick random q in [0, N*N) and check if not in S.
            # If in S, retry.
            q_found = False
            for _ in range(100):
                q = random.randint(0, num_points - 1)
                if not ((S_mask >> q) & 1):
                    q_found = True
                    break
            
            if not q_found:
                break
                
            new_mask = S_mask ^ (1 << p) ^ (1 << q)
            new_list = [x for x in s_list if x != p]
            new_list.append(q)
            
            # Check validity
            if is_isosceles_free_mask(new_mask, new_list) and is_maximal_mask(new_mask, new_list):
                S_mask = new_mask
                s_list = new_list
                # Do not update best_size here, size is same.
                # But structure changed.
                # We don't break, we continue to try removals in next iteration.
                # Actually, let's break to the top of the while loop to try removals again.
                continue
                
            # If swap failed, try another swap? Or give up for this iteration.
            # To avoid infinite loop of failed swaps, we can limit swaps per iteration.
            # But for now, we just do one swap attempt per iteration.
            # If it fails, we might be stuck. 
            # We can add a "kick" move: remove 2 points, add 2 points?
            # For now, just one swap.
            
    return best_mask, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = num_points
    best_mask = 0
    
    seed = 0
    while time.time() < deadline:
        S_mask, s_list = generate_initial_set(seed)
        
        # If initial set is already small, maybe skip local search?
        # No, local search is the main improvement.
        
        final_mask, final_size = local_search(S_mask, s_list, deadline)
        
        if final_size < best_size:
            best_size = final_size
            best_mask = final_mask
            
        seed += 1
        
    # Convert best_mask to list of coords
    result = []
    for i in range(num_points):
        if (best_mask >> i) & 1:
            result.append(coords[i])
            
    return result

# Fix typo in precomputation
# The code above had a syntax error in the first precomputation block (d2 == d d3).
# I will rewrite the precomputation correctly in the final output.

# Re-verify logic:
# 1. Precompute dist.
# 2. Precompute blocked_by[v] (pairs (i,j) that form isosceles with v).
# 3. Precompute conflict_mask[i][j] (bitmask of k such that (i,j,k) is isosceles).
# 4. Greedy generation.
# 5. Local search: Remove if maximal, Swap if stuck.

# One optimization: In is_maximal_mask, iterating over all v not in S and checking all pairs in blocked_by[v] is O(N^3) roughly.
# With N=16, num_points=256.
# |S| is maybe 50-100.
# Outside points ~ 150-200.
# blocked_by[v] size? 
# Number of pairs (i,j) is ~ 256^2/2 = 32k.
# Average blocked_by size is small.
# So it should be fast enough.

# Let's ensure the deadline is respected.
