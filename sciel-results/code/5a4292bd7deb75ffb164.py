import time
import random
import numpy as np

N = 16
NUM_POINTS = N * N

# Precompute points
POINTS = [(x, y) for x in range(N) for y in range(N)]

# Precompute squared distances
dist2 = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i, j] = d
        dist2[j, i] = d

# Precompute conflict pairs
conflict_pairs = [[] for _ in range(NUM_POINTS)]

for p in range(NUM_POINTS):
    for i in range(NUM_POINTS):
        if i == p:
            continue
        for j in range(i + 1, NUM_POINTS):
            if j == p:
                continue
            d_pi = dist2[p, i]
            d_pj = dist2[p, j]
            d_ij = dist2[i, j]
            
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                if i < j:
                    conflict_pairs[p].append((i, j))
                else:
                    conflict_pairs[p].append((j, i))

def can_add(in_S, p):
    """Check if point p can be added to set in_S without creating an isosceles triangle."""
    for i, j in conflict_pairs[p]:
        if in_S[i] and in_S[j]:
            return False
    return True

def is_maximal(in_S):
    """Check if the set represented by boolean array in_S is maximal."""
    for p in range(NUM_POINTS):
        if in_S[p]:
            continue
        blocked = False
        for i, j in conflict_pairs[p]:
            if in_S[i] and in_S[j]:
                blocked = True
                break
        if not blocked:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Use bitsets for fast operations
    # blocked_by[p] is a bitset of points that block p (i.e., p cannot be added if any two of these are in S)
    # Actually, let's think differently:
    # For each point p, define block_mask[p] as a bitset where bit q is set if q is "in conflict" with p
    # But the conflict is: p is blocked if there exist i,j in S such that (p,i,j) is isosceles
    # This is not a simple pairwise conflict.
    
    # Let me use a different approach: 
    # For each point p, precompute which pairs (i,j) block it.
    # Then p is blocked if any such pair has both i and j in S.
    
    # For branch and bound, let's use a simpler formulation:
    # We want to find a minimal maximal independent set in the "isosceles triangle" hypergraph.
    # A set S is valid if no three points form an isosceles triangle.
    # S is maximal if every point not in S is "blocked" by S.
    
    # Let's try a different structural approach: 
    # Use simulated annealing with bitset-based feasibility checks.
    
    # Precompute for each point p, the list of pairs that block it
    # blocked_pairs[p] = list of (i, j) such that p is blocked if both i and j are in S
    
    # Actually, let's use a completely different method: 
    # Start from a random valid set, then use hill climbing with swaps.
    
    # First, get a good initial solution using greedy
    for restart in range(30):
        if time.time() > deadline:
            break
        
        random.seed(2000 + restart)
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        in_S = np.zeros(NUM_POINTS, dtype=bool)
        current_set = []
        
        for p in order:
            if can_add(in_S, p):
                in_S[p] = True
                current_set.append(p)
        
        # Make it maximal
        for p in order:
            if not in_S[p] and can_add(in_S, p):
                in_S[p] = True
                current_set.append(p)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        # Now do local search: try to remove points
        current_in_S = in_S.copy()
        current_list = list(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_list)
            
            for p in current_list:
                if time.time() > deadline:
                    break
                if current_in_S[p]:
                    current_in_S[p] = False
                    if p in current_list:
                        current_list.remove(p)
                    
                    if is_maximal(current_in_S):
                        improved = True
                        if len(current_list) < best_size:
                            best_size = len(current_list)
                            best_set = list(current_list)
                        break
                    else:
                        current_in_S[p] = True
                        if p not in current_list:
                            current_list.append(p)
        
        # Try swap-based local search
        if time.time() < deadline - 10:
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Try to swap: remove p, add q
                random.shuffle(current_list)
                found_swap = False
                
                for p in current_list[:]:
                    if time.time() > deadline:
                        break
                    
                    if not current_in_S[p]:
                        continue
                    
                    # Try removing p
                    current_in_S[p] = False
                    temp_list = [x for x in current_list if x != p]
                    
                    # Try adding some q not in S
                    candidates = [q for q in range(NUM_POINTS) if not current_in_S[q]]
                    random.shuffle(candidates)
                    
                    for q in candidates:
                        if can_add(current_in_S, q):
                            # Check if removing p and adding q keeps it maximal
                            current_in_S[q] = True
                            if is_maximal(current_in_S):
                                # Accept the swap
                                current_list = temp_list + [q]
                                if len(current_list) < best_size:
                                    best_size = len(current_list)
                                    best_set = list(current_list)
                                found_swap = True
                                break
                            else:
                                current_in_S[q] = False
                    
                    if found_swap:
                        break
                    
                    # Restore p
                    current_in_S[p] = True
                
                if not found_swap:
                    break

    if best_set is None:
        in_S = np.zeros(NUM_POINTS, dtype=bool)
        current_set = []
        for p in range(NUM_POINTS):
            if can_add(in_S, p):
                in_S[p] = True
                current_set.append(p)
        return [POINTS[i] for i in current_set]
        
    return [POINTS[i] for i in best_set]
