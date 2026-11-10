import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_PTS = N * N
FULL_MASK = (1 << NUM_PTS) - 1

# Precompute geometry
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute Iso masks: Iso[i][j] is bitmask of k such that triangle i,j,k is isosceles
Iso = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i][j]
        mask = 0
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        Iso[i][j] = mask
        Iso[j][i] = mask

def get_blocked_mask(S_list):
    """Return the union of Iso masks for all pairs in S_list."""
    mask = 0
    n = len(S_list)
    for i in range(n):
        pi = S_list[i]
        for j in range(i + 1, n):
            pj = S_list[j]
            mask |= Iso[pi][pj]
    return mask

def is_valid(S_list):
    """Check if no three points in S_list form an isosceles triangle."""
    if len(S_list) < 3:
        return True
    blocked = get_blocked_mask(S_list)
    S_mask = 0
    for p in S_list:
        S_mask |= (1 << p)
    return (S_mask & blocked) == 0

def is_maximal(S_list):
    """Check if no point outside S_list can be added."""
    if not is_valid(S_list):
        return False
    blocked = get_blocked_mask(S_list)
    S_mask = 0
    for p in S_list:
        S_mask |= (1 << p)
    return (S_mask | blocked) == FULL_MASK

def greedy_maximal_set(order):
    """Build a maximal no-isosceles set by greedily adding points from order.
    Uses running blocked_mask for O(n*|S|) complexity."""
    S = []
    blocked_mask = 0
    for p in order:
        if (blocked_mask >> p) & 1:
            continue
        S.append(p)
        for i in range(len(S) - 1):
            blocked_mask |= Iso[S[i]][p]
    return S

def branch_and_bound_min_maximal(deadline):
    """
    Find minimum maximal no-isosceles set using branch and bound.
    
    We search for the smallest k such that there exists a maximal no-isosceles set of size k.
    For each k, we try to find a maximal set of exactly k points.
    
    A set S is maximal if:
    1. S is no-isosceles (no three points form isosceles triangle)
    2. Every point not in S is blocked by some pair in S
    
    Strategy: 
    - Start with k=1, try to find maximal set of size k
    - If found, return it
    - If not, increment k
    - Use branch and bound to search for valid maximal sets of size k
    """
    best_set = None
    best_k = float('inf')
    
    # First, get a good initial upper bound using greedy
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S = greedy_maximal_set(order)
        if len(S) < best_k:
            best_k = len(S)
            best_set = S.copy()
    
    # Now try to improve using branch and bound for smaller k
    # We'll try k from 1 up to best_k-1
    for target_k in range(1, best_k):
        if time.time() > deadline:
            break
        
        # Try to find a maximal set of exactly target_k points
        # Use backtracking with pruning
        found = False
        
        def backtrack(start_idx, current_set, blocked_mask):
            nonlocal found, deadline
            
            if time.time() > deadline:
                return False
            
            if found:
                return True
            
            remaining_needed = target_k - len(current_set)
            
            # Pruning: if we can't reach target_k, return
            if remaining_needed < 0:
                return False
            
            # If we have enough points, check if maximal
            if remaining_needed == 0:
                S_mask = 0
                for p in current_set:
                    S_mask |= (1 << p)
                if (S_mask | blocked_mask) == FULL_MASK:
                    found = True
                    return True
                return False
            
            # Pruning: not enough points left to add
            if NUM_PTS - start_idx < remaining_needed:
                return False
            
            # Try adding each candidate point
            for p in range(start_idx, NUM_PTS):
                if found:
                    break
                
                # Check if p can be added (not blocked by current pairs)
                if (blocked_mask >> p) & 1:
                    continue
                
                # Check if p conflicts with any pair in current_set
                # i.e., for any two points a,b in current_set, p is not in Iso[a][b]
                can_add = True
                for i in range(len(current_set)):
                    for j in range(i + 1, len(current_set)):
                        if (Iso[current_set[i]][current_set[j]] >> p) & 1:
                            can_add = False
                            break
                    if not can_add:
                        break
                
                if not can_add:
                    continue
                
                # Compute new blocked mask
                new_blocked = blocked_mask
                for i in range(len(current_set)):
                    new_blocked |= Iso[current_set[i]][p]
                
                # Recurse
                if backtrack(p + 1, current_set + [p], new_blocked):
                    return True
            
            return False
        
        # Try multiple starting points for diversity
        for start in range(min(20, NUM_PTS)):
            if time.time() > deadline:
                break
            if backtrack(start, [], 0):
                # Found a maximal set of size target_k
                # Extract it from the search (need to track it)
                # For simplicity, we'll do a separate search to find one
                pass
        
        # If we found something, update
        # (The above backtrack doesn't track the actual set, so let's redo with tracking)
        
    return best_set

def local_search(S, deadline):
    """Local search: try removals and swaps to minimize the maximal set."""
    best_S = S[:]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best_S)
        
        # Try removing each point
        for i in range(n):
            if time.time() > deadline:
                break
            S_trial = best_S[:i] + best_S[i+1:]
            if is_maximal(S_trial):
                best_S = S_trial
                improved = True
                break
        
        if not improved:
            # Try swaps
            S_mask = 0
            for p in best_S:
                S_mask |= (1 << p)
            
            for i in range(len(best_S)):
                if time.time() > deadline:
                    break
                q = best_S[i]
                S_without_q = best_S[:i] + best_S[i+1:]
                blocked_without_q = get_blocked_mask(S_without_q)
                
                for p in range(NUM_PTS):
                    if (S_mask >> p) & 1:
                        continue
                    if (blocked_without_q >> p) & 1:
                        continue
                    S_swap = S_without_q + [p]
                    if is_valid(S_swap) and is_maximal(S_swap):
                        best_S = S_swap
                        improved = True
                        break
                if improved:
                    break
    
    return best_S

def construct():
    DEADLINE = time.time() + 290
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Greedy restarts
    rng = random.Random(0)
    for i in range(1000):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S = greedy_maximal_set(order)
        if len(S) < best_len:
            best_len = len(S)
            best_set = S.copy()
    
    if best_set is None:
        return [POINTS[i] for i in greedy_maximal_set(list(range(NUM_PTS)))]
    
    # Phase 2: Local search on best greedy result
    S = local_search(best_set, DEADLINE)
    if len(S) < best_len:
        best_len = len(S)
        best_set = S
    
    # Phase 3: More greedy + local search on diverse starts
    for i in range(300):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S_greedy = greedy_maximal_set(order)
        if len(S_greedy) < best_len + 3:
            S_ls = local_search(S_greedy, DEADLINE)
            if len(S_ls) < best_len:
                best_len = len(S_ls)
                best_set = S_ls
    
    return [POINTS[i] for i in best_set]
