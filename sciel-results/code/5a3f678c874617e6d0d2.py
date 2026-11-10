import time
import random
import numpy as np

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

# Precompute Iso masks
# Iso[i][j] is a bitmask of k such that triangle i,j,k is isosceles
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

# Compute weight for each point: distance from center (higher = more edge-like)
center = (N - 1) / 2.0
weights = [0.0] * NUM_PTS
for i in range(NUM_PTS):
    x, y = POINTS[i]
    weights[i] = abs(x - center) + abs(y - center)

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

def weighted_greedy_construct(order, deadline):
    """Build a maximal no-isosceles set by greedily adding points from weighted order."""
    S = []
    for p in order:
        if time.time() > deadline:
            break
        can_add = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if (Iso[S[i]][S[j]] >> p) & 1:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.append(p)
            
    # Complete to maximal if not already maximal
    if not is_maximal(S):
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
        blocked = get_blocked_mask(S)
        unblocked = FULL_MASK & ~S_mask & ~blocked
        # Add points from unblocked in weight order
        candidates = [p for p in range(NUM_PTS) if (unblocked >> p) & 1]
        candidates.sort(key=lambda p: -weights[p])
        for p in candidates:
            if time.time() > deadline:
                break
            can_add = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if (Iso[S[i]][S[j]] >> p) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.append(p)
    
    return S

def local_search_reduce(S, deadline):
    """
    Reduce the size of a maximal no-isosceles set S using:
    1. Removal phase: try removing each point if maximality is preserved
    2. Targeted swap phase: identify critical points and swap them with better alternatives
    3. Repeat until no improvement or deadline
    """
    S = S[:]
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        n = len(S)
        
        # Phase 1: Simple removal - try each point
        for i in range(n):
            if time.time() > deadline:
                return S
            S_trial = S[:i] + S[i+1:]
            if is_maximal(S_trial):
                S = S_trial
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Targeted swap - find points that are "critical" for maximality
        # A point q is critical if removing it makes the set non-maximal
        # Then we look for a point p not in S that can replace q while maintaining maximality
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
        blocked_full = get_blocked_mask(S)
        
        # Find unblocked points (candidates for addition)
        unblocked_mask = FULL_MASK & ~S_mask & ~blocked_full
        unblocked_pts = [p for p in range(NUM_PTS) if (unblocked_mask >> p) & 1]
        
        if not unblocked_pts:
            # No candidates to swap with
            break
        
        # Sort unblocked points by weight (prefer edge points)
        unblocked_pts.sort(key=lambda p: -weights[p])
        
        # For each point in S, try swapping it with a high-weight unblocked point
        found_swap = False
        for i in range(len(S)):
            if time.time() > deadline:
                return S
            q = S[i]
            S_without_q = S[:i] + S[i+1:]
            
            # Check if removing q breaks maximality
            blocked_wo_q = get_blocked_mask(S_without_q)
            S_mask_wo_q = 0
            for p in S_without_q:
                S_mask_wo_q |= (1 << p)
            
            # Points that were blocked by pairs involving q are now unblocked
            # These are the points that need to be re-blocked
            newly_unblocked = ~blocked_wo_q & FULL_MASK & ~S_mask_wo_q
            newly_unblocked_pts = [p for p in range(NUM_PTS) if (newly_unblocked >> p) & 1]
            
            if not newly_unblocked_pts:
                # Removing q doesn't break maximality (shouldn't happen if Phase 1 failed)
                continue
            
            # Try to find a point p in unblocked_pts that can replace q
            # p must be valid with S_without_q and must block all newly_unblocked points
            for p in unblocked_pts[:20]:  # Limit to top 20 by weight
                if time.time() > deadline:
                    return S
                
                # Check if p is valid with S_without_q
                valid = True
                for j in range(len(S_without_q)):
                    if (Iso[S_without_q[j]][p] >> p) & 1 or \
                       any((Iso[S_without_q[j]][S_without_q[k]] >> p) & 1 
                           for k in range(len(S_without_q)) if k != j):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check if p blocks all newly_unblocked points
                # p with each point in S_without_q creates new blocking
                S_swap = S_without_q + [p]
                if is_maximal(S_swap):
                    S = S_swap
                    found_swap = True
                    improved = True
                    no_improve_count = 0
                    break
            
            if found_swap:
                break
        
        if not found_swap:
            no_improve_count += 1
            if no_improve_count >= 5:
                break
    
    return S

def construct():
    DEADLINE = time.time() + 290
    best_set = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Weighted greedy restarts with edge-biased ordering
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
        
        # Create order biased toward edge points
        order = list(range(NUM_PTS))
        
        # Mix of pure weight order and random perturbation
        if i % 3 == 0:
            # Pure weight order (edge first)
            order.sort(key=lambda p: -weights[p])
        elif i % 3 == 1:
            # Random with weight bias
            rng.shuffle(order)
            # Move high-weight points to front
            order.sort(key=lambda p: (-weights[p] if rng.random() < 0.3 else rng.random()))
        else:
            # Fully random
            rng.shuffle(order)
        
        S = weighted_greedy_construct(order, DEADLINE)
        
        if len(S) < best_len:
            best_len = len(S)
            best_set = S.copy()
            
    if best_set is None:
        return [POINTS[i] for i in weighted_greedy_construct(
            sorted(range(NUM_PTS), key=lambda p: -weights[p]), DEADLINE)]
        
    # Phase 2: Local search on top candidates
    # Collect diverse candidates
    top_candidates = [best_set.copy()]
    
    # Add a few more diverse candidates from different restart strategies
    for i in range(5):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        # Different perturbation strategies
        if i == 0:
            order.sort(key=lambda p: -weights[p])
        elif i == 1:
            rng.shuffle(order)
        elif i == 2:
            order.sort(key=lambda p: weights[p])  # Center first
        elif i == 3:
            order.sort(key=lambda p: (p % N, p // N))  # Row-major
        else:
            order.sort(key=lambda p: ((p % N) + (p // N)))  # Diagonal
        
        S = weighted_greedy_construct(order, DEADLINE)
        if S and len(S) <= best_len + 5:
            top_candidates.append(S.copy())
    
    # Refine each candidate
    for S_cand in top_candidates:
        if time.time() > DEADLINE:
            break
        S_refined = local_search_reduce(S_cand, DEADLINE)
        if len(S_refined) < best_len:
            best_len = len(S_refined)
            best_set = S_refined.copy()
            
    return [POINTS[i] for i in best_set]

# Fallback if construct fails or times out badly (shouldn't happen with guards)
if __name__ == "__main__":
    res = construct()
    print(len(res))
