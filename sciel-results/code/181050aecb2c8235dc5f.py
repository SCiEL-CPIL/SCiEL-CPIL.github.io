"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE_MARGIN = 5.0

def _time_remaining():
    return 330.0 - (time.time() - _start_time) - _DEADLINE_MARGIN

def _precompute_conflicts():
    """
    Precompute conflicts[A][B] = set of points C such that A, B, C form an isosceles triangle.
    Includes degenerate cases (collinear with one point in middle, or all same).
    """
    all_points = [(x, y) for x in range(N) for y in range(N)]
    conflicts = {}
    for A in all_points:
        conflicts[A] = {}
        for B in all_points:
            if A == B:
                continue
            d_ab_sq = (A[0] - B[0])**2 + (A[1] - B[1])**2
            block_set = set()
            for C in all_points:
                if C == A or C == B:
                    continue
                d_ac_sq = (A[0] - C[0])**2 + (A[1] - C[1])**2
                d_bc_sq = (B[0] - C[0])**2 + (B[1] - C[1])**2
                if d_ab_sq == d_ac_sq or d_ab_sq == d_bc_sq or d_ac_sq == d_bc_sq:
                    block_set.add(C)
            conflicts[A][B] = block_set
    return conflicts

# Precompute once
CONFLICTS = _precompute_conflicts()
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

def _greedy_set_cover(seed=0):
    """
    Construct a maximal no-isosceles set using a greedy set cover heuristic.
    Goal: Minimize the number of points in 'chosen' such that all other points are 'blocked'.
    """
    random.seed(seed)
    chosen = []
    chosen_set = set()
    blocked = set()
    available = set(ALL_POINTS)
    
    while available:
        if _time_remaining() <= 0:
            break
        
        best_point = None
        best_gain = -1
        best_new_blocked = None
        
        # To add variety, we can shuffle available or just pick best with random tie-break
        # For efficiency, we iterate through available.
        # N=16, so |available| <= 256.
        # For each candidate, we compute new blocked.
        
        candidates = list(available)
        random.shuffle(candidates)
        
        for p in candidates:
            # Calculate new blocked points if we add p
            # New blocked = existing blocked U { q | exists a in chosen such that q in conflicts[a][p] }
            new_blocked_add = set()
            for a in chosen:
                new_blocked_add.update(CONFLICTS[a][p])
            
            # The points that become newly blocked are those in new_blocked_add that are currently in available
            # Note: p itself is not blocked by this, but p is removed from available because it's chosen.
            # Points in new_blocked_add that are in available will be removed from available.
            
            # Gain = number of points in available that will be removed (blocked)
            # Note: p is in available, but p is chosen, not blocked.
            # So we count | (new_blocked_add ∩ available) |
            # But we also need to ensure that adding p doesn't create an isosceles triangle with two points in chosen?
            # Wait, the definition of 'blocked' is: q is blocked if it forms an isosceles triangle with ANY pair in chosen.
            # If we add p, we must ensure p does NOT form an isosceles triangle with any pair in chosen.
            # This is the validity constraint.
            
            # Check validity: p must not form iso with any pair in chosen
            valid = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    # Check if p, chosen[i], chosen[j] is iso
                    # We can use precomputed conflicts?
                    # conflicts[chosen[i]][chosen[j]] contains p if it is iso.
                    if p in CONFLICTS[chosen[i]][chosen[j]]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Calculate gain
            # Points in available that will be blocked by adding p
            # These are points q in available such that q is in conflicts[a][p] for some a in chosen.
            # OR q is in conflicts[p][a]? Symmetric.
            # We already computed new_blocked_add = union of conflicts[a][p] for a in chosen.
            # So gain = len(new_blocked_add & available)
            
            gain = len(new_blocked_add & available)
            
            # Tie-breaking: random or by index
            if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                best_gain = gain
                best_point = p
                best_new_blocked = new_blocked_add
        
        if best_point is None:
            # No valid point can be added? This shouldn't happen if we start with empty set.
            # If chosen is empty, any point is valid.
            # If chosen is not empty, maybe all available points are invalid?
            # Then the current set is maximal? No, if available is not empty, and no point can be added,
            # it means every point in available forms an iso triangle with some pair in chosen.
            # That is exactly the definition of being blocked!
            # So if best_point is None, it means all points in available are already blocked?
            # But we maintain 'available' as points NOT in chosen and NOT in blocked.
            # So if best_point is None, it implies a bug or that available is empty.
            # Actually, if a point is in available, it is NOT blocked.
            # If it is not blocked, it does NOT form an iso with any pair in chosen.
            # So it IS valid to add.
            # So best_point should never be None if available is not empty.
            break
        
        # Commit the choice
        chosen.append(best_point)
        chosen_set.add(best_point)
        blocked.update(best_new_blocked)
        available.discard(best_point)
        available -= best_new_blocked
    
    return chosen

def _try_remove_points(chosen):
    """
    Try to remove points from chosen while maintaining maximality.
    If we remove p, we must check if the set is still maximal.
    A set is maximal if no point in ALL_POINTS \ chosen can be added.
    """
    chosen = list(chosen)
    
    # We need to efficiently check maximality.
    # Maximal means: For every q in ALL_POINTS \ chosen, q forms an iso triangle with some pair in chosen.
    
    for idx in range(len(chosen) - 1, -1, -1):
        if _time_remaining() <= 0:
            break
            
        p = chosen[idx]
        candidate_chosen = chosen[:idx] + chosen[idx+1:]
        
        # Check if candidate_chosen is still maximal
        # This requires checking all q in ALL_POINTS \ candidate_chosen
        # If any q can be added (i.e., does not form iso with any pair in candidate_chosen), then it's not maximal.
        
        is_maximal = True
        candidate_set = set(candidate_chosen)
        
        for q in ALL_POINTS:
            if q in candidate_set:
                continue
            
            # Check if q can be added to candidate_chosen
            # i.e., q does not form iso with any pair in candidate_chosen
            can_add = True
            for i in range(len(candidate_chosen)):
                for j in range(i+1, len(candidate_chosen)):
                    if q in CONFLICTS[candidate_chosen[i]][candidate_chosen[j]]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                is_maximal = False
                break
        
        if is_maximal:
            chosen = candidate_chosen
            
    return chosen

def construct():
    best = None
    
    # Try multiple seeds for the greedy step
    for seed in range(50):
        if _time_remaining() <= 0:
            break
        
        cand = _greedy_set_cover(seed)
        cand = _try_remove_points(cand)
        
        if best is None or len(cand) < len(best):
            best = cand
            
    return best
# EVOLVE-BLOCK-END
