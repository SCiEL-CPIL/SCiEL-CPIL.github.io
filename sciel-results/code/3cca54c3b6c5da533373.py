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

def greedy_construct(order):
    """Build a maximal no-isosceles set by greedily adding points from order."""
    S = []
    for p in order:
        can_add = True
        # Check if p is blocked by any existing pair in S
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
    Try to reduce the size of a maximal no-isosceles set S.
    Uses a two-phase approach:
    1. Try to remove points one by one if the set remains maximal.
    2. If stuck, try swap moves (remove one, add one) to find a smaller maximal set.
    """
    S = S[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(S)
        
        # Phase 1: Simple removal
        for i in range(n):
            if time.time() > deadline:
                return S
            S_trial = S[:i] + S[i+1:]
            if is_maximal(S_trial):
                S = S_trial
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Swap moves
        # We want to find a point q in S and a point p not in S such that
        # S - {q} + {p} is a valid maximal set.
        # To be efficient, we don't check all pairs. We sample.
        
        # Randomly select a point to remove
        i = random.randint(0, len(S) - 1)
        q = S[i]
        S_without_q = S[:i] + S[i+1:]
        
        # Compute blocked mask for S_without_q
        # Note: S_without_q might not be maximal or valid, but we are checking validity of the new set
        # Actually, S_without_q is a subset of a valid set, so it is valid.
        # But it might not be maximal.
        
        # Candidates for addition:
        # Must be valid with S_without_q
        # And must help block points that S_without_q fails to block
        # Or simply, the new set must be maximal.
        
        # Let's identify "unblocked" points in S_without_q
        # blocked_mask_wo_q = get_blocked_mask(S_without_q)
        # unblocked = FULL_MASK & ~blocked_mask_wo_q & ~S_mask_wo_q
        # Any point in S_without_q that was only blocking 'q' or points blocked by 'q' might now be unblocked?
        # No, removal of q only *decreases* the blocked set.
        
        # A point p can be added to S_without_q if:
        # 1. p is not in S_without_q
        # 2. p is not blocked by any pair in S_without_q (i.e., p is not in get_blocked_mask(S_without_q))
        
        blocked_wo_q = get_blocked_mask(S_without_q)
        S_mask_wo_q = 0
        for p in S_without_q:
            S_mask_wo_q |= (1 << p)
            
        # Potential candidates are points not in S_without_q and not blocked by S_without_q
        # candidates_mask = FULL_MASK & ~S_mask_wo_q & ~blocked_wo_q
        candidates = []
        # Iterate through bits of candidates_mask
        # For efficiency in Python, we can just iterate range(NUM_PTS)
        # But NUM_PTS is 256, so it's fine.
        
        # To find smaller sets, we prefer adding points that block many things.
        # Let's just pick a few random candidates from the valid ones.
        
        valid_candidates = []
        for p in range(NUM_PTS):
            if not (S_mask_wo_q >> p) & 1 and not (blocked_wo_q >> p) & 1:
                valid_candidates.append(p)
                
        if not valid_candidates:
            continue
            
        # Shuffle to randomize
        random.shuffle(valid_candidates)
        
        # Try a few swaps
        found = False
        for p in valid_candidates[:10]:
            if time.time() > deadline:
                return S
            S_swap = S_without_q + [p]
            if is_maximal(S_swap):
                # Check if it's smaller? 
                # S_swap has same size as S.
                # We only want to accept if it allows further reductions?
                # Or if it's the same size, we might be stuck in a local minimum.
                # However, accepting same-size moves can help escape local minima.
                # But we want to minimize size. 
                # If we accept a same-size move, we might get stuck cycling.
                # Let's only accept if it leads to a reduction in the next step?
                # That's expensive.
                # Alternative: Accept same-size moves with a probability or if it changes the structure.
                # For now, let's just accept it to explore, but mark that we didn't reduce size.
                S = S_swap
                improved = True
                found = True
                break
                
        if not found:
             # If no swap found, we might be stuck.
             # Try a different removal point?
             pass
             
    return S

def construct():
    DEADLINE = time.time() + 290
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Greedy restarts to get good initial solutions
    rng = random.Random(42)
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S = greedy_construct(order)
        
        if len(S) < best_len:
            best_len = len(S)
            best_set = S.copy()
            
    if best_set is None:
        return [POINTS[i] for i in greedy_construct(list(range(NUM_PTS)))]
        
    # Phase 2: Local Search on the best set found
    # We might want to run local search on top K best sets, not just the very best,
    # in case the best greedy set is a bad local minimum.
    
    # Let's store top 5 sets
    top_sets = []
    # Re-run a few to get a diverse top list? Or just use the one found?
    # To save time, let's just use the best one found in Phase 1.
    # But maybe the 2nd best greedy set reduces better.
    # Let's collect a few during Phase 1.
    
    # Actually, let's re-structure slightly:
    # Collect candidates in Phase 1.
    
    # Reset for a cleaner top-k collection
    top_candidates = []
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S = greedy_construct(order)
        
        # Keep top 5 unique sets (by sorted tuple)
        key = tuple(sorted(S))
        if key not in [tuple(sorted(s)) for s in top_candidates]:
            top_candidates.append(S)
            if len(top_candidates) > 5:
                # Remove the largest
                top_candidates.sort(key=lambda x: len(x))
                top_candidates.pop()
                
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
