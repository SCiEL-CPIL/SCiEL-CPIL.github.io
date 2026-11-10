# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES_TOTAL = 2450
N_PER_LINE = p*p  # 49
N_DIRS = 50 # 49 slopes + vertical

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _precompute_lines():
    """
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    Also returns line_dirs: shape (N_LINES_TOTAL, N_TOTAL) ? No, too big.
    We need for each point x, the lines passing through it.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
    for xa in range(p):
        for xb in range(p):
            for k in range(N_PER_LINE):
                ya = k // p
                yb = k % p
                lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
            line_idx += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    
    return lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # 1. Precompute structure
    lines = _precompute_lines()
    
    # point_lines[c] = list of line indices containing point c
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
            
    # Initialize S with empty set? Or random?
    # Empty set is valid?
    # If S is empty, for any x, L intersect S is empty.
    # If x not in S (all x), we need L intersect S = empty. This is true for all L.
    # So S=empty is valid. |S|=0.
    # We want to maximize S.
    
    # Let's start with a random subset or the seed from previous version?
    # The previous version's seed was good. Let's use a similar seed generation but maybe sparser to allow more growth?
    # Or just start with random points and add greedily?
    
    # Strategy: 
    # 1. Start with a random valid S (e.g., empty).
    # 2. Greedy Add: Add random points if valid.
    # 3. Local Search: Swaps.
    
    in_S = np.zeros(N_TOTAL, dtype=np.int8)
    
    # Initialize bad counts
    # bad[x] = number of directions v such that L_{x,v} intersect S is NOT subset of {x}
    # If S is empty, bad[x] = 0 for all x.
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # line_s_counts[i] = number of points in S on line i
    line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
    
    # Helper to check if adding c is valid
    def can_add(c):
        # c must not be in S
        if in_S[c]: return False
        
        # Check all lines through c
        for l_idx in point_lines[c]:
            cnt = line_s_counts[l_idx]
            line_pts = lines[l_idx, :]
            
            # We need to check if any x in L \ {c} becomes "more bad"
            # and if that makes bad[x] >= 50
            
            # Logic from previous analysis:
            # Adding c increases bad[x] for x in L\{c} if:
            # x in S and cnt == 1 (becomes 2, Good->Bad)
            # x not in S and cnt == 0 (becomes 1, Good->Bad)
            
            if cnt == 0:
                # x not in S, cnt 0 -> 1. Bad increases for x not in S.
                # Check if any such x has bad[x] >= 49
                mask = (line_pts != c) & (in_S[line_pts] == 0)
                if np.any(mask):
                    if np.max(bad[line_pts[mask]]) >= 49:
                        return False
            elif cnt == 1:
                # x in S, cnt 1 -> 2. Bad increases for x in S.
                # Check if any such x has bad[x] >= 49
                mask = (line_pts != c) & (in_S[line_pts] == 1)
                if np.any(mask):
                    if np.max(bad[line_pts[mask]]) >= 49:
                        return False
            # else cnt > 1: No bad increases (Bad stays Bad)
            
        return True

    def add_point(c):
        in_S[c] = 1
        for l_idx in point_lines[c]:
            cnt = line_s_counts[l_idx]
            line_s_counts[l_idx] += 1
            line_pts = lines[l_idx, :]
            
            if cnt == 0:
                mask = (line_pts != c) & (in_S[line_pts] == 0)
                bad[line_pts[mask]] += 1
            elif cnt == 1:
                mask = (line_pts != c) & (in_S[line_pts] == 1)
                bad[line_pts[mask]] += 1

    def remove_point(c):
        in_S[c] = 0
        for l_idx in point_lines[c]:
            cnt = line_s_counts[l_idx] # Current count including c
            line_s_counts[l_idx] -= 1
            line_pts = lines[l_idx, :]
            
            # Removing c decreases bad[x] for x in L\{c} if:
            # x in S and cnt == 2 (becomes 1, Bad->Good)
            # x not in S and cnt == 1 (becomes 0, Bad->Good)
            
            if cnt == 2:
                # x in S, cnt 2 -> 1. Bad decreases.
                mask = (line_pts != c) & (in_S[line_pts] == 1)
                bad[line_pts[mask]] -= 1
            elif cnt == 1:
                # x not in S, cnt 1 -> 0. Bad decreases.
                mask = (line_pts != c) & (in_S[line_pts] == 0)
                bad[line_pts[mask]] -= 1

    # Phase 1: Greedy Add
    # Random order
    np.random.seed(42)
    order = np.random.permutation(N_TOTAL)
    
    for c in order:
        if time.time() > _DEADLINE - 30:
            break
        if can_add(c):
            add_point(c)
            
    # Phase 2: Local Search (Swaps)
    # Try to find improvements.
    # A swap (remove u, add v) is valid if:
    # 1. State after remove u is valid (u itself might become invalid? No, we check global validity).
    #    Actually, if we remove u, u is no longer in S.
    #    Condition for u (now not in S): Needs a line L with L intersect S = empty.
    #    Condition for v (now in S): Needs a line L with L intersect S = {v}.
    #    Condition for others: Must remain valid.
    
    # Instead of checking full validity, we can use the bad counts.
    # But swap involves two changes.
    # Let's just try random swaps and check validity by recalculating bad?
    # Recalculating bad is O(N_LINES * N_PER_LINE) ~ 100k ops.
    # 600s / 0.1ms = 600k swaps? Maybe.
    
    # Better: Check validity incrementally?
    # If we remove u and add v:
    # 1. Remove u. Update bad.
    # 2. Check if u is valid (bad[u] < 50).
    # 3. Check if v is valid (can_add(v) logic, but v is not in S yet).
    #    Wait, can_add(v) assumes u is already removed.
    # 4. If both valid, and all others valid (bad[x] < 50 for all x), accept.
    #    We need to ensure no other x became invalid.
    #    Removing u can only DECREASE bad[x].
    #    Adding v can only INCREASE bad[x].
    #    So if we check that adding v doesn't make any x invalid (given u is removed), and u is valid,
    #    and we know removing u didn't make anyone invalid (it only helped), then it's valid.
    
    def try_swap(u, v):
        # u in S, v not in S
        if in_S[u] == 0 or in_S[v] == 1:
            return False
            
        # 1. Remove u
        remove_point(u)
        
        # Check if u is valid (bad[u] < 50)
        if bad[u] >= 50:
            # Restore
            add_point(u)
            return False
            
        # 2. Try to add v
        if can_add(v):
            add_point(v)
            # Check if v is valid? can_add ensures others are ok.
            # Does can_add check v itself?
            # In can_add(c), we check lines through c.
            # For x=c, we established bad[c] doesn't change in a way that breaks validity?
            # Wait.
            # If c is added, and it was "good" (bad[c] < 50), does it remain good?
            # bad[c] counts bad directions for c.
            # When c is not in S, good dir means C=0.
            # When c is in S, good dir means C=1.
            # If C=0 for all dirs? Then bad[c]=50. Invalid.
            # So if c was valid (had at least one good dir, i.e., C=0),
            # after adding, that dir has C=1, so it is still good.
            # So bad[c] does not increase?
            # Let's re-verify.
            # Dir v is good for c (not in S) if C_v = 0.
            # Dir v is good for c (in S) if C_v = 1.
            # If C_v = 0, it was good. After add, C_v = 1, it is good.
            # So if c had a good dir, it still has it.
            # So bad[c] does not increase.
            # Thus, if c was valid before add, it is valid after add.
            
            return True
        else:
            # Restore
            add_point(u)
            return False

    # Local Search Loop
    # Keep track of best |S|
    best_size = np.sum(in_S)
    best_S = in_S.copy()
    
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < _DEADLINE - 10:
        if no_improve_count > max_no_improve:
            # Restart? Or just break?
            # Let's try to restart with a random perturbation of best_S
            in_S = best_S.copy()
            # Recalculate bad and line_s_counts
            bad = np.zeros(N_TOTAL, dtype=np.int32)
            line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
            for c in range(N_TOTAL):
                if in_S[c]:
                    add_point(c)
            current_size = np.sum(in_S)
            no_improve_count = 0
            max_no_improve += 500 # Increase patience?
            
        # Pick random u in S, v not in S
        # To speed up, we can pick u from S and v from not S
        # But random selection is fine for now
        
        # Use numpy random choice for speed?
        # in_S is int8 array.
        # indices_S = np.where(in_S==1)[0]
        # indices_notS = np.where(in_S==0)[0]
        
        # This is expensive.
        # Let's maintain lists?
        # Or just try random indices.
        
        u = np.random.randint(0, N_TOTAL)
        v = np.random.randint(0, N_TOTAL)
        
        if in_S[u] and not in_S[v]:
            if try_swap(u, v):
                current_size += 0 # Size stays same?
                # Wait, swap keeps size same.
                # We want to INCREASE size.
                # So we should prefer Add over Swap.
                # But if we can't Add, Swap might unlock future Adds.
                # Let's count swaps as "progress" only if they lead to an Add later?
                # No, just accept swaps.
                no_improve_count = 0 # Reset?
                # Actually, swap doesn't increase size.
                # We should only reset no_improve if size increases.
                # But if we never increase size, we loop forever.
                # So we must allow swaps to happen, but if we don't get an increase, we restart.
                
                # However, if we accept a swap, the state changes.
                # We should NOT reset no_improve_count on swap alone.
                # We only reset on size increase.
                
        # Try Add
        # We can try to add a random point.
        # If we can add, size increases.
        
        # Let's alternate: 50% Add, 50% Swap
        if np.random.rand() < 0.5:
             # Try Add
             v = np.random.randint(0, N_TOTAL)
             if not in_S[v] and can_add(v):
                 add_point(v)
                 current_size += 1
                 if current_size > best_size:
                     best_size = current_size
                     best_S = in_S.copy()
                 no_improve_count = 0
             else:
                 no_improve_count += 1
        else:
             # Try Swap
             u = np.random.randint(0, N_TOTAL)
             v = np.random.randint(0, N_TOTAL)
             if in_S[u] and not in_S[v]:
                 if try_swap(u, v):
                     # State changed, but size same.
                     # Don't reset no_improve_count?
                     # If we don't, we will restart eventually.
                     # This is good.
                     pass
                 else:
                     no_improve_count += 1
             else:
                 no_improve_count += 1

    # Restore best_S
    in_S = best_S
    
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
