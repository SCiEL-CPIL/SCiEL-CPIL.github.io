import time
import random
import numpy as np

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles triples: for each point p, which pairs (a,b) form isosceles with p
# A triple (p, a, b) is isosceles if d(p,a)==d(p,b) or d(p,a)==d(a,b) or d(p,b)==d(a,b)
# For maximality: point p is blocked by set S if there exist a,b in S such that (p,a,b) is isosceles
# For no-isosceles: set S is valid if for all a,b,c in S, (a,b,c) is NOT isosceles

# Precompute: for each point p, the set of points that are "isosceles-partners" with p
# i.e., for point p, blocked_by[p] = set of points q such that there exists r where (p,q,r) isosceles
# Actually, we need: p is blocked by S iff there exist a,b in S with (p,a,b) isosceles
# This means: for each p, precompute which pairs (a,b) make (p,a,b) isosceles
# Then p is blocked by S iff S contains at least one such pair

# More efficient: for each p, compute the set of "forbidden pairs"
# p is blocked if S intersects the forbidden-pair set in at least 2 points that form a forbidden pair
# Actually: p is blocked iff there exist a,b in S such that (p,a,b) is isosceles
# So for each p, we need: does S contain any pair from the isosceles-pair-set of p?

# Let's precompute for each p: iso_pairs[p] = list of (a,b) such that (p,a,b) is isosceles and a<b
# Then p is blocked by S iff there exists (a,b) in iso_pairs[p] with a in S and b in S

# For the no-isosceles constraint: S is valid iff for all triples (a,b,c) in S, it's not isosceles
# Equivalently: for all a,b in S, the set of c such that (a,b,c) is isosceles doesn't intersect S

# Precompute: for each pair (a,b), which points c make (a,b,c) isosceles
# iso_with_pair[a][b] = set of c such that (a,b,c) is isosceles
# Then S is no-isosceles iff for all a,b in S, S ∩ iso_with_pair[a][b] = empty (excluding a,b themselves)

# Let's use bitsets (Python ints) for speed
# For each point p, iso_mask[p] = bitmask of points q such that there exists r with (p,q,r) isosceles
# Wait, that's not quite right. Let me think again.

# For maximality: p is blocked by S iff ∃ a,b ∈ S: (p,a,b) is isosceles
# For each p, define blocked_mask[p] = bitmask of all points such that... 
# Actually: p is blocked by S iff (S as bitmask) contains at least one pair from iso_pairs[p]
# This is hard to check with a single bitmask.

# Alternative: for each p, precompute the set of points that "conflict" with p
# p conflicts with q if there exists r such that (p,q,r) is isosceles
# Then p is blocked by S iff S contains at least 2 points from {p} ∪ conflicts... 
# No, that's not right either.

# Let me just precompute: for each p, the set of pairs (a,b) where a<b, a≠p, b≠p, and (p,a,b) is isosceles
# Store as a list of (a,b) pairs. Then checking if p is blocked by S:
#   for each (a,b) in iso_pairs[p]: if a in S_set and b in S_set: return True

# For no-isosceles check on set S:
#   for each a,b in S: for each c in S: if (a,b,c) is isosceles: return False
# Precompute: for each pair (a,b), the set of c such that (a,b,c) is isosceles
# Then S is no-isosceles iff for all a,b in S: S ∩ iso_c_with_pair[a][b] = ∅ (excluding a,b)

# Let's precompute iso_c_with_pair as a 2D array of bitmasks
# iso_c_mask[i][j] = bitmask of points c such that (i,j,c) is isosceles
iso_c_mask = np.zeros((N_TOTAL, N_TOTAL), dtype=np.uint64)  # 256 points fits in 256 bits... need more
# 256 points need 256 bits = 4 uint64s. Let's use Python ints instead.
iso_c_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        mask = 0
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                mask |= (1 << k)
        iso_c_mask[i][j] = mask
        iso_c_mask[j][i] = mask

# For each point p, precompute which other points q have the property that
# there exists r such that (p,q,r) is isosceles. This is:
# conflict_mask[p] = union over q of... no.
# p is blocked by S iff ∃ a,b ∈ S: (p,a,b) is isosceles
# = ∃ a,b ∈ S: a ∈ iso_c_mask[p][b] (i.e., p is isosceles with a and b)
# So for each p and each a, the set of b such that (p,a,b) is isosceles is:
#   {b : p ∈ iso_c_mask[a][b]} = {b : iso_c_mask[a][b] has bit p set}
# Let's precompute: for each p, for each a (a≠p), the set of b (b≠p, b≠a) such that (p,a,b) is isosceles
# This is: b such that iso_c_mask[a][b] has bit p set, i.e., b such that p is in iso_c_mask[a][b]
# = b such that iso_c_mask[b][a] has bit p set (symmetric)

# For each p, precompute: blocked_pair_mask[p][a] = bitmask of b such that (p,a,b) is isosceles
# Then p is blocked by S iff ∃ a ∈ S: (S_mask & blocked_pair_mask[p][a]) has at least one bit set (and that bit ≠ a)
# Actually we need b ∈ S too, and b ≠ a.

# Let's precompute for each p: a list of (a, bitmask_of_b) where bitmask_of_b = {b : (p,a,b) isosceles}
# Then p is blocked by S iff ∃ a ∈ S: (S_mask & bitmask_of_b) has a bit set that is not a itself
# (We need b ∈ S and b ≠ a, and b ≠ p which is guaranteed since p ∉ S for the maximality check)

blocked_pairs = [[] for _ in range(N_TOTAL)]
for p in range(N_TOTAL):
    for a in range(N_TOTAL):
        if a == p:
            continue
        # Find all b such that (p,a,b) is isosceles
        mask = 0
        for b in range(N_TOTAL):
            if b == p or b == a:
                continue
            # (p,a,b) is isosceles iff p ∈ iso_c_mask[a][b]
            if iso_c_mask[a][b] & (1 << p):
                mask |= (1 << b)
        if mask:
            blocked_pairs[p].append((a, mask))

def _is_no_isosceles(S_mask, S_list):
    """Check if set S is no-isosceles using precomputed masks."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            # Check if any c in S makes (a,b,c) isosceles
            # c is isosceles with (a,b) iff c ∈ iso_c_mask[a][b]
            if iso_c_mask[a][b] & S_mask:
                # Need to exclude a and b themselves from the mask
                # iso_c_mask[a][b] already excludes a and b (we set them to 0 in precompute? Let's check)
                # Actually in precompute we skipped k==i and k==j, so a and b are not in the mask
                return False
    return True

def _is_blocked(p, S_mask, S_list):
    """Check if point p is blocked by set S (forms isosceles with some pair in S)."""
    if len(S_list) < 2:
        return False
    for a, mask_b in blocked_pairs[p]:
        if S_mask & (1 << a):
            # a is in S, check if any b in mask_b is also in S
            if S_mask & mask_b:
                return True
    return False

def _is_maximal(S_mask, S_list):
    """Check if S is maximal: no point can be added."""
    for p in range(N_TOTAL):
        if S_mask & (1 << p):
            continue
        if not _is_blocked(p, S_mask, S_list):
            return False
    return True

def _can_add(p, S_mask, S_list):
    """Check if point p can be added to S without violating no-isosceles."""
    if len(S_list) < 2:
        return True
    # For each a in S, check if p is isosceles with a and some b in S
    # i.e., ∃ a,b in S: (p,a,b) isosceles
    # This is the same as _is_blocked(p, S_mask, S_list)
    return not _is_blocked(p, S_mask, S_list)

def _greedy_build(deadline_margin=15):
    """
    Greedy construction: start empty, repeatedly add the point that blocks the most
    currently-unblocked points while maintaining no-isosceles property.
    """
    S_list = []
    S_mask = 0
    
    while True:
        # Find all addable points
        addable = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if _can_add(p, S_mask, S_list):
                addable.append(p)
        
        if not addable:
            break
        
        # Check if already maximal
        if _is_maximal(S_mask, S_list):
            break
        
        # Find unblocked points (points not in S that are not yet blocked)
        unblocked = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if not _is_blocked(p, S_mask, S_list):
                unblocked.append(p)
        
        if not unblocked:
            break
        
        # Score each addable point by how many unblocked points it would block
        best_p = None
        best_gain = -1
        for p in addable:
            gain = 0
            for q in unblocked:
                if q == p:
                    gain += 1
                    continue
                # Check if q would be blocked by adding p to S
                # q blocked by S∪{p} iff ∃ a,b in S∪{p}: (q,a,b) isosceles
                # Either (q,a,b) with a,b in S (already checked, q is unblocked so no)
                # Or one of a,b is p: (q,p,a) or (q,a,p) isosceles for some a in S
                blocked = False
                for a in S_list:
                    # Check (q,p,a) isosceles
                    if iso_c_mask[p][a] & (1 << q):
                        blocked = True
                        break
                    if iso_c_mask[q][a] & (1 << p):
                        blocked = True
                        break
                if blocked:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        S_list.append(best_p)
        S_mask |= (1 << best_p)
        
        if time.time() > DEADLINE - deadline_margin:
            return None
    
    return S_list

def _reduce_to_minimal(S_list, deadline_margin=10):
    """Iteratively remove points if remaining set is still maximal and no-isosceles."""
    S = list(S_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i+1:]
            S_mask_test = 0
            for x in S_test:
                S_mask_test |= (1 << x)
            if _is_no_isosceles(S_mask_test, S_test) and _is_maximal(S_mask_test, S_test):
                S = S_test
                changed = True
                break
        if time.time() > DEADLINE - deadline_margin:
            break
    return S

def _local_search(S_list, deadline_margin=5):
    """Try to reduce S by removing points and re-adding different ones."""
    S = list(S_list)
    best_size = len(S)
    
    for _ in range(200):
        if time.time() > DEADLINE - deadline_margin:
            break
        
        improved = False
        n_S = len(S)
        
        for i in range(n_S):
            S_without = S[:i] + S[i+1:]
            S_mask_without = 0
            for x in S_without:
                S_mask_without |= (1 << x)
            
            if not _is_no_isosceles(S_mask_without, S_without):
                continue
            
            # Try to add a different point to make it maximal again
            for p_add in range(N_TOTAL):
                if S_mask_without & (1 << p_add):
                    continue
                if not _can_add(p_add, S_mask_without, S_without):
                    continue
                S_new = S_without + [p_add]
                S_mask_new = S_mask_without | (1 << p_add)
                if not _is_no_isosceles(S_mask_new, S_new):
                    continue
                if not _is_maximal(S_mask_new, S_new):
                    continue
                # Try to reduce
                reduced = _reduce_to_minimal(S_new, deadline_margin=3)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    S = reduced
                    improved = True
                    break
                if time.time() > DEADLINE - deadline_margin:
                    break
            if improved or time.time() > DEADLINE - deadline_margin:
                break
        
        if not improved:
            break
    
    return S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Pure greedy coverage
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Randomize the order of tie-breaking
        result = _greedy_build(deadline_margin=15)
        if result is None:
            continue
        
        reduced = _reduce_to_minimal(result, deadline_margin=10)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 8:
            return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 2: Local search on best
    if best_set is not None and best_size > 8:
        improved_set = _local_search(best_set, deadline_margin=10)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    # Strategy 3: Try different greedy seeds
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Start with a random small core
        core_size = rng.choice([2, 3, 4])
        core = rng.sample(range(N_TOTAL), core_size)
        core_mask = 0
        for x in core:
            core_mask |= (1 << x)
        
        if not _is_no_isosceles(core_mask, core):
            continue
        
        # Greedily extend from core
        S_list = list(core)
        S_mask = core_mask
        
        while True:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            addable = []
            for p in range(N_TOTAL):
                if S_mask & (1 << p):
                    continue
                if _can_add(p, S_mask, S_list):
                    addable.append(p)
            
            if not addable:
                break
            
            if _is_maximal(S_mask, S_list):
                break
            
            # Pick addable point that blocks most
            unblocked = []
            for p in range(N_TOTAL):
                if S_mask & (1 << p):
                    continue
                if not _is_blocked(p, S_mask, S_list):
                    unblocked.append(p)
            
            if not unblocked:
                break
            
            best_p = None
            best_gain = -1
            for p in addable:
                gain = 0
                for q in unblocked:
                    if q == p:
                        gain += 1
                        continue
                    blocked = False
                    for a in S_list:
                        if iso_c_mask[p][a] & (1 << q):
                            blocked = True
                            break
                        if iso_c_mask[q][a] & (1 << p):
                            blocked = True
                            break
                    if blocked:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            
            if best_p is None:
                break
            
            S_list.append(best_p)
            S_mask |= (1 << best_p)
        
        if len(S_list) < best_size:
            reduced = _reduce_to_minimal(S_list, deadline_margin=10)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
        
        if best_size <= 8:
            break
    
    # Strategy 4: Local search on best again
    if best_set is not None and best_size > 8:
        improved_set = _local_search(best_set, deadline_margin=5)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    if best_set is None:
        # Fallback
        chosen = []
        chosen_mask = 0
        for i in range(N_TOTAL):
            if _can_add(i, chosen_mask, chosen):
                chosen.append(i)
                chosen_mask |= (1 << i)
        best_set = _reduce_to_minimal(chosen, deadline_margin=5)
    
    return [_ALL_POINTS[i] for i in best_set]
