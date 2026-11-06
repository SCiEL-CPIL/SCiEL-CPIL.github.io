import time
import random
import numpy as np

N = 16
DEADLINE = 320.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles relation masks
# iso_mask[i][j] = bitmask of points k such that (i,j,k) forms an isosceles triangle
# (excluding i and j themselves)
iso_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

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
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

# Precompute for each point p: the set of points that conflict with p
# p conflicts with q if there exists r such that (p,q,r) is isosceles
# conflict_mask[p] = bitmask of all q such that p and q "conflict" (can't both be in a no-isosceles set with some third point)
# Actually, for a no-isosceles set S: for all a,b in S, S ∩ iso_mask[a][b] = ∅
# This means: if a,b are in S, then no c in S can be in iso_mask[a][b]
# So the constraint is: for all pairs (a,b) in S, the iso_mask[a][b] ∩ S = ∅ (excluding a,b)

# For maximality: p is blocked by S iff ∃ a,b ∈ S: p ∈ iso_mask[a][b]
# i.e., ∃ a,b ∈ S: (a,b,p) is isosceles

# Let's precompute: for each p, the set of pairs (a,b) in S that would block p
# blocked_by_mask[p] = for each a, the set of b such that p ∈ iso_mask[a][b]
# Then p is blocked by S iff ∃ a ∈ S: (S_mask & blocked_by_mask[p][a]) has a bit set (b ≠ a, b ≠ p)

# Precompute: for each p, for each a ≠ p: 
#   pair_mask[p][a] = bitmask of b (b≠p, b≠a) such that p ∈ iso_mask[a][b]
pair_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

for p in range(N_TOTAL):
    for a in range(N_TOTAL):
        if a == p:
            continue
        mask = 0
        for b in range(N_TOTAL):
            if b == p or b == a:
                continue
            if iso_mask[a][b] & (1 << p):
                mask |= (1 << b)
        pair_mask[p][a] = mask

def _is_no_isosceles(S_mask, S_list):
    """Check if set S is no-isosceles."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_mask[a][b] & S_mask:
                return False
    return True

def _is_blocked(p, S_mask, S_list):
    """Check if point p is blocked by set S."""
    if len(S_list) < 2:
        return False
    for a in S_list:
        if S_mask & (1 << a):
            if S_mask & pair_mask[p][a]:
                return True
    return False

def _is_maximal(S_mask, S_list):
    """Check if S is maximal: no point can be added without violating no-isosceles."""
    for p in range(N_TOTAL):
        if S_mask & (1 << p):
            continue
        # p can be added iff it doesn't form isosceles with any pair in S
        # i.e., p is NOT blocked by S
        if not _is_blocked(p, S_mask, S_list):
            return False
    return True

def _can_add(p, S_mask, S_list):
    """Check if point p can be added to S while maintaining no-isosceles."""
    return not _is_blocked(p, S_mask, S_list)

def _reduce_to_minimal(S_list, deadline_margin=10):
    """Iteratively remove points if the remaining set is still maximal and no-isosceles."""
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
                blocked = False
                for a in S_list:
                    if iso_mask[p][a] & (1 << q):
                        blocked = True
                        break
                    if iso_mask[q][a] & (1 << p):
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

def _simulated_annealing(S_list, deadline_margin=10):
    """
    Simulated Annealing: start from a valid maximal set, try random swaps (remove one, add another)
    accepting worse solutions with probability exp(-delta/T) to escape local minima.
    """
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    best_S = list(S)
    best_size = len(S)
    
    T = 50.0
    T_min = 0.5
    cooling = 0.98
    
    while T > T_min:
        if time.time() > DEADLINE - deadline_margin:
            break
        
        # Try a random move: remove one point, add another
        if len(S) < 2:
            break
        
        i_remove = random.randint(0, len(S) - 1)
        p_remove = S[i_remove]
        
        S_without = S[:i_remove] + S[i_remove+1:]
        S_mask_without = S_mask & ~(1 << p_remove)
        
        # Try to find a point to add
        addable = []
        for p_add in range(N_TOTAL):
            if S_mask_without & (1 << p_add):
                continue
            if _can_add(p_add, S_mask_without, S_without):
                addable.append(p_add)
        
        if not addable:
            T *= cooling
            continue
        
        p_add = random.choice(addable)
        S_new = S_without + [p_add]
        S_mask_new = S_mask_without | (1 << p_add)
        
        # Check validity
        if not _is_no_isosceles(S_mask_new, S_new):
            T *= cooling
            continue
        
        # Check maximality
        if not _is_maximal(S_mask_new, S_new):
            # Try to repair: add more points to make it maximal
            S_repaired = list(S_new)
            S_mask_repaired = S_mask_new
            repaired = True
            while repaired:
                repaired = False
                for p_try in range(N_TOTAL):
                    if S_mask_repaired & (1 << p_try):
                        continue
                    if _can_add(p_try, S_mask_repaired, S_repaired):
                        if not _is_blocked(p_try, S_mask_repaired, S_repaired):
                            S_repaired.append(p_try)
                            S_mask_repaired |= (1 << p_try)
                            repaired = True
                            break
                if time.time() > DEADLINE - deadline_margin:
                    break
            
            if not _is_maximal(S_mask_repaired, S_repaired):
                T *= cooling
                continue
            
            S_new = S_repaired
            S_mask_new = S_mask_repaired
        
        # Accept or reject
        delta = len(S_new) - len(S)
        if delta <= 0 or random.random() < np.exp(-delta / T):
            S = S_new
            S_mask = S_mask_new
            if len(S) < best_size:
                best_size = len(S)
                best_S = list(S)
        
        T *= cooling
    
    return best_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Pure greedy coverage
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 60:
            break
        
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
    
    # Strategy 3: Simulated Annealing
    if best_set is not None and best_size > 8:
        sa_set = _simulated_annealing(best_set, deadline_margin=10)
        if len(sa_set) < best_size:
            best_size = len(sa_set)
            best_set = sa_set[:]
    
    # Strategy 4: Try different greedy seeds
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        core_size = rng.choice([2, 3, 4])
        core = rng.sample(range(N_TOTAL), core_size)
        core_mask = 0
        for x in core:
            core_mask |= (1 << x)
        
        if not _is_no_isosceles(core_mask, core):
            continue
        
        S_list = list(core)
        S_mask = core_mask
        
        while True:
            if time.time() - start_time > DEADLINE - 20:
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
                        if iso_mask[p][a] & (1 << q):
                            blocked = True
                            break
                        if iso_mask[q][a] & (1 << p):
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
    
    # Strategy 5: Final local search
    if best_set is not None and best_size > 8:
        improved_set = _local_search(best_set, deadline_margin=5)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    if best_set is None:
        chosen = []
        chosen_mask = 0
        for i in range(N_TOTAL):
            if _can_add(i, chosen_mask, chosen):
                chosen.append(i)
                chosen_mask |= (1 << i)
        best_set = _reduce_to_minimal(chosen, deadline_margin=5)
    
    return [_ALL_POINTS[i] for i in best_set]
