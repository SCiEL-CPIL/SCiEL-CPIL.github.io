import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles triples as bitmasks
# iso_c_mask[i][j] = bitmask of points c such that (i,j,c) is isosceles
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

# For each point p, precompute the set of pairs (a,b) that would make p blocked
# blocked_pairs[p] = list of (a, mask_b) where mask_b = {b : (p,a,b) is isosceles}
blocked_pairs = [[] for _ in range(N_TOTAL)]
for p in range(N_TOTAL):
    for a in range(N_TOTAL):
        if a == p:
            continue
        mask = 0
        for b in range(N_TOTAL):
            if b == p or b == a:
                continue
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
            if iso_c_mask[a][b] & S_mask:
                return False
    return True


def _is_blocked(p, S_mask, S_list):
    """Check if point p is blocked by set S (forms isosceles with some pair in S)."""
    if len(S_list) < 2:
        return False
    for a, mask_b in blocked_pairs[p]:
        if S_mask & (1 << a):
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
    return not _is_blocked(p, S_mask, S_list)


def _count_violations(S_mask, S_list):
    """Count the number of isosceles triples in S."""
    n = len(S_list)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            count += bin(iso_c_mask[a][b] & S_mask).count('1')
    return count


def _walk_sat(S_list, max_steps=5000, rng=None):
    """
    Walk-SAT style local search:
    - Start with a set S
    - Objective: minimize |S| subject to no-isosceles and maximality
    - Flip variables (add/remove points) to reduce violations and size
    """
    if rng is None:
        rng = random.Random()
    
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    best_S = S[:]
    best_size = len(S)
    best_violations = _count_violations(S_mask, S)
    
    for step in range(max_steps):
        if not _is_no_isosceles(S_mask, S):
            # Try to remove a point that is part of many violations
            # Find the point in S that participates in the most isosceles triples
            point_violation_count = {p: 0 for p in S}
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a = S[i]
                    b = S[j]
                    violations = iso_c_mask[a][b] & S_mask
                    # For each c in violations, both a and b are part of the triple
                    # Actually, the triple is (a,b,c), so a and b are the "base"
                    # But c is also part of it
                    # Count for a, b, and each c
                    point_violation_count[a] += 1
                    point_violation_count[b] += 1
                    # Count each c
                    v = violations
                    while v:
                        lsb = v & (-v)
                        c = lsb.bit_length() - 1
                        if c in point_violation_count:
                            point_violation_count[c] += 1
                        v ^= lsb
            
            # Remove the point with highest violation count
            if point_violation_count:
                worst_point = max(point_violation_count, key=lambda p: point_violation_count[p])
                S.remove(worst_point)
                S_mask &= ~(1 << worst_point)
            continue
        
        # S is no-isosceles, now try to reduce size while maintaining maximality
        if not _is_maximal(S_mask, S):
            # Need to add points to make it maximal
            # Find unblocked points
            unblocked = []
            for p in range(N_TOTAL):
                if S_mask & (1 << p):
                    continue
                if not _is_blocked(p, S_mask, S):
                    unblocked.append(p)
            
            if not unblocked:
                break
            
            # Add a random unblocked point that can be added
            addable = [p for p in unblocked if _can_add(p, S_mask, S)]
            if not addable:
                # Can't add any, try removing a point to free up space
                if len(S) > 1:
                    p_remove = rng.choice(S)
                    S.remove(p_remove)
                    S_mask &= ~(1 << p_remove)
                continue
            
            p_add = rng.choice(addable)
            S.append(p_add)
            S_mask |= (1 << p_add)
            continue
        
        # S is no-isosceles and maximal, try to reduce size
        if len(S) <= best_size:
            # Try to remove a random point
            if len(S) > 1:
                p_remove = rng.choice(S)
                S_test = S[:]
                S_test.remove(p_remove)
                S_mask_test = S_mask & ~(1 << p_remove)
                
                if _is_no_isosceles(S_mask_test, S_test) and _is_maximal(S_mask_test, S_test):
                    S = S_test
                    S_mask = S_mask_test
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = S[:]
                        best_violations = 0
            continue
        
        # S is larger than best, try to shrink
        # Remove a random point
        if len(S) > 1:
            p_remove = rng.choice(S)
            S.remove(p_remove)
            S_mask &= ~(1 << p_remove)
    
    return best_S


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


def _local_search(S_list, deadline_margin=5, max_iterations=500):
    """
    Local search: try 1-swap moves (remove one, add one) to reduce size.
    """
    S = list(S_list)
    best_size = len(S)
    
    for _ in range(max_iterations):
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


def _simulated_annealing(S_list, deadline_margin=10, max_iterations=2000):
    """
    Simulated annealing: accept worse solutions with decreasing probability.
    """
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    best_S = S[:]
    best_size = len(S)
    
    temperature = 5.0
    cooling = 0.999
    
    for _ in range(max_iterations):
        if time.time() > DEADLINE - deadline_margin:
            break
        
        # Generate a neighbor: remove one point and optionally add one
        if len(S) > 1:
            p_remove = random.choice(S)
            S_new = S[:]
            S_new.remove(p_remove)
            S_mask_new = S_mask & ~(1 << p_remove)
            
            # Optionally add a random point
            if random.random() < 0.5:
                candidates = [p for p in range(N_TOTAL) if not (S_mask_new & (1 << p))]
                if candidates:
                    p_add = random.choice(candidates)
                    if _can_add(p_add, S_mask_new, S_new):
                        S_new.append(p_add)
                        S_mask_new |= (1 << p_add)
            
            if not _is_no_isosceles(S_mask_new, S_new):
                temperature *= cooling
                continue
            
            if not _is_maximal(S_mask_new, S_new):
                temperature *= cooling
                continue
            
            new_size = len(S_new)
            if new_size <= best_size:
                S = S_new
                S_mask = S_mask_new
                if new_size < best_size:
                    best_size = new_size
                    best_S = S_new[:]
            else:
                # Accept with probability exp(-delta/T)
                delta = new_size - len(S)
                prob = np.exp(-delta / max(temperature, 0.01))
                if random.random() < prob:
                    S = S_new
                    S_mask = S_mask_new
            
            temperature *= cooling
        else:
            break
    
    return best_S


def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy construction with multiple random restarts
    for attempt in range(30):
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
    
    # Strategy 2: Walk-SAT from greedy solutions
    if best_set is not None:
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - 40:
                break
            
            # Start from a perturbed version of the best solution
            S_perturbed = best_set[:]
            # Remove a random point
            if len(S_perturbed) > 1:
                p_remove = rng.choice(S_perturbed)
                S_perturbed.remove(p_remove)
            
            result = _walk_sat(S_perturbed, max_steps=3000, rng=rng)
            reduced = _reduce_to_minimal(result, deadline_margin=5)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
        
        if best_size <= 8:
            return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 3: Simulated annealing from best
    if best_set is not None and best_size > 8:
        result = _simulated_annealing(best_set, deadline_margin=15, max_iterations=1500)
        reduced = _reduce_to_minimal(result, deadline_margin=5)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
    
    # Strategy 4: Local search on best
    if best_set is not None and best_size > 8:
        improved_set = _local_search(best_set, deadline_margin=10, max_iterations=300)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    # Strategy 5: More Walk-SAT iterations from random cores
    for attempt in range(30):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Start with a random small core
        core_size = rng.choice([2, 3, 4, 5])
        core = rng.sample(range(N_TOTAL), core_size)
        core_mask = 0
        for x in core:
            core_mask |= (1 << x)
        
        if not _is_no_isosceles(core_mask, core):
            continue
        
        result = _walk_sat(core, max_steps=2000, rng=rng)
        reduced = _reduce_to_minimal(result, deadline_margin=5)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 8:
            break
    
    # Strategy 6: Final local search
    if best_set is not None and best_size > 8:
        improved_set = _local_search(best_set, deadline_margin=5, max_iterations=200)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    if best_set is None:
        # Fallback: greedy construction
        chosen = []
        chosen_mask = 0
        for i in range(N_TOTAL):
            if _can_add(i, chosen_mask, chosen):
                chosen.append(i)
                chosen_mask |= (1 << i)
        best_set = _reduce_to_minimal(chosen, deadline_margin=5)
    
    return [_ALL_POINTS[i] for i in best_set]
