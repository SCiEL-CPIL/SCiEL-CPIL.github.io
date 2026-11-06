import itertools
import time
import random
import numpy as np

N = 16
V = N * N
ALL_MASK = (1 << V) - 1

def _precompute():
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
        
    # Precompute conflict masks
    # conflict_mask[i][j] is a Python int with bit k set if {i,j,k} is isosceles
    conflict_mask = [[0] * V for _ in range(V)]
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i][j]
            mask = 0
            for k in range(V):
                if k == i or k == j:
                    continue
                dik = dist[i][k]
                djk = dist[j][k]
                if dik == dij or djk == dij or dik == djk:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
            
    return points, conflict_mask

_points, _conflict_mask = None, None

def _init():
    global _points, _conflict_mask
    if _points is None:
        _points, _conflict_mask = _precompute()

def _is_valid(S_indices, S_mask):
    """
    Check if S_indices is a valid no-isosceles set.
    """
    for i in range(len(S_indices)):
        a = S_indices[i]
        for j in range(i + 1, len(S_indices)):
            b = S_indices[j]
            if S_mask & _conflict_mask[a][b]:
                return False
    return True

def _get_unblocked_count(S_indices, S_mask):
    """
    Returns the number of points not in S and not blocked by S.
    """
    blocked = 0
    for i in range(len(S_indices)):
        a = S_indices[i]
        for j in range(i + 1, len(S_indices)):
            b = S_indices[j]
            blocked |= _conflict_mask[a][b]
    
    # Unblocked are those not in S and not in blocked
    # We want count of bits set in (ALL_MASK & ~(S_mask | blocked))
    return (ALL_MASK & ~(S_mask | blocked)).bit_count()

def _greedy_build(seed):
    """
    Greedy construction of a maximal set.
    """
    rng = random.Random(seed)
    S = []
    S_mask = 0
    blocked = 0
    
    while True:
        if (S_mask | blocked) == ALL_MASK:
            break
            
        candidates_mask = ALL_MASK & ~(S_mask | blocked)
        
        if candidates_mask == 0:
            break

        best_score = -1
        best_candidate = -1
        
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            p = lsb.bit_length() - 1
            cm ^= lsb
            
            valid = True
            new_blocked = 0
            for a in S:
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
                new_blocked |= _conflict_mask[p][a]
            
            if not valid:
                continue
            
            new_coverage = new_blocked & ~(blocked | S_mask)
            score = new_coverage.bit_count()
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate != -1:
                if rng.random() < 0.5:
                    best_candidate = p

        if best_candidate == -1:
            break

        S.append(best_candidate)
        S_mask |= (1 << best_candidate)
        for a in S[:-1]:
            blocked |= _conflict_mask[best_candidate][a]

    return S

def _simulated_annealing(S_init, seed, time_limit=10.0):
    """
    Simulated Annealing to minimize |S| + alpha * |U(S)|.
    """
    rng = random.Random(seed)
    S = list(S_init)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    alpha = 1.0
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    start_time = time.time()
    best_S = list(S)
    best_cost = len(S) + alpha * _get_unblocked_count(S, S_mask)
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Remove a point
            if not S:
                continue
            idx = rng.randrange(len(S))
            p = S[idx]
            S_new = S[:idx] + S[idx+1:]
            S_mask_new = S_mask & ~(1 << p)
        elif move_type < 0.9:
            # Add a point
            # Find candidates not in S
            candidates_mask = ALL_MASK & ~S_mask
            if candidates_mask == 0:
                continue
            # Pick random candidate
            cm = candidates_mask
            count = cm.bit_count()
            r = rng.randrange(count)
            # Find the r-th set bit
            # Simple method: iterate
            tmp = cm
            for _ in range(r):
                lsb = tmp & -tmp
                tmp ^= lsb
            p = (tmp & -tmp).bit_length() - 1
            
            # Check validity of S U {p}
            valid = True
            for a in S:
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
            if not valid:
                continue
                
            S_new = S + [p]
            S_mask_new = S_mask | (1 << p)
        else:
            # Swap
            if not S:
                continue
            idx = rng.randrange(len(S))
            p_remove = S[idx]
            
            candidates_mask = ALL_MASK & ~S_mask
            if candidates_mask == 0:
                continue
            cm = candidates_mask
            count = cm.bit_count()
            r = rng.randrange(count)
            tmp = cm
            for _ in range(r):
                lsb = tmp & -tmp
                tmp ^= lsb
            p_add = (tmp & -tmp).bit_length() - 1
            
            # Check validity of (S \ {p_remove}) U {p_add}
            S_mask_temp = S_mask & ~(1 << p_remove)
            valid = True
            # Check p_add against S \ {p_remove}
            for a in S:
                if a == p_remove:
                    continue
                if _conflict_mask[p_add][a] & S_mask_temp:
                    valid = False
                    break
            if not valid:
                continue
            
            S_new = S[:idx] + S[idx+1:] + [p_add]
            S_mask_new = S_mask_temp | (1 << p_add)
            
        # Calculate cost
        unblocked_new = _get_unblocked_count(S_new, S_mask_new)
        cost_new = len(S_new) + alpha * unblocked_new
        cost_curr = len(S) + alpha * _get_unblocked_count(S, S_mask)
        
        delta = cost_new - cost_curr
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            S = S_new
            S_mask = S_mask_new
            
            if cost_new < best_cost:
                best_cost = cost_new
                best_S = list(S)
        
        T *= cooling
        
    return best_S

def construct():
    _init()
    
    best_size = float('inf')
    best_set = None
    start_time = time.time()
    deadline = start_time + 290.0
    
    trial_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        # Get initial greedy set
        S_greedy = _greedy_build(seed)
        
        # Run SA
        # Limit time per SA run to allow multiple trials
        sa_time = 5.0 if time.time() - start_time < 200 else 1.0
        S_sa = _simulated_annealing(S_greedy, seed, time_limit=sa_time)
        
        # Check if S_sa is maximal
        # If not, we might have ended in a non-maximal state.
        # We should try to repair it or discard it if it's not better.
        # Actually, the cost function penalizes non-maximality, so it should be close to maximal.
        # Let's verify maximality of best_S from SA.
        
        S_mask = 0
        for p in S_sa:
            S_mask |= (1 << p)
        
        unblocked = _get_unblocked_count(S_sa, S_mask)
        
        # If it's not maximal, try to add points to make it maximal?
        # Or just use it if it's valid and small?
        # The task requires a MAXIMAL set.
        # So we must ensure maximality.
        
        if unblocked > 0:
            # Try to repair by adding points greedily
            S_repair = list(S_sa)
            S_mask_repair = S_mask
            blocked_repair = 0
            for i in range(len(S_repair)):
                a = S_repair[i]
                for j in range(i + 1, len(S_repair)):
                    b = S_repair[j]
                    blocked_repair |= _conflict_mask[a][b]
            
            while (S_mask_repair | blocked_repair) != ALL_MASK:
                candidates_mask = ALL_MASK & ~(S_mask_repair | blocked_repair)
                if candidates_mask == 0:
                    break
                # Pick random candidate
                cm = candidates_mask
                count = cm.bit_count()
                r = random.Random(seed + trial_count + 1).randrange(count)
                tmp = cm
                for _ in range(r):
                    lsb = tmp & -tmp
                    tmp ^= lsb
                p = (tmp & -tmp).bit_length() - 1
                
                # Check validity
                valid = True
                for a in S_repair:
                    if _conflict_mask[p][a] & S_mask_repair:
                        valid = False
                        break
                if not valid:
                    continue
                
                S_repair.append(p)
                S_mask_repair |= (1 << p)
                for a in S_repair[:-1]:
                    blocked_repair |= _conflict_mask[p][a]
            
            S_final = S_repair
        else:
            S_final = S_sa
            
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = list(S_final)
            
            if best_size <= 5:
                break
                
        trial_count += 1

    if best_set is None:
        return []
        
    return [_points[i] for i in best_set]
