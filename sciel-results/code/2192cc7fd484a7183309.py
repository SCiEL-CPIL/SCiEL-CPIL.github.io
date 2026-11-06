import itertools
import time
import random
import numpy as np

N = 16
V = N * N
FULL_MASK = (1 << V) - 1

def _precompute():
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
        
    # Precompute conflict masks
    # conflict_mask[i][j] is a bitmask of points k such that {i, j, k} forms an isosceles triangle
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
            
    return points, conflict_mask, dist

_points, _conflict_mask, _dist = None, None, None

def _init():
    global _points, _conflict_mask, _dist
    if _points is None:
        _points, _conflict_mask, _dist = _precompute()

def _is_valid_set(S_mask, S_list):
    """Check if S is a valid no-isosceles set."""
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if S_mask & _conflict_mask[a][b]:
                return False
    return True

def _is_maximal_set(S_mask, S_list):
    """Check if S is maximal."""
    blocked = 0
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            blocked |= _conflict_mask[a][b]
    return (S_mask | blocked) == FULL_MASK

def _get_blocked_mask(S_list, S_mask):
    """Compute the blocked mask for a given set."""
    blocked = 0
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            blocked |= _conflict_mask[a][b]
    return blocked

def _greedy_construct(seed):
    """Construct a maximal set using greedy heuristic."""
    rng = random.Random(seed)
    S = []
    S_mask = 0
    blocked = 0
    
    # Order points randomly to avoid bias
    points_order = list(range(V))
    rng.shuffle(points_order)
    
    for p in points_order:
        if (S_mask | blocked) & (1 << p):
            continue
            
        # Check if p can be added
        valid = True
        for a in S:
            if _conflict_mask[p][a] & S_mask:
                valid = False
                break
                
        if not valid:
            continue
            
        # Add p
        S.append(p)
        S_mask |= (1 << p)
        for a in S[:-1]:
            blocked |= _conflict_mask[p][a]
            
        # If all points are covered, we are done
        if (S_mask | blocked) == FULL_MASK:
            break
            
    # Ensure maximality by adding random points if not maximal
    while not _is_maximal_set(S_mask, S):
        candidates = []
        for p in range(V):
            if (S_mask | blocked) & (1 << p):
                continue
            valid = True
            for a in S:
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
            if valid:
                candidates.append(p)
                
        if not candidates:
            break
            
        p = rng.choice(candidates)
        S.append(p)
        S_mask |= (1 << p)
        for a in S[:-1]:
            blocked |= _conflict_mask[p][a]
            
    return S

def _remove_and_repair(S, seed):
    """
    Try to remove points from S while maintaining maximality.
    Returns a new set if improvement is found, else None.
    """
    rng = random.Random(seed)
    S = list(S)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
        
    improved = False
    # Try removing each point in random order
    indices = list(range(len(S)))
    rng.shuffle(indices)
    
    for idx in indices:
        p_remove = S[idx]
        trial_S = S[:idx] + S[idx+1:]
        trial_mask = S_mask & ~(1 << p_remove)
        
        # Check validity (should still be valid since we are removing)
        # Check maximality
        if _is_maximal_set(trial_mask, trial_S):
            S = trial_S
            S_mask = trial_mask
            improved = True
            # Restart loop as indices changed
            indices = list(range(len(S)))
            rng.shuffle(indices)
            
    if improved:
        return S
    return None

def _perturb(S, rng, strength=1):
    """
    Perturb the set by performing random swaps or removals/additions.
    Must return a valid set.
    """
    S = list(S)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
        
    # Try a few random moves
    for _ in range(strength * 10):
        move_type = rng.random()
        
        if move_type < 0.5 and len(S) > 1:
            # Swap
            i = rng.randrange(len(S))
            p_out = S[i]
            base = S[:i] + S[i+1:]
            base_mask = S_mask & ~(1 << p_out)
            
            # Find a valid point to add
            candidates = []
            for p_in in range(V):
                if base_mask & (1 << p_in):
                    continue
                valid = True
                for a in base:
                    if _conflict_mask[p_in][a] & base_mask:
                        valid = False
                        break
                if valid:
                    candidates.append(p_in)
            
            if candidates:
                p_in = rng.choice(candidates)
                new_S = base + [p_in]
                new_mask = base_mask | (1 << p_in)
                if _is_valid_set(new_mask, new_S):
                    S = new_S
                    S_mask = new_mask
                    break
                    
        elif move_type < 0.8:
            # Add
            candidates = []
            for p_in in range(V):
                if S_mask & (1 << p_in):
                    continue
                valid = True
                for a in S:
                    if _conflict_mask[p_in][a] & S_mask:
                        valid = False
                        break
                if valid:
                    candidates.append(p_in)
            
            if candidates:
                p_in = rng.choice(candidates)
                new_S = S + [p_in]
                new_mask = S_mask | (1 << p_in)
                if _is_valid_set(new_mask, new_S):
                    S = new_S
                    S_mask = new_mask
                    break
                    
        else:
            # Remove (if maximal, this will break maximality, but we just need a valid set to restart ILS)
            # Actually, for ILS we usually want to stay in the neighborhood of valid maximal sets.
            # Let's stick to swaps and adds for perturbation to keep it valid.
            pass
            
    return S

def construct():
    _init()
    
    best_set = None
    best_size = float('inf')
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Phase 1: Generate multiple initial solutions using Greedy
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        S = _greedy_construct(seed * 100 + 1)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
    if best_set is None:
        return []
        
    # Phase 2: Iterated Local Search (ILS)
    rng = random.Random(42)
    current_S = best_set
    current_size = best_size
    
    # We will run ILS until time limit
    # Each iteration:
    # 1. Try to improve current_S via remove-and-repair
    # 2. If no improvement, perturb current_S
    # 3. If perturbed solution is better, accept it
    
    while time.time() < deadline - 10:
        # Step 1: Local Search (Remove and Repair)
        improved_S = _remove_and_repair(current_S, rng.randint(0, 10000))
        
        if improved_S is not None:
            current_S = improved_S
            current_size = len(current_S)
            if current_size < best_size:
                best_size = current_size
                best_set = list(current_S)
                # If we reach a very small size, we can stop or keep polishing
                if best_size <= 8:
                    break
        else:
            # Step 2: Perturb
            perturbed_S = _perturb(current_S, rng, strength=1)
            
            # If perturbation resulted in a valid set (which it should),
            # we might want to check if it's better immediately or just move to it.
            # Since perturb usually keeps size or increases it slightly, we move to it
            # to escape local minima.
            if _is_valid_set(0, perturbed_S): # Quick check
                # Ensure maximality for the next remove-and-repair step?
                # _remove_and_repair assumes input is maximal?
                # Actually, _remove_and_repair just checks if removing a point keeps it maximal.
                # If the input is not maximal, it might not improve.
                # So we should ensure perturbed_S is maximal or just let it be.
                # Let's force maximality by adding points if needed?
                # Or just use it as a new starting point for the next local search.
                
                # To be safe, let's make it maximal before local search
                S_mask = 0
                for p in perturbed_S:
                    S_mask |= (1 << p)
                
                if not _is_maximal_set(S_mask, perturbed_S):
                    # Add random points to make it maximal
                    blocked = _get_blocked_mask(perturbed_S, S_mask)
                    while not _is_maximal_set(S_mask, perturbed_S):
                        candidates = []
                        for p in range(V):
                            if (S_mask | blocked) & (1 << p):
                                continue
                            valid = True
                            for a in perturbed_S:
                                if _conflict_mask[p][a] & S_mask:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(p)
                        if not candidates:
                            break
                        p = rng.choice(candidates)
                        perturbed_S.append(p)
                        S_mask |= (1 << p)
                        for a in perturbed_S[:-1]:
                            blocked |= _conflict_mask[p][a]
                            
                current_S = perturbed_S
                current_size = len(current_S)
                
                # If the perturbed solution is already better (unlikely for random perturbation
                # unless we got lucky with a swap that reduced size), update best
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current_S)

    return [_points[i] for i in best_set]
