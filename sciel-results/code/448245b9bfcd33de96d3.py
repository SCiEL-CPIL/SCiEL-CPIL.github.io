"""Direct constructive approach for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and pair-level isosceles conflicts."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # Precompute for each pair (a, b), the set of points c such that {a,b,c} is isosceles
    # pair_conflicts[a, b] = set of c where triangle (a,b,c) is isosceles
    pair_conflicts = np.zeros((V, V), dtype=object)
    
    for a in range(V):
        for b in range(a + 1, V):
            dab = dist[a, b]
            # c is blocked if d(a,b)==d(a,c) or d(a,b)==d(b,c) or d(a,c)==d(b,c)
            mask1 = (dist[a] == dab)  # d(a,c) == d(a,b)
            mask2 = (dist[b] == dab)  # d(b,c) == d(a,b)
            mask3 = (dist[a] == dist[b])  # d(a,c) == d(b,c)
            combined = mask1 | mask2 | mask3
            combined[a] = False
            combined[b] = False
            pair_conflicts[a, b] = set(np.where(combined)[0].tolist())
            pair_conflicts[b, a] = pair_conflicts[a, b]
    
    return points, dist, pair_conflicts

def _can_add(p, S, pair_conflicts):
    """Check if point p can be added to set S without creating isosceles triangle."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if p in pair_conflicts[a, b]:
                return False
    return True

def _is_maximal(S, pair_conflicts):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    for p in range(V):
        if p in S_set:
            continue
        if _can_add(p, S, pair_conflicts):
            return False
    return True

def _greedy_build(pair_conflicts, rng, time_budget_check=None):
    """
    Greedy build: pick points that maximize blocked points.
    Returns a valid (but not necessarily maximal) set.
    """
    S = []
    U = set(range(V))
    
    while U:
        # Find all candidates that can be added
        candidates = []
        for p in U:
            if _can_add(p, S, pair_conflicts):
                candidates.append(p)
        
        if not candidates:
            break
        
        if time_budget_check and time_budget_check():
            break
        
        # Score each candidate by how many points in U it would block
        best_score = -1
        best_p = None
        best_blocked = set()
        
        # Sample up to 50 candidates for speed if many
        if len(candidates) > 50:
            sampled = rng.sample(candidates, 50)
        else:
            sampled = candidates
        
        for p in sampled:
            # Compute how many points in U would be blocked by adding p
            new_blocked = set()
            for a in S:
                conflicts = pair_conflicts[a, p]
                for u in U:
                    if u != p and u in conflicts:
                        new_blocked.add(u)
            
            score = len(new_blocked)
            
            if score > best_score:
                best_score = score
                best_p = p
                best_blocked = new_blocked
            elif score == best_score and rng.random() < 0.3:
                best_p = p
                best_blocked = new_blocked
        
        if best_p is None:
            break
        
        S.append(best_p)
        U.remove(best_p)
        if best_blocked:
            U -= best_blocked
    
    return S

def _local_search(S, pair_conflicts, rng, deadline):
    """
    Local search with removal and swap moves to minimize |S| while maintaining
    validity and maximality.
    """
    S = list(S)
    improved = True
    iters = 0
    max_iters = 200
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break
        
        # Try removals first
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            trial_S = S[:idx] + S[idx+1:]
            if not trial_S:
                continue
            # Check validity: no isosceles triangle in trial_S
            valid = True
            for i in range(len(trial_S)):
                for j in range(i + 1, len(trial_S)):
                    a, b = trial_S[i], trial_S[j]
                    for k in trial_S:
                        if k != a and k != b:
                            if k in pair_conflicts[a, b]:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            if _is_maximal(trial_S, pair_conflicts):
                S = trial_S
                improved = True
                break
        
        if improved or time.time() > deadline:
            continue
        
        # Try swaps: remove one point, add another
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:idx] + S[idx+1:]
            
            # Find points that can be added to S_without
            S_set = set(S_without)
            addable = []
            for p in range(V):
                if p in S_set:
                    continue
                if _can_add(p, S_without, pair_conflicts):
                    addable.append(p)
            
            for p in addable:
                if time.time() > deadline:
                    break
                trial_S = S_without + [p]
                # Check validity
                valid = True
                for i in range(len(trial_S)):
                    for j in range(i + 1, len(trial_S)):
                        a, b = trial_S[i], trial_S[j]
                        for k in trial_S:
                            if k != a and k != b:
                                if k in pair_conflicts[a, b]:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check maximality
                if _is_maximal(trial_S, pair_conflicts):
                    S = trial_S
                    improved = True
                    break
            
            if improved:
                break
    
    return S

def _construct_direct(dist, points, pair_conflicts, time_budget=280.0):
    """
    Direct construction with greedy build and local search.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V  # worst case
    best_set = None
    
    # Strategy: Try multiple random greedy builds + local search
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build
        S = _greedy_build(pair_conflicts, rng, time_budget_check=lambda: time.time() > deadline)
        
        if not S:
            continue
        
        # Local search to reduce size
        S = _local_search(S, pair_conflicts, rng, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_conflicts = _precompute()
    return _construct_direct(dist, points, pair_conflicts)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
