"""Direct constructive approach for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and pair-level isosceles conflicts as bitsets."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # Precompute for each pair (a, b), the set of points c such that {a,b,c} is isosceles
    # pair_mask[a, b] = boolean array of length V where True means c is blocked
    pair_mask = np.zeros((V, V, V), dtype=bool)
    
    for a in range(V):
        for b in range(a + 1, V):
            dab = dist[a, b]
            mask1 = (dist[a] == dab)  # d(a,c) == d(a,b)
            mask2 = (dist[b] == dab)  # d(b,c) == d(a,b)
            mask3 = (dist[a] == dist[b])  # d(a,c) == d(b,c)
            combined = mask1 | mask2 | mask3
            combined[a] = False
            combined[b] = False
            pair_mask[a, b] = combined
            pair_mask[b, a] = combined
    
    return points, dist, pair_mask

def _can_add(p, S_indices, pair_mask):
    """Check if point p can be added to set S without creating isosceles triangle."""
    n = len(S_indices)
    if n < 2:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            if pair_mask[S_indices[i], S_indices[j], p]:
                return False
    return True

def _is_maximal(S, pair_mask):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    n = len(S)
    for p in range(V):
        if p in S_set:
            continue
        can_add = True
        for i in range(n):
            for j in range(i + 1, n):
                if pair_mask[S[i], S[j], p]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _greedy_build(pair_mask, rng, deadline):
    """
    Greedy build: pick points that maximize blocked points.
    Returns a valid (but not necessarily maximal) set.
    """
    S = []
    U_mask = np.ones(V, dtype=bool)
    
    while U_mask.any():
        if time.time() > deadline:
            break
        
        # Find all candidates that can be added
        candidates = []
        for p in np.where(U_mask)[0]:
            if _can_add(p, S, pair_mask):
                candidates.append(p)
        
        if not candidates:
            break
        
        # Score each candidate by how many points in U it would block
        best_score = -1
        best_p = None
        
        # Sample up to 60 candidates for speed if many
        if len(candidates) > 60:
            sampled = rng.sample(candidates, 60)
        else:
            sampled = candidates
        
        for p in sampled:
            # Compute how many points in U would be blocked by adding p
            blocked_count = 0
            for a in S:
                # Count U points that conflict with pair (a, p)
                conflicts = pair_mask[a, p]
                blocked_count += np.sum(conflicts & U_mask)
            
            if blocked_count > best_score:
                best_score = blocked_count
                best_p = p
            elif blocked_count == best_score and rng.random() < 0.3:
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        U_mask[best_p] = False
        # Update U_mask: remove points blocked by new pairs with best_p
        for a in S[:-1]:
            U_mask &= ~pair_mask[a, best_p]
    
    return S

def _local_search(S, pair_mask, rng, deadline):
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
            if len(trial_S) < 2:
                continue
            # Check validity: no isosceles triangle in trial_S
            valid = True
            n = len(trial_S)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        if k != i and k != j:
                            if pair_mask[trial_S[i], trial_S[j], trial_S[k]]:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            if _is_maximal(trial_S, pair_mask):
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
                if _can_add(p, S_without, pair_mask):
                    addable.append(p)
            
            if not addable:
                continue
                
            # Try a few random addable points
            rng.shuffle(addable)
            for p in addable[:10]:
                if time.time() > deadline:
                    break
                trial_S = S_without + [p]
                # Check validity
                valid = True
                n = len(trial_S)
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(n):
                            if k != i and k != j:
                                if pair_mask[trial_S[i], trial_S[j], trial_S[k]]:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check maximality
                if _is_maximal(trial_S, pair_mask):
                    S = trial_S
                    improved = True
                    break
            
            if improved:
                break
    
    return S

def _construct_direct(dist, points, pair_mask, time_budget=290.0):
    """
    Direct construction with greedy build and local search.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V  # worst case
    best_set = None
    
    # Strategy: Try multiple random greedy builds + local search
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build
        S = _greedy_build(pair_mask, rng, deadline)
        
        if not S:
            continue
        
        # Local search to reduce size
        S = _local_search(S, pair_mask, rng, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            # If we find a very good solution, tighten deadline to allow more trials
            if best_size <= 55:
                deadline = start_time + min(time_budget, time_budget * 0.85)
    
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_mask = _precompute()
    return _construct_direct(dist, points, pair_mask)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
