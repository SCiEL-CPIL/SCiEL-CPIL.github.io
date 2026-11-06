"""Direct constructive approach for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and isosceles conflict relations using bitmasks."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # Precompute per-pair blocked sets as numpy boolean arrays
    # pair_blocked[i,j] = boolean array of all k such that {i,j,k} forms isosceles triangle
    pair_blocked = np.zeros((V, V, V), dtype=bool)
    
    for i in range(V):
        for j in range(i + 1, V):
            dab = dist[i, j]
            # k where d(i,k) == dab (k != j)
            mask1 = (dist[i] == dab).copy()
            mask1[j] = False
            # k where d(j,k) == dab (k != i)
            mask2 = (dist[j] == dab).copy()
            mask2[i] = False
            # k where d(i,k) == d(j,k) (perpendicular bisector)
            mask3 = (dist[i] == dist[j]).copy()
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            pair_blocked[i, j] = combined
            pair_blocked[j, i] = combined
    
    return points, dist, pair_blocked

def _is_valid_set(S, pair_blocked):
    """Check if S contains no isosceles triangle using precomputed pair conflicts."""
    S_list = list(S)
    S_set = set(S_list)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            blocked = pair_blocked[a, b]
            # Check if any other point in S is in the blocked set
            for k in S_list:
                if k != a and k != b and blocked[k]:
                    return False
    return True

def _is_maximal(S, pair_blocked):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    for p in range(V):
        if p in S_set:
            continue
        # Check if adding p would create an isosceles triangle with any pair in S
        can_add = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a, b = S[i], S[j]
                if pair_blocked[a, b][p]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _can_add_point(p, S, pair_blocked):
    """Check if point p can be added to set S without creating isosceles."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if pair_blocked[a, b][p]:
                return False
    return True

def _get_blocked_points(p, S, pair_blocked):
    """Get set of points that become blocked when adding p to S."""
    blocked = set()
    for s in S:
        mask = pair_blocked[p, s]
        for k in range(V):
            if k != p and k != s and mask[k]:
                blocked.add(k)
    return blocked

def _construct_direct(dist, pair_blocked, points, time_budget=280.0):
    """
    Direct construction with optimized local search and branch-and-bound.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V + 1
    best_set = None
    
    def _greedy_build(rng, max_iter=100):
        """Greedy build: pick point that blocks most remaining points."""
        S = []
        U = set(range(V))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if not U:
                break
            
            # Find candidates that can be added
            candidates = []
            for p in U:
                if _can_add_point(p, S, pair_blocked):
                    candidates.append(p)
            
            if not candidates:
                break
            
            # Score each candidate by how many points in U it would block
            best_score = -1
            best_p = None
            
            # Sample candidates if too many
            if len(candidates) > 30:
                sampled = rng.sample(candidates, 30)
            else:
                sampled = candidates
            
            for p in sampled:
                # Count how many points in U would be blocked by adding p
                blocked_count = 0
                for s in S:
                    mask = pair_blocked[p, s]
                    for u in U:
                        if u != p and u != s and mask[u]:
                            blocked_count += 1
                
                score = blocked_count
                
                if score > best_score or (score == best_score and rng.random() < 0.3):
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            S.append(best_p)
            U.discard(best_p)
            
            # Remove points that can no longer be added
            new_U = set()
            for u in U:
                if _can_add_point(u, S, pair_blocked):
                    new_U.add(u)
            U = new_U
        
        return S
    
    def _local_search(S, rng):
        """Local search: try removals and swaps to reduce size while maintaining maximality."""
        S = list(S)
        
        # First: try to remove points (keep maximality)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(S)):
                trial_S = S[:idx] + S[idx+1:]
                if len(trial_S) < 2:
                    continue
                # Check validity
                if not _is_valid_set(trial_S, pair_blocked):
                    continue
                # Check maximality
                if _is_maximal(trial_S, pair_blocked):
                    S = trial_S
                    improved = True
                    break
        
        # Then: try swaps (remove one, add one)
        for it in range(30):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(S)):
                for p in range(V):
                    if p in S:
                        continue
                    trial_S = S[:idx] + [p] + S[idx+1:]
                    # Check validity
                    if not _is_valid_set(trial_S, pair_blocked):
                        continue
                    # Check maximality
                    if _is_maximal(trial_S, pair_blocked):
                        S = trial_S
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return S
    
    def _branch_and_bound(S, remaining, depth, rng):
        """
        Branch-and-bound to find minimal maximal set.
        S: current selected points
        remaining: points that can still be added
        depth: recursion depth
        """
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        # If we already have a good solution, prune
        if len(S) >= best_size:
            return
        
        # Check if S is maximal
        if _is_maximal(S, pair_blocked):
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
            return
        
        # If no remaining points can be added but S is not maximal, this is a dead end
        if not remaining:
            return
        
        # Try each remaining point
        for p in remaining:
            if time.time() > deadline:
                return
            
            new_S = S + [p]
            if len(new_S) >= best_size:
                continue
            
            # Check validity
            if not _is_valid_set(new_S, pair_blocked):
                continue
            
            # Compute new remaining set
            new_remaining = set()
            for u in remaining:
                if u != p and _can_add_point(u, new_S, pair_blocked):
                    new_remaining.add(u)
            
            _branch_and_bound(new_S, new_remaining, depth + 1, rng)
    
    # Multiple trials with different random seeds
    for trial in range(150):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build
        S = _greedy_build(rng)
        
        if not S:
            continue
        
        # Local search
        S = _local_search(S, rng)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Try branch-and-bound from this solution (limited depth)
        if len(S) <= best_size and time.time() < deadline - 30:
            # Compute remaining points
            remaining = set()
            for p in range(V):
                if p not in S and _can_add_point(p, S, pair_blocked):
                    remaining.add(p)
            
            # Only do B&B if remaining is small enough
            if len(remaining) <= 20:
                _branch_and_bound(S, remaining, 0, rng)
    
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_blocked = _precompute()
    return _construct_direct(dist, pair_blocked, points)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
