"""Direct constructive approach for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and isosceles conflict relations."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # Precompute per-pair blocked sets as bitmasks
    # pair_blocked[i,j] = bitmask of all k such that {i,j,k} forms isosceles triangle
    # Store as list of arrays for fast access
    pair_blocked = np.zeros((V, V), dtype=object)
    
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
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            blocked = pair_blocked[a, b]
            # Check if any other point in S is in the blocked set
            for k in S:
                if k != a and k != b and blocked[k]:
                    return False
    return True

def _is_maximal(S, dist, pair_blocked):
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

def _get_blocked_by_pair(a, b, pair_blocked):
    """Get set of points blocked by pair (a,b)."""
    blocked = pair_blocked[a, b]
    return set(np.where(blocked)[0])

def _construct_direct(dist, pair_blocked, points, time_budget=280.0):
    """
    Direct construction with optimized local search.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V + 1
    best_set = None
    
    rng = random.Random(42)
    
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
            best_blocked_mask = None
            
            # Sample candidates if too many
            if len(candidates) > 40:
                sampled = rng.sample(candidates, 40)
            else:
                sampled = candidates
            
            for p in sampled:
                # Compute blocked points in U
                new_blocked = 0
                for a in S:
                    blocked_mask = pair_blocked[a, p]
                    # Count blocked points in U
                    for u in U:
                        if u != p and blocked_mask[u]:
                            new_blocked += 1
                
                # Also consider blocking effect of pairs (p, s) for s in S
                # Actually the blocking is: for any existing pair (a,b) in S, 
                # adding p creates new pairs (a,p) and (b,p) which block more points
                # But the key metric is: how many points in U become non-addable
                # A point u in U becomes non-addable if there exists a pair (x,y) in S∪{p}
                # such that u is in pair_blocked[x,y]
                
                # Simpler: count how many u in U are blocked by at least one new pair
                # New pairs are (p, s) for s in S
                blocked_count = 0
                for s in S:
                    blocked_mask = pair_blocked[p, s]
                    for u in U:
                        if u != p and u != s and blocked_mask[u]:
                            blocked_count += 1
                
                score = blocked_count
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_blocked_mask = None
                elif score == best_score and rng.random() < 0.2:
                    best_p = p
                    best_blocked_mask = None
            
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
        for it in range(100):
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
                if _is_maximal(trial_S, dist, pair_blocked):
                    S = trial_S
                    improved = True
                    break
            if not improved:
                break
        
        # Then: try swaps (remove one, add one, possibly reduce size or same size)
        for it in range(50):
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
                    if _is_maximal(trial_S, dist, pair_blocked):
                        S = trial_S
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return S
    
    # Multiple trials with different random seeds
    for trial in range(200):
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
    
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_blocked = _precompute()
    return _construct_direct(dist, pair_blocked, points)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
