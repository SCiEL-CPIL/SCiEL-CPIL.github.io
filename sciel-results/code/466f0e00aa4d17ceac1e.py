"""Minimum maximal no-isosceles set via bitmask greedy + local search."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points
ALL_MASK = (1 << V) - 1

def _precompute():
    """Precompute points, distances, and pair-conflict bitmasks."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2).astype(np.int32)
    
    # For each pair (i, j), compute bitmask of all k such that {i,j,k} is isosceles
    # Isosceles: d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k)
    pair_conflicts = np.zeros((V, V), dtype=np.uint64)
    # Actually V=256, so we need multiple uint64s or use a list of sets
    # Let's use a list of sets for simplicity, then convert to frozenset for hashing
    
    pair_conf = [None] * (V * V)
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i, j]
            # k where d(i,k) == dij (k != j)
            mask1 = (dist[i] == dij)
            mask1[j] = False
            # k where d(j,k) == dij (k != i)
            mask2 = (dist[j] == dij)
            mask2[i] = False
            # k where d(i,k) == d(j,k)
            mask3 = (dist[i] == dist[j])
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            ks = np.where(combined)[0]
            s = set(ks.tolist())
            pair_conf[i * V + j] = s
            pair_conf[j * V + i] = s
    return points, dist, pair_conf

def _pair_key(a, b):
    if a > b:
        a, b = b, a
    return a * V + b

def _construct(dist, pair_conf, time_budget=310.0):
    """
    Greedy construction with bitmask tracking + local search to minimize set size.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V
    best_set = None
    
    def _time_left():
        return time.time() < deadline
    
    def _greedy_build(rng):
        """Greedy: pick points that block the most remaining candidates."""
        S = []
        S_set = set()
        blocked = set()  # points that can't be added due to isosceles with pairs in S
        U = set(range(V))
        
        while U:
            if not _time_left():
                break
            # Find candidates in U that aren't blocked
            candidates = [p for p in U if p not in blocked]
            if not candidates:
                break
            
            # Score each candidate by how many new points it blocks
            best_score = -1
            best_p = None
            best_new_blocked = set()
            
            cand_list = candidates[:]
            rng.shuffle(cand_list)
            
            # Limit candidates for speed
            if len(cand_list) > 80:
                cand_list = cand_list[:80]
            
            for p in cand_list:
                # Compute new blocked points if we add p
                new_blocked = set()
                for a in S:
                    key = _pair_key(a, p)
                    pc = pair_conf[key]
                    for c in pc:
                        if c in U and c != p:
                            new_blocked.add(c)
                score = len(new_blocked)
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_new_blocked = new_blocked
                elif score == best_score and rng.random() < 0.2:
                    best_p = p
                    best_new_blocked = new_blocked
            
            if best_p is None:
                break
            
            S.append(best_p)
            S_set.add(best_p)
            U.discard(best_p)
            blocked |= best_new_blocked
            U -= best_new_blocked
        
        return S, blocked
    
    def _is_maximal(S, blocked):
        """S is maximal iff every point is either in S or in blocked."""
        for p in range(V):
            if p not in S and p not in blocked:
                return False
        return True
    
    def _recompute_blocked(S):
        """Recompute blocked set from scratch for a given S."""
        blocked = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                key = _pair_key(S[i], S[j])
                blocked |= pair_conf[key]
        return blocked
    
    def _local_search(S, blocked):
        """Try to remove points while maintaining maximality."""
        S = list(S)
        improved = True
        while improved and _time_left():
            improved = False
            for idx in range(len(S)):
                if not _time_left():
                    break
                p = S[idx]
                # Try removing p
                S_trial = S[:idx] + S[idx+1:]
                if not S_trial:
                    continue
                # Recompute blocked for S_trial
                blocked_trial = _recompute_blocked(S_trial)
                # Check maximality
                if _is_maximal(S_trial, blocked_trial):
                    S = S_trial
                    blocked = blocked_trial
                    improved = True
                    break
        return S, blocked
    
    # Multiple trials with different random seeds
    for trial in range(2000):
        if not _time_left():
            break
        
        rng = random.Random(trial * 7919 + 12345)
        S, blocked = _greedy_build(rng)
        
        if not S:
            continue
        
        # Local search to minimize
        S, blocked = _local_search(S, blocked)
        
        # Verify maximality
        if not _is_maximal(S, blocked):
            continue
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    return best_set

def construct():
    points, dist, pair_conf = _precompute()
    result_indices = _construct(dist, pair_conf)
    if result_indices is None:
        return []
    return [points[i] for i in result_indices]

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
