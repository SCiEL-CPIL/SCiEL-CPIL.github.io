import time
import random
import itertools
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
    
    # For each pair (i, j), find all k such that {i,j,k} forms an isosceles triangle
    # Isosceles means: d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k)
    
    # We will store for each pair (i, j) the set of blocked points.
    # Since V=256, there are 256*255/2 = 32640 pairs.
    # Storing a set for each pair might be memory heavy but manageable.
    # Alternatively, we can store a bitset (numpy array of bools or integers).
    # Let's use a dictionary mapping (i, j) to a numpy array of blocked points (indices).
    # Or better, a 2D array of bitsets? 
    # A 256x256 array of 256-bit integers is 256*256*8 bytes = 524KB. Very small.
    
    # Let's create a 2D array `blocked_by_pair` where blocked_by_pair[i, j] is a bitset of points k
    # such that {i, j, k} is isosceles.
    
    blocked_by_pair = np.zeros((V, V), dtype=np.uint64) # Wait, 256 bits needs 4 uint64s.
    # Let's use a list of lists of sets for simplicity first, then optimize if needed.
    # Actually, let's precompute a 3D boolean array? 256*256*256 = 16M booleans = 16MB. Fine.
    # iso[i, j, k] = True if {i, j, k} is isosceles.
    
    iso = np.zeros((V, V, V), dtype=bool)
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i, j]
            # k where d(i,k) == dij (but k != j)
            mask1 = (dist[i] == dij)
            mask1[j] = False
            # k where d(j,k) == dij (but k != i)
            mask2 = (dist[j] == dij)
            mask2[i] = False
            # k where d(i,k) == d(j,k) (perpendicular bisector)
            mask3 = (dist[i] == dist[j])
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            ks = np.where(combined)[0]
            for k in ks:
                iso[i, j, k] = True
                iso[j, i, k] = True
                # Note: iso is symmetric in i,j but not necessarily k? 
                # The condition is symmetric for the set {i,j,k}.
                # So iso[i, j, k] should imply iso[k, i, j] etc.
                # Let's just fill all permutations or rely on the check logic.
                # For efficiency in checking if a point p can be added to S:
                # We need to check if for any pair (a, b) in S, iso[a, b, p] is True.
                # So having iso[a, b, p] is sufficient.
                
    return points, dist, iso

# Global precomputation
_points, _dist, _iso = _precompute()

def _can_add_point(p, S):
    """Check if point p can be added to set S without forming an isosceles triangle."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if _iso[S[i], S[j], p]:
                return False
    return True

def _is_maximal(S):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    for p in range(V):
        if p in S_set:
            continue
        if _can_add_point(p, S):
            return False
    return True

def _greedy_construct(rng, time_budget_remaining):
    """
    Greedy construction with heuristic.
    Returns a set S.
    """
    S = []
    remaining = set(range(V))
    
    while remaining:
        # Find valid candidates
        valid = []
        for p in remaining:
            if _can_add_point(p, S):
                valid.append(p)
        
        if not valid:
            break
        
        # Heuristic: Pick point with max "blocking degree"
        # Degree of v = number of u in remaining \ {v} that are blocked by v given S
        # u is blocked by v if exists a in S such that {u, v, a} is isosceles
        # Note: if S is empty, v doesn't block u (unless we consider pairs in S, which are none).
        # If S is empty, deg is 0 for all.
        
        best_v = None
        best_deg = -1
        best_ties = []
        
        # Shuffle valid to break ties randomly if needed, or just pick first max
        rng.shuffle(valid)
        
        for v in valid:
            deg = 0
            for u in remaining:
                if u == v:
                    continue
                # Check if u is blocked by v given S
                # u is blocked if exists a in S such that {u, v, a} is isosceles
                for a in S:
                    if _iso[u, v, a]:
                        deg += 1
                        break
            if deg > best_deg:
                best_deg = deg
                best_v = v
                best_ties = [v]
            elif deg == best_deg:
                best_ties.append(v)
        
        # Randomly pick from ties
        chosen = rng.choice(best_ties)
        S.append(chosen)
        remaining.remove(chosen)
        
        # Update remaining: remove points that are now blocked by S
        # A point q is blocked if exists a, b in S such that {q, a, b} is isosceles
        new_remaining = set()
        for q in remaining:
            compatible = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if _iso[S[i], S[j], q]:
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                new_remaining.add(q)
        remaining = new_remaining
        
        # Check time
        if time.time() > time_budget_remaining:
            break
            
    return S

def _local_search_reduce(S, time_budget_remaining):
    """
    Try to remove points from S while maintaining validity and maximality.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        # Shuffle S to try different removal orders
        for idx in range(len(S)):
            if time.time() > time_budget_remaining:
                return S
            
            # Try removing S[idx]
            trial_S = S[:idx] + S[idx+1:]
            
            # Check validity of trial_S
            valid = True
            for i in range(len(trial_S)):
                for j in range(i + 1, len(trial_S)):
                    for k in range(len(trial_S)):
                        if k != i and k != j:
                            if _iso[trial_S[i], trial_S[j], trial_S[k]]:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check maximality of trial_S
            if _is_maximal(trial_S):
                S = trial_S
                improved = True
                # Restart the loop or continue? 
                # Restarting is safer to find more reductions.
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    best_set = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple greedy constructions
    trial = 0
    while time.time() < deadline:
        # Allocate some time for greedy, some for local search
        # Greedy is fast, local search can be slower
        greedy_budget = time.time() + 5  # 5 seconds for greedy
        ls_budget = deadline - 5  # Rest for local search
        
        S = _greedy_construct(rng, greedy_budget)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Only do local search if we have time and the set is promising
        if time.time() < deadline - 10:
            S_reduced = _local_search_reduce(S, deadline - 5)
            if len(S_reduced) < best_size:
                best_size = len(S_reduced)
                best_set = S_reduced.copy()
        
        trial += 1
        # Increase randomness or change strategy slightly?
        # For now, just run more trials.
        
    if best_set is None:
        # Fallback: return empty set? Or a trivial maximal set?
        # Empty set is not maximal if N>1.
        # A single point is not maximal.
        # Just return whatever we found, or an empty list if nothing found (should not happen)
        return []
        
    return best_set

# Test harness (optional, remove for final submission if not needed, but good for checking)
if __name__ == "__main__":
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Size: {len(result)}, Time: {end - start:.2f}s")
    # Verify validity
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(len(result)):
                if k != i and k != j:
                    if _iso[result[i], result[j], result[k]]:
                        valid = False
                        break
    print(f"Valid: {valid}")
    # Verify maximality
    print(f"Maximal: {_is_maximal(result)}")
