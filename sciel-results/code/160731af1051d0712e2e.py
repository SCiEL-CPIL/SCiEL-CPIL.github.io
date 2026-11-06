"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Precompute grid points
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances between all pairs of points
# dist2[i, j] = squared distance between point i and point j
dist2 = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
xs = [p[0] for p in POINTS]
ys = [p[1] for p in POINTS]
xs = np.array(xs)
ys = np.array(ys)

# Vectorized distance calculation
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = dx**2 + dy**2

def can_add_point_vectorized(p_idx, S_indices):
    """
    Check if p_idx can be added to the set of points S_indices without creating an isosceles triangle.
    S_indices is a list or array of indices.
    
    Conditions for validity of S U {p}:
    1. For all a in S, dist(p, a) must be unique among {dist(p, b) for b in S}.
       i.e., no two points in S are at the same distance from p.
    2. For all a, b in S, dist(a, b) must not equal dist(p, a) or dist(p, b).
       Actually, the condition is: no isosceles triangle in S U {p}.
       Existing triangles in S are assumed valid.
       New triangles involving p: (p, a, b) for a, b in S.
       (p, a, b) is isosceles if:
       - dist(p, a) == dist(p, b)  (handled by condition 1)
       - dist(p, a) == dist(a, b)
       - dist(p, b) == dist(a, b)
       
       So we need to check:
       For each a in S:
         d_pa = dist2[p_idx, a]
         If there exists b in S such that dist2[a, b] == d_pa, then invalid.
         (This covers both dist(p,a)==dist(a,b) and dist(p,b)==dist(a,b) if we check all a)
         Wait, if we check for each a in S if there is a b in S with dist(a,b) == dist(p,a), 
         does that cover dist(p,b) == dist(a,b)?
         If dist(p,b) == dist(a,b), then for index b in S, dist(p,b) is the distance from p to b.
         The check "for b in S, is there a c in S such that dist(b,c) == dist(p,b)?" would catch it.
         So yes, checking for each a in S if any b in S has dist(a,b) == dist(p,a) is sufficient.
    """
    if len(S_indices) == 0:
        return True
    
    S_arr = np.array(S_indices)
    
    # 1. Check uniqueness of distances from p to S
    dists_p_to_S = dist2[p_idx, S_arr]
    if len(dists_p_to_S) != len(np.unique(dists_p_to_S)):
        return False
        
    # 2. Check for conflicts: dist(p, a) == dist(a, b)
    # For each a in S, we want to know if any b in S is at distance dist(p, a) from a.
    # We can do this by iterating over a in S, but we want to vectorize.
    # Let D_pa = dist2[p_idx, S_arr]
    # For each a_i in S_arr, we look at the row dist2[a_i, S_arr] and check if D_pa[i] is in it.
    # This is a bit tricky to fully vectorize without a loop over S, but S is small (max ~100).
    # However, we can use a trick:
    # Create a boolean matrix M where M[i, j] = (dist2[S_arr[i], S_arr[j]] == dist2[p_idx, S_arr[i]])
    # Then check if any M[i, j] is True for i != j? No, b can be any point in S, including a? 
    # If b == a, dist(a, a) = 0. dist(p, a) is never 0. So b != a is guaranteed if dist > 0.
    # So we just need to check if there exists i, j in range(len(S)) such that dist2[S[i], S[j]] == dist2[p, S[i]].
    
    # Let's extract the submatrix of distances between points in S
    # sub_dist2[i, j] = dist2[S_arr[i], S_arr[j]]
    sub_dist2 = dist2[np.ix_(S_arr, S_arr)]
    
    # D_pa[i] = dist2[p_idx, S_arr[i]]
    # We need to check if for any i, D_pa[i] appears in sub_dist2[i, :]
    # Note: sub_dist2[i, i] = 0. D_pa[i] > 0. So we don't worry about i==j.
    
    # Vectorized check:
    # Expand D_pa to shape (len(S), 1)
    D_pa_col = dists_p_to_S[:, None]
    # Compare with sub_dist2 (shape len(S), len(S))
    # match[i, j] = (sub_dist2[i, j] == D_pa[i])
    matches = (sub_dist2 == D_pa_col)
    
    if np.any(matches):
        return False
        
    return True

def is_maximal_vectorized(S_indices):
    """
    Check if no point can be added to S_indices.
    """
    S_set = set(S_indices)
    S_arr = np.array(S_indices)
    
    # For each point p not in S, check if it can be added.
    # This is O(N^2 * K) if done naively.
    # We can try to vectorize the "can add" check for all p at once?
    # It's complex. Given N=16, NUM_POINTS=256.
    # Checking all 256 points individually with vectorized inner loop is fast enough.
    
    for p_idx in range(NUM_POINTS):
        if p_idx in S_set:
            continue
        if can_add_point_vectorized(p_idx, S_indices):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Strategy: Multiple greedy restarts with vectorized checks.
    # Greedy: Pick points in random order, add if valid.
    # Then, try to minimize by removing points if maximality is preserved.
    
    num_restarts = 5000  # We can afford many restarts with vectorization
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        
        # 1. Greedy Construction
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        current_set = []
        
        for p_idx in order:
            if can_add_point_vectorized(p_idx, current_set):
                current_set.append(p_idx)
                
        # 2. Minimization Phase
        # Try to remove points. If removal keeps the set maximal, accept.
        # We do this iteratively.
        changed = True
        while changed:
            changed = False
            # Try removing points in random order
            indices_to_try = list(range(len(current_set)))
            random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if time.time() > deadline:
                    break
                
                p_to_remove = current_set[idx]
                trial_set = [x for i, x in enumerate(current_set) if i != idx]
                
                # Check if trial_set is maximal
                if is_maximal_vectorized(trial_set):
                    current_set = trial_set
                    changed = True
                    break
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Optional: Local search / perturbation
        # If we have time, we could try swapping points or more complex moves.
        # But for now, let's see if greedy + minimize is enough with many restarts.
        
        # If we find a very small set, we might stop early or just continue to find even smaller.
        if best_size <= 10: # Heuristic: if we get very small, it's hard to improve much
             # But we should still try to find the minimum possible.
             pass

    if best_set is None:
        # Fallback: simple greedy
        current_set = []
        for p_idx in range(NUM_POINTS):
            if can_add_point_vectorized(p_idx, current_set):
                current_set.append(p_idx)
        return [POINTS[i] for i in current_set]
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
