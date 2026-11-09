"""Optimized approach: construct a maximal no-isosceles set on [0,N)^2.
Uses vectorized distance checks and local search to minimize set size."""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # safety margin before 300s kill

# Precompute all points and indices
ALL_POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=np.int32)
NUM_POINTS = len(ALL_POINTS)

# Precompute squared distances between all pairs
# D[i, j] is squared distance between ALL_POINTS[i] and ALL_POINTS[j]
# We can compute this using broadcasting
diff = ALL_POINTS[:, np.newaxis, :] - ALL_POINTS[np.newaxis, :, :]
DIST_MATRIX = np.sum(diff * diff, axis=2)

def _can_add_indices(p_idx, S_indices):
    """
    Check if point p_idx can be added to set S_indices.
    Returns True if no isosceles triangle is formed.
    """
    if len(S_indices) < 2:
        return True
    
    # Distances from p to all points in S
    d_p_s = DIST_MATRIX[p_idx, S_indices]
    
    # Distances between all pairs in S
    # We need to check if any d_p_s[i] == d_p_s[j] OR d_p_s[i] == d_s[i,j]
    
    # Check 1: Is p equidistant to any two points in S?
    # This is equivalent to checking if any value in d_p_s appears more than once?
    # No, that's not right. d_p_s[i] == d_p_s[j] means d(p, S[i]) == d(p, S[j]).
    # This forms an isosceles triangle with vertex p.
    
    # Efficient check for duplicates in d_p_s:
    # If len(set(d_p_s)) < len(d_p_s), there is a duplicate.
    if len(np.unique(d_p_s)) < len(d_p_s):
        return False

    # Check 2 & 3: Is d(p, S[i]) == d(S[i], S[j]) or d(p, S[j]) == d(S[i], S[j])?
    # This requires checking against the submatrix of distances within S.
    # Let S be sorted indices.
    # D_S is the submatrix of DIST_MATRIX for indices in S.
    D_S = DIST_MATRIX[np.ix_(S_indices, S_indices)]
    
    # For each pair (i, j) in S, let d_ij = D_S[i, j].
    # We need to ensure d_p_s[i] != d_ij and d_p_s[j] != d_ij.
    
    # Vectorized approach:
    # Create a matrix of d_p_s broadcasted against D_S.
    # d_p_s has shape (k,). D_S has shape (k, k).
    # We want to check if d_p_s[i] == D_S[i, j] OR d_p_s[j] == D_S[i, j].
    
    # Let's construct a boolean matrix.
    # match_ij = (d_p_s[:, None] == D_S) | (d_p_s[None, :] == D_S)
    # If any element in match_ij is True (excluding diagonal where i==j, but D_S[i,i]=0 and d_p_s>0 usually, 
    # unless p is in S which is not the case here), then it's invalid.
    
    # Note: D_S[i, i] = 0. d_p_s is distance to p. Since p is not in S, d_p_s > 0.
    # So diagonal of D_S (0) will never match d_p_s.
    
    # However, we must be careful. The condition is for distinct i, j.
    # The matrix match_ij will have True at (i, j) if d(p, S[i]) == d(S[i], S[j]) OR d(p, S[j]) == d(S[i], S[j]).
    
    # We can just check if any True exists in the off-diagonal?
    # Actually, if i=j, d(S[i], S[i])=0. d(p, S[i]) > 0. So no match on diagonal.
    # So we can just check `np.any(match_ij)`.
    
    # Optimization: Only check upper triangle? No, numpy is fast enough for small k.
    # k is at most 256. 256x256 matrix is small.
    
    # But we can be smarter.
    # For a fixed i, we check if d_p_s[i] is in D_S[i, :] (excluding i) or if any d_p_s[j] (j!=i) is D_S[i, j].
    
    # Let's stick to the broadcast for simplicity and speed (NumPy is fast).
    # d_p_s shape (k,)
    # D_S shape (k, k)
    
    # We need to check:
    # 1. d_p_s[i] == D_S[i, j] for j != i
    # 2. d_p_s[j] == D_S[i, j] for j != i
    
    # This is equivalent to:
    # (d_p_s[:, None] == D_S) | (d_p_s[None, :] == D_S)
    
    # Let's verify indices.
    # Row i, Col j:
    # Term 1: d_p_s[i] == D_S[i, j]
    # Term 2: d_p_s[j] == D_S[i, j]
    
    # This covers all cases.
    
    # One edge case: What if D_S[i, j] is 0? Only if i=j. But d_p_s > 0. So safe.
    
    match_matrix = (d_p_s[:, np.newaxis] == D_S) | (d_p_s[np.newaxis, :] == D_S)
    
    # If any match is found, it's an isosceles triangle.
    if np.any(match_matrix):
        return False
        
    return True

def _build_greedy_fast(order_seed=0):
    """
    Fast greedy construction.
    """
    rng = random.Random(order_seed)
    S_indices = []
    addable = set(range(NUM_POINTS))
    
    # Precompute random noise for tie-breaking
    noise = np.random.default_rng(order_seed).random(NUM_POINTS) * 0.01
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        # Iterate through addable points
        # To speed up, we can sample if addable is large, but 256 is small.
        for p_idx in addable:
            if not _can_add_indices(p_idx, S_indices):
                continue
            
            # Calculate "degree" or blocking count
            # How many other addable points q would become non-addable if p is added?
            # new_S = S_indices + [p_idx]
            # We need to count q in addable \ {p} such that not _can_add_indices(q, new_S)
            
            # This is expensive. 
            # Heuristic: Instead of exact block count, we can use a simpler heuristic or approximate it.
            # Or, we can just pick the first valid one with some randomization?
            # The "minimum degree" heuristic is good but expensive.
            
            # Let's try a cheaper heuristic: 
            # Score based on distance to existing points?
            # Or just random valid?
            
            # Given time constraints, let's stick to a simplified greedy:
            # Pick a random valid point, but prefer those with lower "connectivity" if we can compute it cheaply.
            
            # Actually, let's just pick the first valid point in a shuffled order?
            # That's the "random greedy".
            
            # Let's implement a slightly smarter one:
            # Count how many points in S are at the same distance from p?
            # If p is equidistant to many points in S, it's "bad" (high risk of future conflicts? No, it's already valid).
            
            # Let's just use the random valid point approach for speed, and rely on local search to fix it.
            
            # For the sake of this edit, I will implement a "random valid" greedy which is O(N * K * K) per step.
            # With N=256, K~10, it's fast.
            
            # To make it "minimum-like", we can sort candidates by distance to centroid? No.
            
            # Let's just pick one randomly from the valid ones to ensure diversity.
            # We will collect all valid ones and pick one.
            
            # This is O(M * K^2) where M is addable size.
            # M decreases as we go.
            
            pass # Placeholder, we will just pick the first valid one we find in a shuffled list
        
        # Re-implementation: Shuffled list of addable
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if _can_add_indices(p_idx, S_indices):
                best_p = p_idx
                break
                
        if best_p is None:
            break
            
        S_indices.append(best_p)
        addable.discard(best_p)
        
        # Update addable
        # A point q is addable if _can_add_indices(q, S_indices) is True
        # We can filter the remaining addable set.
        new_addable = set()
        for q in addable:
            if _can_add_indices(q, S_indices):
                new_addable.add(q)
        addable = new_addable
        
    return S_indices

def _local_search(S_indices, rng, max_iters=50):
    """
    Try to reduce the size of the set by removing points.
    If a point can be removed without violating the no-isosceles condition, remove it.
    Then, try to add points back to ensure maximality? 
    Wait, the problem asks for a MAXIMAL set.
    If we remove a point, the set might no longer be maximal (we might be able to add other points).
    But the score is just the number of points.
    So we want a set that is:
    1. No-isosceles
    2. Maximal (cannot add any point)
    3. Minimum size among such sets.
    
    If we remove a point p from a maximal set S, S' = S \ {p} is still no-isosceles.
    But S' is not necessarily maximal. We might be able to add points to S'.
    However, if we just return S', it's not a valid solution because it's not maximal.
    So we must re-maximize S'.
    
    Strategy:
    1. Remove a point p.
    2. Check if S' is no-isosceles (it is).
    3. Re-maximize S' (greedily add points until no more can be added).
    4. If the resulting set is smaller than S, keep it.
    
    This is expensive.
    Alternative:
    Try to swap points?
    
    Let's try the "Remove and Re-maximize" approach.
    """
    current_S = list(S_indices)
    current_size = len(current_S)
    
    for _ in range(max_iters):
        if not current_S:
            break
            
        # Try removing each point
        removed = False
        for i in range(len(current_S)):
            p_idx = current_S[i]
            
            # Remove p_idx
            temp_S = [x for j, x in enumerate(current_S) if j != i]
            
            # Re-maximize temp_S
            # We need to find if we can add any points to temp_S.
            # If we can't add any points, it's maximal.
            # If we can add points, we add them greedily.
            
            # Check if temp_S is maximal.
            # A set is maximal if no point in ALL_POINTS \ temp_S can be added.
            
            # This check is O(N * K^2).
            # N=256, K~10. 256 * 100 = 25600 ops. Fast.
            
            is_maximal = True
            new_points_to_add = []
            
            # To speed up, we can just try to add points.
            # If we find a point we can add, we add it.
            # We want to see if the final size is smaller.
            
            # Actually, if we remove 1 point, and we have to add >= 1 point to make it maximal,
            # the size might not decrease.
            
            # Let's just greedily maximize temp_S and see the size.
            
            S_max = list(temp_S)
            addable = set(ALL_POINTS_indices) - set(S_max)
            
            # Greedy addition
            while True:
                added_any = False
                # Shuffle to be deterministic-ish or random?
                # Let's use the rng
                candidates = list(addable)
                rng.shuffle(candidates)
                
                for q in candidates:
                    if _can_add_indices(q, S_max):
                        S_max.append(q)
                        addable.discard(q)
                        added_any = True
                        # Update addable? 
                        # For efficiency in this inner loop, we can just re-filter or break early?
                        # If we add one, we should check others.
                        # But to keep it simple and fast, let's just do one pass?
                        # No, we need to fully maximize.
                        break # Inner break, restart outer while? 
                        # This is inefficient.
                
                if not added_any:
                    break
                    
            # Now S_max is maximal.
            if len(S_max) < current_size:
                current_S = S_max
                current_size = len(S_max)
                removed = True
                # Restart loop with new smaller set
                break
                
        if not removed:
            break
            
    return current_S

# Helper for indices
ALL_POINTS_indices = list(range(NUM_POINTS))

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    
    # Run multiple greedy starts
    attempts = 50
    
    for seed in range(attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        S_indices = _build_greedy_fast(order_seed=seed)
        
        # Local search
        rng = random.Random(seed)
        S_indices = _local_search(S_indices, rng, max_iters=20)
        
        if len(S_indices) < best_size:
            best_size = len(S_indices)
            best_set = S_indices
            
    # Convert indices back to points
    if best_set is None:
        # Fallback
        best_set = [0]
        
    return [tuple(ALL_POINTS[i]) for i in best_set]

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
