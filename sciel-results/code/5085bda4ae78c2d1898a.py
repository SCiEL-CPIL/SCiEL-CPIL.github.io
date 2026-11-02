"""Rewritten program for kissing number in dimension 11."""
import time
import numpy as np

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock budget with safety margin

    # We search for integer vectors v in Z^11 such that:
    #   ||v||^2 = 4  (so ||v|| = 2)
    #   and for any two distinct vectors u, w in the set:
    #       ||u - w||^2 >= 4
    #
    # Since ||u||^2 = ||w||^2 = 4, we have:
    #   ||u - w||^2 = ||u||^2 + ||w||^2 - 2 u·w = 8 - 2 u·w
    # So the constraint ||u - w||^2 >= 4 becomes:
    #   8 - 2 u·w >= 4  =>  u·w <= 2
    #
    # Also, u·w must be an integer (since u, w are integer vectors).
    # And u·w can range from -4 to 4 (by Cauchy-Schwarz, |u·w| <= 4).
    # So the constraint is u·w <= 2.
    #
    # Vectors with norm^2 = 4 in Z^11 are those with:
    #   - Two ±1 entries and rest 0 (norm^2 = 1+1 = 2) — NO, that's norm^2=2
    #   Wait, let me recompute. Norm^2 = 4 means sum of squares = 4.
    #   Possibilities for integer vectors with sum of squares = 4:
    #     a) One entry ±2, rest 0: 2^2 = 4 ✓
    #     b) Two entries ±1, ±1, rest 0: 1+1 = 2 ✗ (that's norm^2=2)
    #     c) Four entries ±1: 1+1+1+1 = 4 ✓
    #     d) One ±2 and ... no, 4+anything > 4 unless rest 0
    #
    # So valid vectors with norm^2 = 4:
    #   Type A: one coordinate is ±2, others 0. Count: 11 * 2 = 22
    #   Type B: four coordinates are ±1, others 0. Count: C(11,4) * 2^4 = 330 * 16 = 5280
    #
    # Total candidate vectors: 22 + 5280 = 5302
    #
    # Now we need to find the maximum subset where every pair has dot product <= 2.
    #
    # Let's check conflicts:
    #   Type A (±2 on coordinate i) vs Type A (±2 on coordinate j):
    #     If i != j: dot = 0 <= 2 ✓
    #     If i == j: dot = (±2)(±2) = ±4. If same sign: 4 > 2 ✗. If opposite: -4 <= 2 ✓
    #     So for each coordinate, we can have at most one of {+2, -2}.
    #
    #   Type A (2 on coord i) vs Type B (±1 on coords including i):
    #     If Type B has +1 on coord i: dot = 2*1 = 2 <= 2 ✓
    #     If Type B has -1 on coord i: dot = 2*(-1) = -2 <= 2 ✓
    #     So no conflict between Type A and Type B!
    #
    #   Type B vs Type B: two vectors with ±1 on 4 coords each.
    #     Dot product = sum over intersection of signs.
    #     If they share k coordinates, dot product is in {-k, -k+2, ..., k}.
    #     We need dot <= 2.
    #     If k <= 2: max dot = 2 <= 2 ✓ (always ok)
    #     If k = 3: dot can be up to 3 > 2 ✗ (conflict if all 3 signs match)
    #     If k = 4: dot can be up to 4 > 2 ✗
    #
    # So the main constraint is between Type B vectors that share 3 or 4 coordinates.

    # Generate all candidate vectors
    candidates = []
    # Type A: one ±2
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))

    # Type B: four ±1
    import itertools
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            candidates.append(tuple(v))

    n = len(candidates)
    # n = 22 + 5280 = 5302

    # Build conflict graph: edge between i, j if dot(candidates[i], candidates[j]) > 2
    # We want maximum independent set.
    
    # For efficiency, precompute dot products only when needed.
    # Use simulated annealing on the independent set problem.
    
    # Convert to numpy for speed
    cand_arr = np.array(candidates, dtype=np.int8)
    
    # Precompute pairwise dot products? 5302^2 / 2 ~ 14M entries, each int8 dot of 11 elements.
    # That might be too slow to precompute all. Let's compute on the fly or in blocks.
    
    # Actually, let's think about this differently. The structure is:
    # - Type A vectors: at most 11 (one per coordinate, choosing sign)
    # - Type B vectors: need to pick a subset with pairwise dot <= 2
    
    # Since Type A doesn't conflict with Type B, we can always include all 11 Type A vectors
    # (choosing one sign per coordinate). Actually we should choose signs optimally, but
    # since they don't conflict with Type B, we just pick one per coordinate.
    
    # Focus on Type B: find max subset of the 5280 Type B vectors with pairwise dot <= 2.
    
    type_b = []
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            type_b.append(tuple(v))
    
    type_b_arr = np.array(type_b, dtype=np.int8)
    nb = len(type_b_arr)  # 5280
    
    # For Type B vectors, two vectors conflict if they share >= 3 coordinates AND the dot on shared coords > 2.
    # Actually simpler: just compute dot product.
    
    # Simulated annealing for maximum independent set on Type B vectors.
    # State: binary vector x of length nb, x[i]=1 means vector i is included.
    # Objective: maximize sum(x) subject to for all i<j with x[i]=x[j]=1: dot(i,j) <= 2.
    
    # This is hard. Let's use a greedy + local search approach instead, which may be faster.
    
    # Greedy: repeatedly pick the vector with fewest conflicts with already-picked vectors.
    
    # First, let's try a simple greedy approach and see what we get.
    
    # For speed, let's work with the structure:
    # Each Type B vector is defined by (coords, signs) where coords is a 4-subset of {0..10}.
    # Two Type B vectors conflict if their coord sets intersect in >= 3 elements AND dot > 2.
    
    # Let's use simulated annealing with a time budget.
    
    rng = np.random.default_rng(42)
    
    # Initialize with greedy
    # For each vector, count how many other vectors it conflicts with.
    # Pick the one with min conflicts, remove it and its neighbors, repeat.
    
    # This is O(n^2) per iteration which is too slow for 5280 vectors.
    
    # Alternative: use the structure. Group Type B vectors by their coordinate set.
    # There are C(11,4) = 330 coordinate sets, each with 16 sign patterns.
    
    # Two vectors with different coordinate sets:
    #   If |intersection| <= 2: never conflict (dot <= 2 always)
    #   If |intersection| = 3: conflict only if dot on shared coords > 2
    #   If |intersection| = 4 (same set): conflict if dot > 2, i.e., they agree on >= 3 of 4 signs
    
    # This is getting complex. Let me just do a direct simulated annealing with vectorized dot products.
    
    # Precompute the full dot product matrix for Type B vectors.
    # 5280 x 5280 matrix of dots. Each entry is sum of 11 products of int8 values.
    # Using numpy: type_b_arr @ type_b_arr.T gives the full matrix.
    
    dot_matrix = type_b_arr @ type_b_arr.T  # shape (5280, 5280)
    # Conflict if dot > 2
    conflict = dot_matrix > 2  # boolean matrix, conflict[i,j] = True if i and j conflict
    
    # Note: diagonal is 4 (norm^2), so conflict[i,i] = True, but we don't care about self.
    np.fill_diagonal(conflict, False)
    
    # Now solve max independent set on this graph using simulated annealing.
    
    # State: binary array x of length nb
    # Start with all zeros, add vectors greedily.
    
    # Greedy initialization
    x = np.zeros(nb, dtype=np.int8)
    remaining = np.ones(nb, dtype=bool)
    
    while remaining.any():
        # Find the vertex with minimum degree among remaining
        degrees = conflict[:, remaining].sum(axis=1)
        remaining_indices = np.where(remaining)[0]
        min_deg_idx = remaining_indices[np.argmin(degrees[remaining_indices])]
        x[min_deg_idx] = 1
        remaining[min_deg_idx] = False
        remaining[conflict[min_deg_idx]] = False
    
    best_x = x.copy()
    best_count = x.sum()
    
    # Simulated annealing to improve
    # Moves: flip a single bit, or swap (remove one, add another)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    current_x = best_x.copy()
    current_count = best_count
    
    # Precompute for fast evaluation
    # When we flip x[i], the change in count is ±1.
    # We need to check if adding i creates conflicts with existing selected vertices.
    
    # For speed, maintain a "conflict count" array: for each unselected vertex,
    # how many selected vertices does it conflict with.
    
    conflict_count = np.zeros(nb, dtype=np.int32)
    for i in range(nb):
        if current_x[i]:
            conflict_count[conflict[i]] += 1
    
    # Actually this is O(n^2) to compute initially. Let's do it properly.
    # conflict_count[j] = sum over i where x[i]=1 and conflict[i,j]=True
    
    # Use matrix multiplication: conflict_count = (conflict @ current_x) where conflict is 0/1
    conflict_int = conflict.astype(np.int32)
    conflict_count = conflict_int @ current_x
    
    elapsed = time.time() - start_time
    max_time = 110.0  # leave 10s margin
    
    while time.time() - start_time < max_time:
        # Try a random move: pick a random vertex to flip
        i = rng.integers(0, nb)
        
        if current_x[i]:
            # Remove i
            new_count = current_count - 1
            # This is always valid (removing can't create conflicts)
            # Accept if new_count >= current_count or with probability exp((new_count - current_count)/T)
            delta = new_count - current_count  # = -1
            if delta >= 0 or rng.random() < np.exp(delta / T):
                current_x[i] = 0
                current_count = new_count
                conflict_count[conflict[i]] -= 1
                if current_count > best_count:
                    best_count = current_count
                    best_x = current_x.copy()
        else:
            # Add i
            # Check if i conflicts with any currently selected vertex
            if conflict_count[i] == 0:
                new_count = current_count + 1
                delta = new_count - current_count  # = +1
                # Always accept improving move
                current_x[i] = 1
                current_count = new_count
                conflict_count[conflict[i]] += 1
                if current_count > best_count:
                    best_count = current_count
                    best_x = current_x.copy()
            else:
                # Conflict exists; accept with probability exp(-conflict_count[i] / T)
                if rng.random() < np.exp(-conflict_count[i] / T):
                    # Accept the conflict temporarily
                    current_x[i] = 1
                    current_count = current_count + 1
                    conflict_count[conflict[i]] += 1
                    if current_count > best_count:
                        # Only update best if actually valid
                        # Check if this is truly conflict-free
                        # For now, skip updating best during conflicts
                        pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final: ensure best_x is valid (no conflicts)
    # Re-verify best_x
    # For safety, do a cleanup pass on best_x
    # Remove any vertex that conflicts with another in best_x
    changed = True
    while changed:
        changed = False
        for i in np.where(best_x)[0]:
            for j in np.where(best_x)[0]:
                if i < j and conflict[i, j]:
                    # Remove the one with higher degree
                    deg_i = conflict[i].sum()
                    deg_j = conflict[j].sum()
                    if deg_i >= deg_j:
                        best_x[i] = 0
                    else:
                        best_x[j] = 0
                    changed = True
                    break
            if changed:
                break
    
    # Combine Type A and Type B results
    # Type A: pick one vector per coordinate (11 vectors)
    type_a_vectors = []
    for i in range(D):
        v = [0] * D
        v[i] = 2
        type_a_vectors.append(v)
    
    # Type B vectors from best_x
    type_b_selected = []
    for i in range(nb):
        if best_x[i]:
            type_b_selected.append(list(type_b[i]))
    
    all_vectors = type_a_vectors + type_b_selected
    result = np.array(all_vectors, dtype=np.int64)
    
    return result


if __name__ == "__main__":
    res = construct()
    print(f"Number of vectors: {len(res)}")
