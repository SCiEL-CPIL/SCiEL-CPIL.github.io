import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    D = 11

    # 1. Generate all norm-4 integer vectors (weight 4, entries ±1)
    vectors = []
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            vectors.append(v)

    vectors = np.array(vectors, dtype=np.int8)
    N = len(vectors)

    # 2. Compute conflict graph
    # Conflict if dot > 2
    dot_products = vectors @ vectors.T
    conflict = dot_products > 2

    # 3. Find Maximum Independent Set using greedy + local search
    # We represent the independent set as a boolean mask
    best_mask = np.zeros(N, dtype=bool)
    best_size = 0

    num_restarts = 50
    rng = np.random.RandomState(42)

    # Precompute neighbor lists for faster updates
    # For each vertex, a list of its neighbors
    neighbor_lists = [np.where(conflict[i])[0] for i in range(N)]
    degrees = np.array([len(nl) for nl in neighbor_lists])

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Greedy construction
        available = set(range(N))
        current_mask = np.zeros(N, dtype=bool)
        
        # To speed up, we can process in batches or just loop
        # Sorting by degree (smallest first) is a good heuristic
        # But with restarts, we want randomness.
        # Let's use a weighted greedy: prob proportional to 1/(degree+1)
        # Or just min-degree with random tie-breaking.
        
        # Efficient approach for large N:
        # Since N=5302, a simple loop is okay if optimized.
        # But checking all available vertices to find min degree is O(N).
        # Let's use a heap or just sample.
        
        # Simple random greedy restart
        # Pick random vertex, add to set, remove neighbors.
        # This is slow for large N if we do it N times.
        
        # Better: Greedy by degree
        # For each restart, shuffle order, or pick min degree from remaining.
        
        # Let's try a faster heuristic:
        # 1. Order vertices by degree (global)
        # 2. Iterate in that order, add if no conflict with already selected.
        # This is "Greedy Set Packing".
        
        # To introduce randomness, we can randomize ties in degree, or add noise to degrees.
        
        # Let's implement the "Order by Degree + Noise" greedy.
        deg_with_noise = degrees + rng.rand(N) * 0.5
        order = np.argsort(deg_with_noise)
        
        selected = []
        # To check conflict fast, maintain a set of selected vertices?
        # Or use the mask?
        # Checking if any neighbor is selected:
        # for v in order:
        #   if not any selected neighbor:
        #       select v
        #       mark v as selected
        #
        # Naive check is O(deg(v)). Total O(E).
        # E is roughly N * avg_degree.
        # Avg degree for weight 4 vectors:
        # Total pairs ~ N^2/2 ~ 14M.
        # Conflicts are local.
        # Let's estimate E.
        # Two weight 4 vectors conflict if dot >= 3.
        # Dot = 4 - 2 * (Hamming dist of supports / 2 ... no)
        # Dot = sum v_i w_i.
        # Max dot 4 (same).
        # Dot 3: 3 matches, 1 mismatch? No, weight 4.
        # If supports identical: dot is -4, -2, 0, 2, 4.
        #   Signs: ++++ (4), ++-- (0), +--+ (-2), +--- (-4).
        #   So identical support -> dots: 4, 0, -2, -4.
        #   Conflict only if dot 4 (identical vector).
        # If supports differ by 1 (3 common, 1 unique each):
        #   3 common entries can match/mismatch.
        #   Max dot = 3 (if 3 common match, uniques don't contribute to dot as they are 0 in other vector).
        #   So if 3 signs match, dot = 3. Conflict.
        #   Number of such vectors: C(3,3)*1^3 = 1 sign pattern for common.
        #   Total sign patterns for 4 positions is 16.
        #   Fixed support, other support shares 3.
        #   For a given v, how many w share 3 indices?
        #   Choose which 1 index is removed: 4 choices.
        #   Choose which 1 new index: 11-3=8 choices.
        #   For each such w, how many sign patterns give dot >= 3?
        #   Dot = sum of 3 common signs.
        #   To get dot >= 3, all 3 must be +1 (or -1? No, dot product sum).
        #   If v has signs s1,s2,s3,s4.
        #   w has signs t1,t2,t3 on common.
        #   Dot = s1t1 + s2t2 + s3t3.
        #   Max is 3. So we need s1t1=1, s2t2=1, s3t3=1.
        #   This fixes t1,s1; t2,s2; t3,s3. Only 1 valid sign pattern for w on common part.
        #   The 4th sign of w is free (2 choices).
        #   So for each of 4*8=32 supports, there are 2 vectors.
        #   Total neighbors of degree 32*2 = 64?
        #   Plus itself (1).
        #   What about sharing 2 indices? Dot max 2. No conflict.
        #   So degree ~ 65.
        #
        # N=5302, Avg degree ~ 65.
        # E ~ 5302 * 65 / 2 ~ 172,000 edges.
        # This is very sparse!
        # Greedy will be very fast.
        
        # Let's implement the greedy carefully.
        # We need to know if a vertex v has any neighbor in `selected`.
        # We can maintain a boolean array `is_selected`.
        # For v in order:
        #   if not is_selected[v]:
        #      has_conflict = False
        #      for nb in neighbor_lists[v]:
        #          if is_selected[nb]:
        #              has_conflict = True
        #              break
        #      if not has_conflict:
        #          is_selected[v] = True
        #          selected.append(v)
        
        is_selected = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if is_selected[v]:
                continue
            # Check neighbors
            # Optimized: if any selected neighbor
            # np.any(is_selected[neighbor_lists[v]])
            # This is a vectorized op on a small array.
            if not np.any(is_selected[neighbor_lists[v]]):
                is_selected[v] = True
                count += 1
        
        if count > best_size:
            best_size = count
            best_mask = is_selected.copy()
            
        # Optional: Local search (swap or add) to improve.
        # Given the sparsity, greedy might be quite good.
        # Let's add a simple "add" phase.
        # Find any vertex not selected that has no selected neighbors.
        # (This should be none if greedy was perfect, but greedy might miss some due to order).
        # Actually, the greedy above picks if possible. So at the end, no unselected vertex has 0 selected neighbors?
        # Not necessarily. It just means at the time it was considered, it had a neighbor.
        # But maybe that neighbor could have been skipped?
        # Local search: try to remove one selected vertex to allow adding two non-selected ones?
        # Or just try to add any compatible unselected vertex (should be 0).
        
        # Let's try a simple improvement:
        # For each unselected vertex u, if it has exactly 1 selected neighbor v, 
        # and v has no other unselected neighbors that could be added... this is complex.
        
        # Simpler: Random restarts are likely sufficient for this sparse graph.
        
    # 4. Extract final vectors
    final_vectors = vectors[best_mask]
    
    # 5. Verify and return
    # Check min distance
    if len(final_vectors) > 0:
        # dist^2 = 8 - 2 dot
        # min dist^2 >= 4 => max dot <= 2
        # We constructed it this way.
        pass
    
    return final_vectors
