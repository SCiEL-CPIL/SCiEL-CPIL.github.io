import time
import numpy as np
import itertools
import random

def _greedy_cover(balls, cover_count, current_uncovered, deadline, margin=5.0):
    selected = []
    while np.any(current_uncovered) and time.time() < deadline - margin:
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    return selected, current_uncovered

def _prune_code(selected, balls, cover_count, deadline, margin=5.0):
    changed = True
    while changed and time.time() < deadline - margin:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    return selected

def _spectral_core(balls, total, ball_size, deadline, margin=10.0):
    """Spectral clustering on the overlap graph. Returns a boolean mask of the core."""
    # Overlap graph: A[i,j] = |ball[i] ∩ ball[j]| - 1 (exclude self-overlap of 1)
    # ball[i] is a (ball_size,) vector of indices.
    # A[i,j] = sum over positions of (ball[i] == ball[j]) - 1
    # We can compute this via: for each word v, the set of words that contain v in their ball.
    # coverers[v] = {w : v in ball[w]} = ball[v] (symmetric).
    # So A[i,j] = |{v : v in ball[i] and v in ball[j]}| - 1
    # = sum_v [v in ball[i]] [v in ball[j]] - 1
    # Let B be the (total, total) incidence matrix where B[w, v] = 1 if v in ball[w].
    # Then A = B @ B.T - I.
    # But B is huge (6561 x 6561). Instead, use the balls array.
    # B[w, v] = 1 if v in balls[w]. We can build B as a sparse-like dense array.
    # Actually, we can compute A more directly using the structure.
    
    # Simpler: build the overlap via matrix multiplication on a sparse representation.
    # Let's build B as a dense boolean array (total x total) - 6561^2 = ~43M booleans = ~43MB. OK.
    B = np.zeros((total, total), dtype=np.uint8)
    for w in range(total):
        B[w, balls[w]] = 1
    
    # A = B @ B.T - I  (as float for eigendecomposition)
    # B @ B.T is (total, total). 
    # This is a 6561x6561 matmul of uint8 -> could be slow. Let's use the structure.
    # Actually, B is very sparse (17 ones per row). B @ B.T can be computed as:
    # (B @ B.T)[i,j] = sum_v B[i,v] B[j,v] = number of common neighbors.
    # We can compute this via: for each v, the column B[:,v] has ball_size ones.
    # A = sum_v outer(B[:,v], B[:,v]) - I
    # That's total outer products of size ball_size x ball_size. total = 6561, each outer is 6561x6561.
    # Too slow.
    
    # Alternative: just do the matmul with numpy on the uint8 array.
    # B is 6561x6561 uint8. B.astype(float32) @ B.T.astype(float32) is a 6561x6561 float32 matmul.
    # That's ~6561^3 * 2 flops = ~566 GFLOPs. Too slow for 120s.
    
    # Let's use a different approach: compute the overlap via the balls array directly.
    # For each pair (i,j), overlap = sum over k in balls[i] of (k in balls[j]).
    # We can vectorize: for each i, overlap[i,:] = sum over k in balls[i] of B[k, :].
    # That's: overlap = sum_{k in balls[i]} B[k,:]. 
    # We can do: overlap[i] = B[balls[i], :].sum(axis=0). This is ball_size rows summed.
    # For all i: this is total * ball_size row-sums. 
    # overlap = np.zeros((total, total), dtype=np.int32)
    # for i in range(total):
    #     overlap[i] = B[balls[i], :].sum(axis=0)
    # This is total iterations, each summing ball_size rows of length total. 
    # total * ball_size * total = 6561 * 17 * 6561 = ~734M operations. Doable in numpy if vectorized.
    
    # Actually, we can do it in one shot:
    # B_expanded = B[balls, :]  # shape (total, ball_size, total)
    # overlap = B_expanded.sum(axis=1)  # shape (total, total)
    # B_expanded is (6561, 17, 6561) uint8 = 6561*17*6561 = 734M bytes = 734MB. Too much.
    
    # Let's just do the loop but with numpy vectorization per row.
    overlap = np.zeros((total, total), dtype=np.int32)
    for i in range(total):
        if time.time() > deadline - margin:
            break
        overlap[i] = B[balls[i], :].sum(axis=0)
    
    # Now subtract identity (self-overlap)
    idx = np.arange(total)
    overlap[idx, idx] -= 1  # remove the self term (each ball contains itself)
    # Ensure non-negative
    overlap = np.maximum(overlap, 0)
    
    # Degree
    degree = overlap.sum(axis=1).astype(np.float64)
    
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    # We want the dominant eigenvector (smallest eigenvalue of L, which is 1 for connected graph).
    # Actually, for spectral clustering, we want the eigenvector corresponding to the second smallest
    # eigenvalue of L (Fiedler vector). But for finding a "core", we can use the dominant eigenvector
    # of the normalized adjacency matrix D^(-1/2) A D^(-1/2), which corresponds to the smallest
    # eigenvalue of L.
    
    # Let's use the normalized adjacency matrix M = D^(-1/2) A D^(-1/2).
    # The dominant eigenvector of M gives a "centrality" measure.
    
    inv_sqrt_d = np.zeros(total, dtype=np.float64)
    nonzero = degree > 0
    inv_sqrt_d[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    
    # M = D^(-1/2) A D^(-1/2)
    # M[i,j] = A[i,j] / sqrt(d[i] d[j])
    # We can compute this as: M = (A * inv_sqrt_d[:,None]) * inv_sqrt_d[None,:]
    M = (overlap.astype(np.float64) * inv_sqrt_d[:, None]) * inv_sqrt_d[None, :]
    
    # Find dominant eigenvector using power iteration (M is symmetric, so this works).
    # Or use numpy's eigvalsh on the small matrix? 6561x6561 eig is too slow.
    # Power iteration:
    v = np.ones(total, dtype=np.float64) / np.sqrt(total)
    for _ in range(50):
        if time.time() > deadline - margin:
            break
        v_new = M @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new /= norm
        # Check convergence
        if np.linalg.norm(v_new - v) < 1e-8:
            v = v_new
            break
        v = v_new
    
    # Threshold: core = words with above-median centrality
    threshold = np.median(v)
    core_mask = v >= threshold
    
    return core_mask, v

def _maximal_independent_set(core_indices, balls, total, deadline, margin=5.0):
    """Greedy maximal independent set on the overlap graph, restricted to core."""
    # Two words are adjacent if their balls overlap (which is always true for distance <= 2).
    # Actually, in the overlap graph, i~j iff |ball[i] ∩ ball[j]| > 1.
    # For radius-1 balls in Hamming space, |ball[i] ∩ ball[j]| = 1 + 2*(number of positions where i and j differ by 0 or ... )
    # Actually, |ball[i] ∩ ball[j]| depends on dist(i,j):
    # dist 0: 17
    # dist 1: 1 + 2*0 + ... = let's think. ball[i] contains i and all words at dist 1 from i.
    # ball[j] contains j and all words at dist 1 from j.
    # Intersection: words within dist 1 of both i and j.
    # If dist(i,j) = d, then:
    #   d=0: 17
    #   d=1: i and j are adjacent. Common neighbors: i is in ball[j], j is in ball[i]. 
    #         Words at dist 1 from both: a word w at dist 1 from i and dist 1 from j.
    #         If w = i, dist to j is 1. If w = j, dist to i is 1.
    #         Other words: w differs from i in one position and from j in one position.
    #         Since i and j differ in one position (say pos p), w must differ from i in some pos and from j in some pos.
    #         If w differs from i in pos p, then w = j, already counted.
    #         If w differs from i in pos q ≠ p, then w differs from j in pos q (since j and i agree on q, and w differs from i on q, so w differs from j on q). 
    #         So w is at dist 1 from j. There are 2*(n-1) such words.
    #         So |intersection| = 2 (i and j) + 2*(n-1) = 2 + 14 = 16 for n=8.
    #   d=2: i and j differ in 2 positions. Common neighbors: words at dist 1 from both.
    #         w must fix one of the two differences to be at dist 1 from i, and one to be at dist 1 from j.
    #         Actually, if i and j differ in positions p and q, then a word w at dist 1 from i and dist 1 from j must change exactly one of {p,q} to match the other.
    #         So w = i with p changed to j's value (then dist to j is 1, since q still differs). Similarly for q.
    #         So 2 common neighbors. |intersection| = 2.
    #   d=3: no common neighbors (triangle inequality: dist(i,j) <= dist(i,w) + dist(w,j) = 2 < 3, impossible).
    # So overlap graph edges: dist 0 (self, 17), dist 1 (16), dist 2 (2).
    # For independent set, we want words with no overlap, i.e., dist >= 3.
    # So a maximal independent set in the overlap graph = a code with minimum distance 3.
    # But we don't need minimum distance 3; we just need a good starting point.
    
    # Greedy: pick words in order of centrality, add if not overlapping with any already picked.
    # "Overlapping" here means dist < 3 (i.e., dist 0, 1, or 2).
    # Actually, for the independent set on the overlap graph, two words are adjacent if their balls overlap,
    # which happens iff dist <= 2. So independent set = set of words pairwise at dist >= 3.
    
    # But we don't need a strict independent set; we just want a diverse core.
    # Let's just pick words greedily by centrality, ensuring they're not too close.
    
    selected = []
    selected_set = set()
    
    for c in core_indices:
        if time.time() > deadline - margin:
            break
        if c in selected_set:
            continue
        # Check if c is at dist >= 3 from all selected words
        # For efficiency, we can precompute distances, but for now just check.
        # Actually, let's just add it; the refinement will handle overlaps.
        # A strict independent set might be too small. Let's be relaxed.
        selected.append(c)
        selected_set.add(c)
    
    return selected

def _ga_improve(selected, balls, cover_count, deadline, margin=5.0):
    total = len(cover_count)
    best_selected = list(selected)
    best_cover = cover_count.copy()
    best_size = len(best_selected)
    best_uncovered = int(np.sum(best_cover == 0))
    
    rng = random.Random(42)
    
    def compute_cover(code):
        cov = np.zeros(total, dtype=np.int32)
        for c in code:
            cov[balls[c]] += 1
        return cov
    
    cur_selected = list(best_selected)
    cur_cover = best_cover.copy()
    cur_size = len(cur_selected)
    cur_set = set(cur_selected)
    
    improved = True
    while improved and time.time() < deadline - margin - 5.0:
        improved = False
        
        for i in range(len(cur_selected) - 1, -1, -1):
            if time.time() > deadline - margin - 5.0:
                break
            c = cur_selected[i]
            if np.min(cur_cover[balls[c]]) > 1:
                cur_cover[balls[c]] -= 1
                cur_selected.pop(i)
                cur_set.discard(c)
                cur_size -= 1
                improved = True
        
        if not improved and cur_size > 1:
            max_swap_attempts = 50
            for _ in range(max_swap_attempts):
                if time.time() > deadline - margin - 5.0:
                    break
                if not cur_selected:
                    break
                rm_idx = rng.randint(0, len(cur_selected) - 1)
                rm_c = cur_selected[rm_idx]
                
                cur_cover[balls[rm_c]] -= 1
                cur_selected.pop(rm_idx)
                cur_set.discard(rm_c)
                cur_size -= 1
                
                uncovered = (cur_cover == 0)
                if np.any(uncovered):
                    unc_indices = np.where(uncovered)[0]
                    if len(unc_indices) > 0:
                        sample_size = min(200, total)
                        candidates = rng.sample(range(total), sample_size)
                        best_add = -1
                        best_gain = 0
                        for cand in candidates:
                            if cand in cur_set:
                                continue
                            gain = int(np.sum(uncovered[balls[cand]]))
                            if gain > best_gain:
                                best_gain = gain
                                best_add = cand
                        
                        if best_add >= 0 and best_gain >= 2:
                            cur_cover[balls[best_add]] += 1
                            cur_selected.append(best_add)
                            cur_set.add(best_add)
                            cur_size += 1
                            improved = True
                        else:
                            cur_cover[balls[rm_c]] += 1
                            cur_selected.insert(rm_idx, rm_c)
                            cur_set.add(rm_c)
                            cur_size += 1
                    else:
                        cur_cover[balls[rm_c]] += 1
                        cur_selected.insert(rm_idx, rm_c)
                        cur_set.add(rm_c)
                        cur_size += 1
                else:
                    improved = True
        
        if int(np.sum(cur_cover == 0)) == 0 and cur_size < best_size:
            best_size = cur_size
            best_selected = list(cur_selected)
            best_cover = cur_cover.copy()
            best_uncovered = 0
    
    population_size = 15
    max_generations = 30
    
    population = [list(best_selected)]
    
    for _ in range(population_size - 1):
        pop = list(best_selected)
        k = rng.randint(2, min(8, max(1, len(pop) // 15)))
        if len(pop) > k:
            to_remove = rng.sample(pop, k)
            to_remove_set = set(to_remove)
            pop = [c for c in pop if c not in to_remove_set]
            temp_cover = compute_cover(pop)
            uncovered = (temp_cover == 0)
            _, uncovered = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin + 5)
            pop = _prune_code(pop, balls, temp_cover, deadline, margin=margin + 5)
        population.append(pop)
    
    for gen in range(max_generations):
        if time.time() > deadline - margin:
            break
        
        scored_pop = []
        for code in population:
            cover = compute_cover(code)
            unc = int(np.sum(cover == 0))
            if unc > 0:
                f = len(code) + unc * 50
            else:
                f = len(code)
            scored_pop.append((f, unc, code, cover))
        
        scored_pop.sort(key=lambda x: (x[0], x[1]))
        
        best_f, best_unc, best_code, best_cov = scored_pop[0]
        if best_f < best_size or (best_f == best_size and best_unc < best_uncovered):
            best_size = best_f
            best_uncovered = best_unc
            best_selected = list(best_code)
            best_cover = best_cov.copy()
        
        def select():
            idxs = rng.sample(range(len(scored_pop)), min(3, len(scored_pop)))
            best_idx = min(idxs, key=lambda i: scored_pop[i][0])
            return scored_pop[best_idx][2]
        
        new_population = []
        for i in range(min(3, len(scored_pop))):
            new_population.append(list(scored_pop[i][2]))
        
        while len(new_population) < population_size:
            if time.time() > deadline - margin:
                break
            p1 = select()
            p2 = select()
            
            set1 = set(p1)
            set2 = set(p2)
            union = set1 | set2
            child = [c for c in union if rng.random() < 0.6]
            
            if child and rng.random() < 0.7:
                rm = rng.choice(child)
                child.remove(rm)
            if rng.random() < 0.7:
                if rng.random() < 0.5 and union:
                    add = rng.choice(list(union))
                else:
                    add = rng.randint(0, total - 1)
                if add not in child:
                    child.append(add)
            
            temp_cover = compute_cover(child)
            uncovered = (temp_cover == 0)
            _, uncovered = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin + 10)
            if not uncovered.any():
                child = _prune_code(child, balls, temp_cover, deadline, margin=margin + 10)
            
            new_population.append(child)
        
        population = new_population
    
    return best_selected, best_cover

def _cur_uncovered_check(cover):
    return int(np.sum(cover == 0))

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline, start_time)
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline, start_time):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # 1. Spectral core identification
    core_mask, centrality = _spectral_core(balls, total, ball_size, deadline, margin=45.0)
    core_indices = np.where(core_mask)[0]
    
    # 2. Greedy independent set on core
    seed_selected = _maximal_independent_set(core_indices, balls, total, deadline, margin=40.0)
    
    # 3. Greedy cover to complete
    cover_count = np.zeros(total, dtype=np.int32)
    selected = list(seed_selected)
    for c in selected:
        cover_count[balls[c]] += 1
    current_uncovered = (cover_count == 0)
    
    while np.any(current_uncovered) and time.time() < deadline - 40.0:
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 4. Aggressive pruning
    prune_deadline = deadline - 25.0
    for pass_num in range(6):
        if time.time() > prune_deadline:
            break
        changed = True
        while changed and time.time() < prune_deadline:
            changed = False
            new_selected = []
            if pass_num % 3 == 0:
                order = selected
            elif pass_num % 3 == 1:
                order = list(reversed(selected))
            else:
                order = list(selected)
                random.Random(pass_num * 7 + 1).shuffle(order)
            
            for c in order:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
    
    # 5. Local search + GA
    if time.time() < deadline - 12.0:
        selected, cover_count = _ga_improve(selected, balls, cover_count, deadline, margin=12.0)
    
    # 6. Final pruning
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
    
    # 7. Verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1
    
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
