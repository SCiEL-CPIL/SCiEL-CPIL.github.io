# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        mask |= (1 << i)
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Tetracode direct sum initialization:
    # Tetracode is a perfect [4,1,3] ternary code (13 words).
    # Direct sum with {0,1,2} in positions 4,5 gives 13*3 = 39 words.
    # Each codeword covers 19 words, total coverage = 39*19 = 741 > 729.
    # This provides a much denser starting point than pure greedy.
    def tetracode_init(seed=0):
        rng = random.Random(seed)
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        # Tetracode: 13 words of length 4 over Z_3
        # Constructed as: {0000, 1111, 2222} union {vectors with 3 non-zero coords}
        tetracode_words = []
        # The 3 zero-weight-4 words
        tetracode_words.append((0,0,0,0))
        tetracode_words.append((1,1,1,1))
        tetracode_words.append((2,2,2,2))
        # 10 words with exactly 3 non-zero coordinates (sum constraint for tetracode)
        # Tetracode = {x in Z_3^4 : x_1+x_2+x_3+x_4 = 0, and weight <= 3}
        # Actually the standard tetracode has 13 codewords. Let me use the known list:
        # 0000, 1111, 2222,
        # 1100, 2200, 0011, 0022,
        # 1010, 2020, 0101, 0202
        # Wait, let me just verify: tetracode is the unique [4,1,3]_3 code.
        # It has 3^1 = 3... no wait, it has 13 words.
        # Actually [4,1,3] means dimension 1, so 3 codewords. That's not right.
        # Tetracode is a nonlinear code with 13 words. Let me use the correct construction.
        # The tetracode consists of:
        # 0000, 1111, 2222 (3 words)
        # and 10 words of weight 3 or 4... 
        # Actually, the tetracode is: all vectors in Z_3^4 with x1+x2+x3+x4=0
        # that have no coordinate equal to 0? No.
        # Let me just use a known valid tetracode:
        # The tetracode has 13 words. A standard representation:
        tetracode_words = [
            (0,0,0,0),
            (1,1,1,1),
            (2,2,2,2),
            (1,1,0,0),
            (2,2,0,0),
            (0,0,1,1),
            (0,0,2,2),
            (1,0,1,0),
            (2,0,2,0),
            (0,1,0,1),
            (0,2,0,2),
            (1,1,2,2),
            (2,2,1,1),
        ]
        # Verify: 13 words. Check distances: min distance should be 3.
        # (0,0,0,0) to (1,1,1,1): dist 4 ✓
        # (0,0,0,0) to (1,1,0,0): dist 2 ✗ -- this is wrong!
        # Let me use the actual tetracode. The tetracode is a [4,1,3] code meaning
        # it's a linear code of dimension 1... but that only has 3 words.
        # I'm confusing myself. Let me just use a known covering code approach.
        
        # Actually, for the direct sum approach, I just need ANY 13-word set of length-4
        # words that covers all of Z_3^4 with radius 1. The perfect ternary code of length 4
        # doesn't exist (needs (3^4)/13 ≈ 19.85, not integer). 
        # So instead, let me use a good 13-word covering of Z_3^4 with radius 1.
        # Each word covers 1+4*2=9 words. 13*9=117 >= 81. Good.
        
        # Let me just use a greedy construction for the tetracode part, then extend.
        # Actually, simpler: just do the full 6-position greedy but with a smarter init.
        
        # Better approach: use a structured init based on known good covering patterns.
        # Start with all words where positions 0-2 form a covering of Z_3^3 (radius 1)
        # and positions 3-5 are fixed. Z_3^3 needs ceil(27/7)=4 words to cover with r=1
        # (each covers 1+3*2=7 words). 4*7=28>=27.
        
        # Let me use: for the 6-position problem, start with a grid-like structure.
        # Use words where (x0+x1+x2) mod 3 == (x3+x4+x5) mod 3 as a base,
        # then add/remove to optimize.
        
        # Simplest strong init: greedy but starting from a structured seed.
        # Use the "all zeros" plus structured perturbations as initial code.
        
        # Actually the simplest strong approach: start with ALL 27 words of form
        # (a,b,c,0,0,0) where (a,b,c) covers Z_3^3 with radius 1 (need 4 words),
        # then extend to cover the rest. But that's just greedy with a head start.
        
        # Let me just do a fast greedy with random tie-breaking, which is what
        # the existing code does. The key improvement is in the local search speed.
        return code_set, cover_count, uncovered_mask
    
    def tetracode_direct_sum_init(seed=0):
        """
        Initialize using a direct-sum-like structure.
        We know K_3(4,1) = 13 (the tetracode). For length 6, we can use
        a product-like structure: take a covering of Z_3^3 with radius 1
        (K_3(3,1) = 4) and combine with another covering.
        
        Actually, the cleanest approach: start with a random but structured
        set of ~40-50 words that already covers most of the space, then
        optimize.
        """
        rng = random.Random(seed)
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        # Generate a structured initial code:
        # Use a 3D grid: for each (a,b) in Z_3 x Z_3, pick a word
        # (a, b, c(a,b), d(a,b), e(a,b), f(a,b)) that maximizes coverage.
        # This gives 9 words initially, then fill in.
        
        # Better: use a known good structure.
        # Start with words of the form (i, i, i, j, j, j) for i,j in {0,1,2}
        # That's 9 words. Each covers 19 words. 9*19 = 174. Not enough.
        
        # Let me just do a fast multi-start greedy and take the best.
        # The real improvement is in local search speed.
        return code_set, cover_count, uncovered_mask
    
    # Perturbation vectors in Z_3^6
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)
    
    # 3-coordinate perturbations for larger moves
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for pos3 in range(pos2 + 1, n):
                for d1 in (1, 2):
                    for d2 in (1, 2):
                        p = [0] * n
                        p[pos1] = d1
                        p[pos2] = d2
                        p[pos3] = 1  # fixed for 3-coord
                        perturbations.append(p)
    
    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    def greedy_start(seed=42):
        rng = random.Random(seed)
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        while uncovered_mask and time.time() < deadline - 15:
            best_idx = -1
            best_cnt = -1
            candidates = []
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    candidates = [i]
                elif cnt == best_cnt:
                    candidates.append(i)
            
            if not candidates:
                break
            best_idx = rng.choice(candidates)
            code_set.add(best_idx)
            m = neighbors[best_idx]
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~m
            
        return code_set, cover_count, uncovered_mask

    def linear_start(seed=42):
        """Generate a code from a structured initial point + greedy completion."""
        rng = random.Random(seed)
        
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        # Structured initialization: use a pattern based on modular arithmetic.
        # Start with words where x0+x1+x2+x3+x4+x5 ≡ 0 (mod 3) -- a hyperplane.
        # This gives 243 words, way too many. Instead, use a sparser structure.
        
        # Use a "staircase" pattern: for each (a,b) in Z_3^2, add word
        # (a, b, (a+b)%3, (a-b)%3, a, b) -- 9 words
        # Then fill in greedily.
        
        # Actually, let's use a random affine subspace of dimension 2 (9 words)
        # plus a random offset, then complete greedily.
        
        # Pick a random 2D affine subspace: {p + s*v1 + t*v2 : s,t in Z_3}
        p = tuple(rng.randint(0, 2) for _ in range(n))
        v1 = tuple(rng.randint(0, 2) for _ in range(n))
        v2 = tuple(rng.randint(0, 2) for _ in range(n))
        
        for s in range(3):
            for t in range(3):
                w = tuple((p[i] + s*v1[i] + t*v2[i]) % 3 for i in range(n))
                idx = idx_map[w]
                if idx not in code_set:
                    code_set.add(idx)
                    for idx2 in neighbor_indices[idx]:
                        cover_count[idx2] += 1
                    uncovered_mask &= ~neighbors[idx]
        
        # Now greedily add more to cover remaining
        while uncovered_mask and time.time() < deadline - 15:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count, max_passes=10):
        changed = True
        passes = 0
        while changed and passes < max_passes and time.time() < deadline - 3:
            changed = False
            passes += 1
            for c in list(code_set):
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        while uncovered_mask and time.time() < deadline - 5:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    def try_1exchange(code_set, cover_count, c, new_idx):
        if new_idx in code_set:
            return False, cover_count
        
        # Fast validity check using bitmasks:
        # After removing c and adding new_idx, a word i is uncovered iff
        # cover_count[i] == 0. We can check this via:
        # words_only_covered_by_c = {i : cover_count[i]==1 and i in neighbors[c]}
        # These must all be covered by new_idx.
        
        # Compute the set of words that would become uncovered
        # = (neighbors[c] where cover_count==1) AND NOT neighbors[new_idx]
        uncovered_after = 0
        for idx in neighbor_indices[c]:
            if cover_count[idx] == 1:
                # This word is only covered by c; check if new_idx covers it
                if not (neighbors[new_idx] >> idx) & 1:
                    uncovered_after |= (1 << idx)
        
        if uncovered_after == 0:
            # Valid exchange
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            code_set.remove(c)
            code_set.add(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] += 1
            return True, cover_count
        else:
            return False, cover_count

    def local_search_1exchange(code_set, cover_count, rng, max_iters=100):
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx in code_set:
                continue
            
            success, cover_count = try_1exchange(code_set, cover_count, c, new_idx)
            if success:
                improved = True
        
        return code_set, cover_count, improved

    def local_search_2exchange(code_set, cover_count, rng, max_iters=50):
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(code_set) < 2:
                break
                
            c1, c2 = rng.sample(list(code_set), 2)
            
            c1_tuple = all_tuples[c1]
            c2_tuple = all_tuples[c2]
            
            found = False
            for attempt in range(5):
                d1 = rng.choice(perturbations)
                d2 = rng.choice(perturbations)
                new1_tuple = tuple((c1_tuple[i] + d1[i]) % q for i in range(n))
                new2_tuple = tuple((c2_tuple[i] + d2[i]) % q for i in range(n))
                new1_idx = idx_map[new1_tuple]
                new2_idx = idx_map[new2_tuple]
                
                if new1_idx in code_set or new2_idx in code_set:
                    continue
                if new1_idx == new2_idx:
                    continue
                
                # Fast validity check: words with cover_count==1 that are in
                # neighbors[c1] or neighbors[c2] must be covered by new1 or new2.
                valid = True
                for idx in neighbor_indices[c1]:
                    if cover_count[idx] == 1:
                        if not ((neighbors[new1_idx] >> idx) & 1) and not ((neighbors[new2_idx] >> idx) & 1):
                            valid = False
                            break
                if valid:
                    for idx in neighbor_indices[c2]:
                        if cover_count[idx] == 1:
                            if not ((neighbors[new1_idx] >> idx) & 1) and not ((neighbors[new2_idx] >> idx) & 1):
                                valid = False
                                break
                
                if valid:
                    for idx in neighbor_indices[c1]:
                        cover_count[idx] -= 1
                    for idx in neighbor_indices[c2]:
                        cover_count[idx] -= 1
                    code_set.remove(c1)
                    code_set.remove(c2)
                    
                    code_set.add(new1_idx)
                    code_set.add(new2_idx)
                    for idx in neighbor_indices[new1_idx]:
                        cover_count[idx] += 1
                    for idx in neighbor_indices[new2_idx]:
                        cover_count[idx] += 1
                    
                    improved = True
                    found = True
                    break
            
            if found:
                continue
        
        return code_set, cover_count, improved

    def full_local_search(code_set, cover_count, rng):
        any_improved = True
        while any_improved and time.time() < deadline - 3:
            any_improved = False
            code_set, cover_count, imp1 = local_search_1exchange(code_set, cover_count, rng, max_iters=80)
            if imp1:
                any_improved = True
            code_set, cover_count, imp2 = local_search_2exchange(code_set, cover_count, rng, max_iters=30)
            if imp2:
                any_improved = True
            code_set, cover_count = reduce_code(code_set, cover_count)
        return code_set, cover_count

    # Main ILS Loop
    rng = random.Random(42)
    
    best_size = float('inf')
    best_code = None
    code_set = set()
    cover_count = [0] * total_words
    
    # Multi-start: mix greedy and linear starts
    num_starts = 8
    for s in range(num_starts):
        if time.time() > deadline - 12:
            break
        seed = 42 + s * 1000
        
        if s % 2 == 0:
            code_set, cover_count, uncovered_mask = greedy_start(seed)
        else:
            code_set, cover_count, uncovered_mask = linear_start(seed)
        
        code_set, cover_count = reduce_code(code_set, cover_count)
        code_set, cover_count = full_local_search(code_set, cover_count, rng)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
        
        if best_size <= 73:
            break
    
    # ILS with structured perturbations
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        
        r = rng.random()
        if r < 0.4:
            # Single exchange perturbation
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx not in code_set:
                success, cover_count = try_1exchange(code_set, cover_count, c, new_idx)
                if not success:
                    # Revert: the exchange wasn't applied, so nothing to do
                    pass
        elif r < 0.7:
            # Remove k random words and repair
            k = rng.randint(2, 4)
            current_code = list(code_set)
            if len(current_code) < k:
                k = max(1, len(current_code) // 2)
            to_remove = rng.sample(current_code, k)
            for c in to_remove:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
            uncovered_mask = get_uncovered_mask(cover_count)
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)
        else:
            # Double exchange perturbation
            if len(code_set) >= 2:
                c1, c2 = rng.sample(list(code_set), 2)
                c1_tuple = all_tuples[c1]
                c2_tuple = all_tuples[c2]
                d1 = rng.choice(perturbations)
                d2 = rng.choice(perturbations)
                new1_tuple = tuple((c1_tuple[i] + d1[i]) % q for i in range(n))
                new2_tuple = tuple((c2_tuple[i] + d2[i]) % q for i in range(n))
                new1_idx = idx_map[new1_tuple]
                new2_idx = idx_map[new2_tuple]
                
                if new1_idx not in code_set and new2_idx not in code_set and new1_idx != new2_idx:
                    # Fast validity check
                    valid = True
                    for idx in neighbor_indices[c1]:
                        if cover_count[idx] == 1:
                            if not ((neighbors[new1_idx] >> idx) & 1) and not ((neighbors[new2_idx] >> idx) & 1):
                                valid = False
                                break
                    if valid:
                        for idx in neighbor_indices[c2]:
                            if cover_count[idx] == 1:
                                if not ((neighbors[new1_idx] >> idx) & 1) and not ((neighbors[new2_idx] >> idx) & 1):
                                    valid = False
                                    break
                    if valid:
                        for idx in neighbor_indices[c1]:
                            cover_count[idx] -= 1
                        for idx in neighbor_indices[c2]:
                            cover_count[idx] -= 1
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(new1_idx)
                        code_set.add(new2_idx)
                        for idx in neighbor_indices[new1_idx]:
                            cover_count[idx] += 1
                        for idx in neighbor_indices[new2_idx]:
                            cover_count[idx] += 1
        
        code_set, cover_count = full_local_search(code_set, cover_count, rng)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

    # Final polish
    if best_code is not None and time.time() < deadline - 5:
        code_set = set(best_code)
        cover_count = [0] * total_words
        for c in code_set:
            for idx in neighbor_indices[c]:
                cover_count[idx] += 1
        
        polish_rng = random.Random(12345)
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            code_set, cover_count, imp = local_search_1exchange(code_set, cover_count, polish_rng, max_iters=200)
            if imp:
                code_set, cover_count = reduce_code(code_set, cover_count)
            code_set, cover_count, imp2 = local_search_2exchange(code_set, cover_count, polish_rng, max_iters=80)
            if imp2:
                code_set, cover_count = reduce_code(code_set, cover_count)
            if not imp and not imp2:
                break
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    code_set = set()
    cover_count = [0] * total_words
    uncovered_mask = full_mask
    
    while uncovered_mask and time.time() < deadline - 10:
        best_idx = -1
        best_cnt = -1
        for i in range(total_words):
            if i in code_set:
                continue
            new_covered = neighbors[i] & uncovered_mask
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1: break
        code_set.add(best_idx)
        for idx in neighbor_indices[best_idx]:
            cover_count[idx] += 1
        uncovered_mask &= ~neighbors[best_idx]
        
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            redundant = True
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
                changed = True
                break
                
    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
