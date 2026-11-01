import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    # Precompute ball arrays as numpy for faster operations
    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_arrays[best_c]] = True
            remaining[ball_arrays[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        """Remove redundant codewords where all points in ball have count >= 2."""
        changed = False
        for c in list(code_set):
            ball = ball_arrays[c]
            if np.all(cnt[ball] >= 2):
                code_set.remove(c)
                cnt[ball] -= 1
                changed = True
        return changed

    def try_swap(code_set, cnt, c_remove, c_add):
        """Try swapping c_remove for c_add. Returns True if valid."""
        if c_add in code_set or c_add == c_remove:
            return False
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        # Check: after removing c_remove, all points still covered?
        # Points in ball_r must have cnt >= 2 OR be in ball_a
        # Points in ball_a but not ball_r will get +1, so they're fine
        # Points in ball_r but not ball_a need cnt >= 2
        overlap_mask = np.zeros(total, dtype=bool)
        overlap_mask[ball_r] = True
        overlap_mask[ball_a] = False  # ball_a overrides: these points get +1
        # Actually simpler: check points in ball_r that are NOT in ball_a
        in_r_not_a = ball_r[np.isin(ball_r, ball_a, assume_unique=False, invert=True)]
        if len(in_r_not_a) > 0 and np.any(cnt[in_r_not_a] < 2):
            return False
        # Valid: apply swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_r] -= 1
        cnt[ball_a] += 1
        return True

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1

    strategies = [
        "natural",
        "random1",
        "random2",
        "random3",
    ]

    for s in strategies:
        if time.time() > deadline - 10:
            break
        if s == "natural":
            order = list(range(total))
        elif s == "random1":
            order = list(range(total))
            random.shuffle(order)
        elif s == "random2":
            order = list(range(total))
            random.shuffle(order)
        elif s == "random3":
            # Weighted: prefer words that are "central" (high degree in overlap graph)
            # Approximate by random
            order = list(range(total))
            random.shuffle(order)
        
        code = greedy_cover(order)
        code_set = set(code)
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
        
        # Delete redundant
        while delete_redundant(code_set, cnt):
            pass
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()

    if best_code is None:
        # Fallback
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)

    # === Local Search Phase ===
    current_code = set(best_code)
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_arrays[c]] += 1

    best_size = len(current_code)
    best_code = list(current_code)

    # Precompute: for each pair, overlap size (for faster swap decisions)
    # Actually, let's just do random swaps and check

    # Local search loop
    no_improve_streak = 0
    max_streak = 800

    # Precompute pairwise overlap counts for faster swap validation
    # overlap[i][j] = number of points in ball_i ∩ ball_j
    # For q=3, n=7, total=2187, this is a 2187x2187 int8 matrix (~4.8MB)
    # Build efficiently: for each point p, find all codewords whose ball contains p
    # Then for each pair in that list, increment overlap
    # Actually simpler: for each pair (i,j), compute |ball_i ∩ ball_j|
    # Two words in ternary length 7 are at distance 0 or 1 in their ball intersection
    # |ball_i ∩ ball_j| = number of words within dist 1 of both i and j
    # If d(i,j)=0: 15. If d(i,j)=1: words within dist 1 of both = those at dist 0 from i (15)
    #   minus those not within dist 1 of j... Let me just compute via the ball arrays.
    # For 2187^2 pairs this is too slow. Use the structure:
    # For two words w1, w2 at Hamming distance d:
    #   d=0: overlap = 15
    #   d=1: overlap = 1 + (n-1)*(q-1) - ... 
    # Actually: ball(w1) ∩ ball(w2) = {x : d(x,w1)<=1 and d(x,w2)<=1}
    # If d(w1,w2)=1: x must be within 1 of both. The words at distance 0 from w1 that are also
    # within 1 of w2: w1 itself (d=1 from w2, ok) + the n-1 neighbors of w1 not in direction of w2
    #   that are within 1 of w2. Plus w2 itself. Plus neighbors of w2 not in direction of w1.
    # This gets complex. Let me just precompute using numpy for the 2187x2187 case.
    # 2187 balls of size 15. For each ball i, mark its 15 points. For each pair, count overlap.
    # Alternative: build a 2187x2187 boolean matrix by iterating over points.
    # For each point p (2187 of them), find which balls contain it (up to 15 balls).
    # Then for each pair in that set, increment a counter.
    # Total work: 2187 points * C(15,2) pairs = 2187 * 105 = ~230K increments. Fast!
    
    overlap = np.zeros((total, total), dtype=np.int8)
    # For each point, find all balls that contain it
    # ball_indices[p] gives the points in ball p. We need the reverse: for point p, which balls contain it?
    # Actually ball_indices[i] = list of points within dist 1 of word i.
    # So point p is in ball_indices[i] iff d(p, i) <= 1.
    # We can build: for each i, for each p in ball_indices[i], add i to balls_containing[p].
    # But we already have ball_indices. Let's build the reverse mapping.
    balls_containing = [[] for _ in range(total)]
    for i in range(total):
        for p in ball_indices[i]:
            balls_containing[p].append(i)
    
    for p in range(total):
        bc = balls_containing[p]
        for a in range(len(bc)):
            for b in range(a+1, len(bc)):
                i = bc[a]
                j = bc[b]
                overlap[i, j] += 1
                overlap[j, i] += 1
    
    # Now for swap validation: points in ball_r not in ball_a = ball_size - overlap[r][a]
    # We need min cnt over those points. We can't easily get that from overlap alone.
    # But we can check: if overlap[r][a] == ball_size, then ball_r ⊆ ball_a, always valid.
    # Otherwise, we need to check the non-overlapping points.
    # Let's keep the isin check but add a fast path for full overlap.
    
    code_list = list(current_code)
    
    while time.time() < deadline - 5.0 and no_improve_streak < max_streak:
        improved = False

        # 1. Delete phase
        while delete_redundant(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            improved = True

        if not improved:
            no_improve_streak += 1
            continue
        else:
            no_improve_streak = 0

        code_list = list(current_code)
        
        # 2. Informed 1-for-1 swap phase
        swap_attempts = 0
        max_swap_attempts = 300
        while swap_attempts < max_swap_attempts and time.time() < deadline - 5.0:
            swap_attempts += 1
            if len(current_code) == 0:
                break

            c_remove = random.choice(code_list)
            if c_remove not in current_code:
                continue
            c_add = random.randint(0, total - 1)

            if c_add == c_remove or c_add in current_code:
                continue

            ov = overlap[c_remove, c_add]
            if ov >= ball_size:
                # ball_r fully contained in ball_a: always valid
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_arrays[c_remove]] -= 1
                cnt[ball_arrays[c_add]] += 1
                improved = True
            else:
                # Need to check non-overlapping points
                ball_r = ball_arrays[c_remove]
                ball_a = ball_arrays[c_add]
                mask_in_a = np.isin(ball_r, ball_a, assume_unique=False)
                points_to_check = ball_r[~mask_in_a]
                if len(points_to_check) > 0 and np.any(cnt[points_to_check] < 2):
                    continue
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_r] -= 1
                cnt[ball_a] += 1
                improved = True

            # After swap, try to delete
            while delete_redundant(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                improved = True

        # 3. 2-for-1 swap: try replacing two codewords with one
        two_for_one_attempts = 0
        max_two_for_one = 100
        while two_for_one_attempts < max_two_for_one and time.time() < deadline - 5.0:
            two_for_one_attempts += 1
            if len(current_code) < 2:
                break

            # Pick two codewords to remove
            c1, c2 = random.sample(list(current_code), 2)
            if c1 not in current_code or c2 not in current_code:
                continue

            ball1 = ball_arrays[c1]
            ball2 = ball_arrays[c2]
            
            # Find candidate c_add: a word whose ball covers all points in ball1 ∪ ball2
            # that are not covered by other codewords
            # First, compute which points in ball1∪ball2 have cnt==1 (only covered by c1 or c2)
            # These must be in ball(c_add)
            union_mask = np.zeros(total, dtype=bool)
            union_mask[ball1] = True
            union_mask[ball2] = True
            
            # Points in union with cnt==1 must be covered by c_add
            # cnt==1 means only one of c1,c2 (or other codewords) covers them
            # Actually cnt counts all codewords. Points with cnt==1 in the union are
            # covered by exactly one codeword total. If that codeword is c1 or c2,
            # removing it leaves them uncovered unless c_add covers them.
            
            # Simpler approach: try a few random c_add candidates and check validity
            for _ in range(5):
                c_add = random.randint(0, total - 1)
                if c_add in current_code:
                    continue
                ball_a = ball_arrays[c_add]
                
                # Check: after removing c1 and c2, adding c_add, all points still covered
                # Points in (ball1 ∪ ball2) \ ball_a must have cnt >= 3 (since we remove 2)
                # Wait: cnt[p] currently counts c1 and c2 if p is in their balls.
                # After removing c1 and c2: cnt_new[p] = cnt[p] - (1 if p in ball1) - (1 if p in ball2) + (1 if p in ball_a)
                # We need cnt_new[p] >= 1 for all p.
                
                # For p in ball1 ∩ ball2: cnt_new = cnt[p] - 2 + (1 if p in ball_a)
                # For p in ball1 \ ball2: cnt_new = cnt[p] - 1 + (1 if p in ball_a)
                # For p in ball2 \ ball1: cnt_new = cnt[p] - 1 + (1 if p in ball_a)
                # For p not in ball1∪ball2: cnt_new = cnt[p] + (1 if p in ball_a) >= cnt[p] >= 1 (ok)
                
                # Critical points: those in ball1∪ball2 not in ball_a
                in_union_not_a = union_mask & ~np.isin(np.arange(total), ball_a, assume_unique=False)
                # This is expensive. Alternative: check specific points.
                
                # Points in ball1 not in ball_a: need cnt >= 2 (since we remove c1, and c2 may or may not cover)
                # Points in ball2 not in ball_a: need cnt >= 2
                # Points in ball1∩ball2 not in ball_a: need cnt >= 3
                
                mask1_in_a = np.isin(ball1, ball_a, assume_unique=False)
                mask2_in_a = np.isin(ball2, ball_a, assume_unique=False)
                
                # ball1 not in ball_a
                b1_not_a = ball1[~mask1_in_a]
                # ball2 not in ball_a  
                b2_not_a = ball2[~mask2_in_a]
                
                valid = True
                if len(b1_not_a) > 0:
                    # For these points, c1 covers them. Does c2 also cover?
                    # If in ball2: cnt >= 3 needed (remove both c1,c2, add c_add which doesn't cover)
                    # If not in ball2: cnt >= 2 needed (remove c1, c2 doesn't cover, c_add doesn't cover)
                    in_b2 = np.isin(b1_not_a, ball2, assume_unique=False)
                    if np.any(in_b2):
                        if np.any(cnt[b1_not_a[in_b2]] < 3):
                            valid = False
                    if valid and np.any(~in_b2):
                        if np.any(cnt[b1_not_a[~in_b2]] < 2):
                            valid = False
                
                if valid and len(b2_not_a) > 0:
                    in_b1 = np.isin(b2_not_a, ball1, assume_unique=False)
                    if np.any(in_b1):
                        if np.any(cnt[b2_not_a[in_b1]] < 3):
                            valid = False
                    if valid and np.any(~in_b1):
                        if np.any(cnt[b2_not_a[~in_b1]] < 2):
                            valid = False
                
                if not valid:
                    continue
                
                # Valid 2-for-1 swap
                current_code.remove(c1)
                current_code.remove(c2)
                current_code.add(c_add)
                cnt[ball1] -= 1
                cnt[ball2] -= 1
                cnt[ball_a] += 1
                improved = True
                break  # Break inner loop, try another pair
                
                # After 2-for-1, try to delete more
                while delete_redundant(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    improved = True

        if not improved:
            no_improve_streak += 1
        else:
            no_improve_streak = 0

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)

    # Convert back to list of lists
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
