import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Covering code for the football-pool problem K_q(n, radius).
    Uses structured init, greedy extension, aggressive trimming, 
    1-swap local search, and 2-swap (2-for-1) local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # radius = 1

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # ---------- Precompute balls ----------
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = int_to_digits(w, n, q)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                digits[pos] = nv
                ball.append(digits_to_int(digits, q))
            digits[pos] = orig
        balls[w] = ball

    # ---------- Inverse index ----------
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # ---------- Structured Init ----------
    def structured_init():
        tetracode = [
            [0, 0, 0, 0], [0, 1, 2, 1], [0, 2, 1, 2],
            [1, 0, 1, 2], [1, 1, 2, 0], [1, 2, 0, 2]
        ]
        # Add a few more good words to the tetracode base if needed? 
        # The standard tetracode is 9 words. This list has 6. 
        # Let's use the full 9-word tetracode for better init.
        tetracode_full = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
            [2, 0, 2, 0],
            [2, 1, 0, 2],
            [2, 2, 1, 1]
        ]
        product_code = []
        for a in tetracode_full:
            for b in tetracode_full:
                product_code.append(a + b)
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        # Greedy extension
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    code, cover_count = structured_init()
    code_set = set(code)

    # ---------- Helpers ----------
    def get_unique_points(c):
        return balls[c][cover_count[balls[c]] == 1]

    def trim():
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True

    trim()

    # ---------- Local Search ----------
    def local_search(time_budget=130.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        rng = random.Random(12345)
        
        # Pre-compute unique points for all codewords to speed up checks?
        # No, cover_count changes, so unique points change.
        
        while time.time() < end_time:
            improved = False
            
            # 1. Trim
            if trim():
                improved = True
                continue # Restart loop after trim

            # 2. 1-Swap (Move)
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > end_time:
                    return
                i = code[idx]
                bp_i = balls[i]
                unique_pts = get_unique_points(i)
                
                if len(unique_pts) == 0:
                    # Should have been caught by trim, but just in case
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates that cover unique_pts
                # Intersection of centers_covering for all w in unique_pts
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                # Exclude current codeword and existing codewords
                candidates = [c for c in candidate_set if c != i and c not in code_set]
                if not candidates:
                    continue
                
                # Score: Minimize waste
                # Waste: Number of points in new ball that are already covered by >= 2 others
                # Or simply: Minimize sum of cover_count in new ball (excluding i's contribution)
                best_swap = None
                best_score = None
                
                # Optimization: Only check a subset of candidates if list is huge
                if len(candidates) > 50:
                    rng.shuffle(candidates)
                    candidates = candidates[:50]

                for j in candidates:
                    bp_j = balls[j]
                    # Simulate move: 
                    # cover_count[bp_i] -= 1
                    # cover_count[bp_j] += 1
                    # We want to minimize the number of points in bp_j that become "over-covered"
                    # i.e., cover_count[bp_j] (after move) >= 2
                    # Current cover_count[bp_j] is C_j. After move, it's C_j + 1.
                    # We want to minimize sum( (C_j + 1) >= 2 ) = sum( C_j >= 1 )
                    # So we want to minimize the number of points in bp_j that are ALREADY covered.
                    waste = int(np.sum(cover_count[bp_j] > 0))
                    
                    if best_score is None or waste < best_score:
                        best_score = waste
                        best_swap = j

                if best_swap is not None:
                    j = best_swap
                    cover_count[bp_i] -= 1
                    cover_count[balls[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

            if improved:
                continue

            # 3. 2-Swap (2-for-1)
            # Look for pairs (i, j) such that U_i U U_j is covered by some k
            # Only consider pairs where U_i and U_j are small
            small_unique = []
            for idx, c in enumerate(code):
                u = get_unique_points(c)
                if len(u) <= 3: # Heuristic threshold
                    small_unique.append((idx, c, u))
            
            # Shuffle to randomize order
            rng.shuffle(small_unique)
            
            found_2swap = False
            for a_idx, a_code, a_u in small_unique:
                if time.time() > end_time:
                    return
                if a_idx >= len(code): continue # Code might have changed
                
                # Find b
                for b_idx, b_code, b_u in small_unique:
                    if b_idx == a_idx or b_idx >= len(code):
                        continue
                    if b_idx < a_idx: # Avoid duplicates
                        continue
                    
                    union_u = list(set(a_u) | set(b_u))
                    if len(union_u) > 6: # Skip if union is too large
                        continue
                    
                    # Find k covering union_u
                    cand_k = None
                    for w in union_u:
                        cc = centers_covering[w]
                        if cand_k is None:
                            cand_k = set(cc)
                        else:
                            cand_k &= cc
                        if not cand_k:
                            break
                    
                    if not cand_k:
                        continue
                    
                    # Exclude a, b and existing
                    candidates_k = [k for k in cand_k if k != a_code and k != b_code and k not in code_set]
                    if not candidates_k:
                        continue
                    
                    # Pick one (e.g., first one, or best waste)
                    # Just pick one for speed
                    k = candidates_k[0]
                    
                    # Perform 2-swap
                    # Remove a, b
                    # Add k
                    # Check validity: k covers U_a U U_b. 
                    # Does removing a, b uncover any other points?
                    # Points in ball(a) \ U_a are covered by others.
                    # Points in ball(b) \ U_b are covered by others.
                    # So yes, valid.
                    
                    # Update counts
                    cover_count[balls[a_code]] -= 1
                    cover_count[balls[b_code]] -= 1
                    cover_count[balls[k]] += 1
                    
                    # Update code list
                    # Remove a and b
                    # Order matters for indices
                    if a_idx < b_idx:
                        code.pop(b_idx)
                        code.pop(a_idx)
                    else:
                        code.pop(a_idx)
                        code.pop(b_idx - 1)
                    code_set.discard(a_code)
                    code_set.discard(b_code)
                    
                    code.append(k)
                    code_set.add(k)
                    
                    improved = True
                    found_2swap = True
                    break
                
                if found_2swap:
                    break
            
            if found_2swap:
                continue
            
            # No improvement in this pass
            # Try random restart of shuffle seed?
            # Or break if stuck for a while?
            # Let's just break to save time, or continue with new shuffle
            if time.time() > end_time - 5:
                break

    local_search(time_budget=130.0)
    trim()

    result = [int_to_digits(c, n, q) for c in code]
    return result
