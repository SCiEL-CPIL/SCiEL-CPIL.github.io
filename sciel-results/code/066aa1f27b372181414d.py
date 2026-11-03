import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    # Precompute balls as numpy array
    def build_balls():
        balls = np.empty((total, ball_size), dtype=np.int32)
        for w in range(total):
            digits = []
            tmp = w
            for i in range(n):
                digits.append(tmp % q)
                tmp //= q
            digits = digits[::-1]

            ball = [w]
            for pos in range(n):
                orig = digits[pos]
                for nv in range(q):
                    if nv != orig:
                        new_word = digits[:]
                        new_word[pos] = nv
                        val = 0
                        for d in new_word:
                            val = val * q + d
                        ball.append(val)
            balls[w] = ball
        return balls

    balls = build_balls()

    # Build inverse index: for each word, which centers cover it
    # This is critical for fast candidate enumeration in local search
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # Tetracode product initialization
    def structured_init():
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        
        def digits_to_int(digits, base):
            num = 0
            for d in reversed(digits):
                num = num * base + d
            return num

        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        
        code_ints = [digits_to_int(c, q) for c in product_code]
        code_ints = list(set(code_ints))
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        # Greedy extension
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            gains = uncovered[balls].sum(axis=1)
            # Exclude already chosen
            for c in code_ints:
                gains[c] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        
        return code_ints, cover_count

    # Greedy set cover (standard)
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)

        while covered_count < total:
            if time.time() > deadline - 5:
                break

            covered_in_balls = covered[balls]
            covered_per_ball = covered_in_balls.sum(axis=1)
            gains = ball_size - covered_per_ball

            best_word = int(np.argmax(gains))
            best_new = int(gains[best_word])
            if best_new == 0:
                break

            code.append(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
                cover_count[bp] += 1

        return code, cover_count

    # Try structured init first, fall back to pure greedy
    code, cover_count = structured_init()
    if not code:
        code, cover_count = greedy_cover()

    code_set = set(code)

    # Aggressive trimming of redundant codewords
    def trim(code, cover_count):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if time.time() > deadline - 10:
                    break
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True
        return code, cover_count

    code, cover_count = trim(code, cover_count)

    # Local search: targeted swap using inverse index (optimized)
    def local_search(code, cover_count, time_budget=100, seed=12345, score_mode='unique'):
        code_set = set(code)
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(seed)

        while improved and time.time() < end_time:
            improved = False
            # Shuffle for diversity
            indices = list(range(len(code)))
            rng.shuffle(indices)

            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                # Find words uniquely covered by i
                unique_mask = cover_count[bp_i] == 1
                unique_pts = bp_i[unique_mask]

                if len(unique_pts) == 0:
                    # Remove redundant
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates that cover ALL unique_pts using inverse index
                candidate_set = None
                for w in unique_pts:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                # Filter out already in code
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue

                best_cand = None
                best_score = float('inf')

                # Precompute mask for intersection with bp_i for exact scoring
                in_bp_i = np.zeros(total, dtype=np.int8)
                in_bp_i[bp_i] = 1

                # Precompute: for each point, how many codewords cover it (current)
                # We need to evaluate: for candidate c, what is the new unique count of c?
                # unique_count_c = count of p in bp_c ∩ bp_i with cover_count[p] == 1
                # Additionally, if score_mode == 'global', also consider the effect on other codewords.
                # A smooth surrogate: minimize sum over all codewords j of unique_count_j.
                # For simplicity, we approximate: the swap removes i and adds c.
                # The unique points of other codewords change only if they share points with bp_i or bp_c.
                # This is expensive to compute exactly, so we use a cheaper proxy:
                # Minimize the number of points in bp_c that have cover_count == 1 (these will become
                # "contested" and may reduce uniqueness of existing codewords).
                
                if score_mode == 'unique':
                    # Minimize unique points of the new codeword c
                    # unique_count_c = count of p in bp_c ∩ bp_i with cover_count[p] == 1
                    for cand in candidates:
                        bp_cand = balls[cand]
                        score = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                        if score < best_score:
                            best_score = score
                            best_cand = cand
                elif score_mode == 'global':
                    # Minimize a proxy for total unique points across all codewords.
                    # After swap: remove i, add c.
                    # For any existing codeword j != i:
                    #   new_cover_count[p] = cover_count[p] - (1 if p in bp_i else 0) + (1 if p in bp_c else 0)
                    #   unique_count_j_new = count of p in bp_j with new_cover_count[p] == 1
                    # This is expensive. Instead, use a cheaper heuristic:
                    # Minimize the "overlap" of c with points that are currently unique to some codeword.
                    # If c covers a point that is unique to codeword j, then after adding c, that point is no longer unique to j.
                    # So we want c to cover as many "unique" points (from other codewords) as possible.
                    # unique_to_others = points p where cover_count[p] == 1 and p is not in bp_i (since i's unique pts are handled)
                    # Actually, if p is unique to i and p in bp_c, that's fine (c takes over).
                    # If p is unique to j (j != i) and p in bp_c, then j loses a unique point. Good.
                    # So: maximize count of p in bp_c that are unique to some codeword j != i.
                    # Proxy: count of p in bp_c with cover_count[p] == 1, minus count of p in bp_c ∩ bp_i with cover_count[p] == 1
                    #       = (cover_count[bp_c] == 1).sum() - (cover_count[bp_c ∩ bp_i] == 1).sum()
                    for cand in candidates:
                        bp_cand = balls[cand]
                        total_unique_in_cand = (cover_count[bp_cand] == 1).sum()
                        unique_in_overlap = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                        # We want to maximize (total_unique_in_cand - unique_in_overlap)
                        # which is the number of points in bp_c that are unique to some codeword j != i
                        score = -(total_unique_in_cand - unique_in_overlap)
                        if score < best_score:
                            best_score = score
                            best_cand = cand
                else:
                    # Combined: minimize unique_count_c + lambda * (something)
                    for cand in candidates:
                        bp_cand = balls[cand]
                        unique_c = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                        total_unique_in_cand = (cover_count[bp_cand] == 1).sum()
                        unique_in_overlap = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                        benefit = total_unique_in_cand - unique_in_overlap
                        score = unique_c - 0.5 * benefit
                        if score < best_score:
                            best_score = score
                            best_cand = cand

                if best_cand is not None:
                    # Perform swap
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    
                    bp_new = balls[best_cand]
                    cover_count[bp_new] += 1
                    code.append(best_cand)
                    code_set.add(best_cand)
                    improved = True

        return code, cover_count

    # Multi-pass local search with different strategies and seeds
    # Pass 1: minimize unique points of new codeword
    code, cover_count = local_search(code, cover_count, time_budget=45, seed=12345, score_mode='unique')
    code, cover_count = trim(code, cover_count)

    # Pass 2: global heuristic - maximize benefit to other codewords
    code, cover_count = local_search(code, cover_count, time_budget=30, seed=54321, score_mode='global')
    code, cover_count = trim(code, cover_count)

    # Pass 3: combined scoring
    code, cover_count = local_search(code, cover_count, time_budget=15, seed=99999, score_mode='combined')
    code, cover_count = trim(code, cover_count)

    # Final aggressive trim: iterate until no more removals possible
    for _ in range(5):
        code, cover_count = trim(code, cover_count)
        if time.time() > deadline - 5:
            break

    # Convert to word lists
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    result = [to_word(w) for w in code]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
