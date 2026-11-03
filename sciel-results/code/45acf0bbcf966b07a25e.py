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

    # Local search: targeted swap using inverse index (optimized with smooth surrogate)
    def local_search(code, cover_count, time_budget=100, seed=12345, weight_unique=1.0, weight_global=0.5):
        code_set = set(code)
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(seed)
        
        # Precompute global unique mask for faster global score calculation if needed?
        # No, cover_count changes. But we can use a vectorized approach for candidate scoring.
        
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
                
                # Vectorized scoring for all candidates at once is hard because balls[cand] varies.
                # But we can iterate. To speed up, we check a small sample or all if few.
                # If candidates is large, we might need to be careful.
                
                for cand in candidates:
                    bp_cand = balls[cand]
                    
                    # Score 1: Unique points of new codeword c that are in bp_i
                    # These are points that will be covered by both i (removed) and c (added).
                    # Wait, if we remove i, these points lose 1 cover. If we add c, they gain 1.
                    # Net change: 0. So they remain covered.
                    # The "cost" is how many OTHER unique points c touches?
                    # Let's stick to the previous logic but combined.
                    
                    unique_c_in_overlap = np.sum((cover_count[bp_cand] == 1) & (in_bp_i[bp_cand] == 1))
                    
                    # Global benefit: points in bp_cand that are unique to SOMEONE (including i)
                    # If a point p is unique to j (j!=i), and c covers it, j loses uniqueness. Good.
                    # If a point p is unique to i, and c covers it, i loses it but c takes over. Neutral/Good.
                    # So we want to maximize count of p in bp_cand with cover_count[p]==1.
                    total_unique_in_cand = (cover_count[bp_cand] == 1).sum()
                    
                    # Smooth surrogate: Minimize (unique_c_in_overlap - weight_global * total_unique_in_cand)
                    # Actually, unique_c_in_overlap is a subset of total_unique_in_cand.
                    # Let's define:
                    # Cost = unique_c_in_overlap * weight_unique - total_unique_in_cand * weight_global
                    # We want to minimize this cost.
                    
                    score = unique_c_in_overlap * weight_unique - total_unique_in_cand * weight_global
                    
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
    # Pass 1: Focus on minimizing local unique points (strong weight_unique)
    if time.time() < deadline - 10:
        code, cover_count = local_search(code, cover_count, time_budget=min(30, deadline - time.time() - 10), seed=12345, weight_unique=1.0, weight_global=0.1)
        code, cover_count = trim(code, cover_count)

    # Pass 2: Balance local and global benefits
    if time.time() < deadline - 10:
        code, cover_count = local_search(code, cover_count, time_budget=min(30, deadline - time.time() - 10), seed=54321, weight_unique=1.0, weight_global=0.5)
        code, cover_count = trim(code, cover_count)

    # Pass 3: Focus on global benefits (help other codewords drop unique points)
    if time.time() < deadline - 10:
        code, cover_count = local_search(code, cover_count, time_budget=min(20, deadline - time.time() - 10), seed=99999, weight_unique=0.5, weight_global=1.0)
        code, cover_count = trim(code, cover_count)

    # Pass 4: Final polish with combined weights
    if time.time() < deadline - 5:
        code, cover_count = local_search(code, cover_count, time_budget=min(15, deadline - time.time() - 5), seed=77777, weight_unique=1.0, weight_global=0.5)
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
