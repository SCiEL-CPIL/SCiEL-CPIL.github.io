import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n

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

    # Precompute balls as a list of numpy arrays
    balls_list = [None] * total_words
    for i in range(total_words):
        d = [0] * n
        for pos in range(n):
            d[pos] = (i // (q ** pos)) % q
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                val = 0
                for k in range(n):
                    val += d[k] * (q ** k)
                neighbors.append(val)
            d[pos] = orig
        balls_list[i] = np.array(neighbors, dtype=np.int32)

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute inverse index: for each word, which centers cover it
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Greedy Set Cover with incremental gain updates
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # gains[c] = number of uncovered words in ball[c]
    gains = np.array([np.sum(uncovered[b]) for b in balls_list], dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        best_center = int(np.argmax(gains))
        max_gain = gains[best_center]
        if max_gain == 0:
            break

        code.append(best_center)
        b = balls[best_center]
        new_uncovered = uncovered[b]
        uncovered[b] = False
        cover_count[b] += 1

        # Update gains for all centers that overlap with b
        # For efficiency, only update centers whose ball intersects b
        # A center c overlaps with b iff ball[c] has any word in b
        # We can use: for each word w in b, all centers in inv_ball[w] need update
        # But that's expensive. Instead, recompute gains lazily.
        # For now, mark that gains need recomputation for affected centers.
        # Simpler: just recompute gains for all centers (expensive but correct)
        # Better: only recompute for centers that share at least one word with b
        affected = set()
        for w in b:
            for c in inv_ball[w]:
                affected.add(c)
        for c in affected:
            gains[c] = np.sum(uncovered[balls[c]])

    # Trimming: remove redundant codewords
    def trim_code():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    trim_code()

    # Local search with swap moves
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    def local_search(seed, max_rounds=300):
        nonlocal code, cover_count
        random.seed(seed)
        code_set = set(code)
        
        improved = True
        swap_round = 0

        while improved and time.time() < deadline and swap_round < max_rounds:
            improved = False
            swap_round += 1

            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                # Find candidates: centers that cover ALL unique words of i
                unique_list = unique.tolist()
                if len(unique_list) == 0:
                    continue

                candidate_set = set(inv_ball[unique_list[0]])
                for w in unique_list[1:]:
                    candidate_set &= set(inv_ball[w])
                    if not candidate_set:
                        break

                candidate_set -= code_set
                if not candidate_set:
                    continue

                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(100, len(candidates))
                candidates = candidates[:n_candidates]

                bi = balls[i]
                bi_mask = np.zeros(total_words, dtype=bool)
                bi_mask[bi] = True

                best_swap = None
                best_cost = float('inf')

                for j in candidates:
                    if time.time() > deadline:
                        return
                    bj = balls[j]
                    bj_mask = np.zeros(total_words, dtype=bool)
                    bj_mask[bj] = True
                    # cost = number of words in bj that are currently uncovered (cover_count==0)
                    # These become over-covered after the swap
                    cost = np.sum((cover_count[bj] == 0))
                    if cost < best_cost:
                        best_cost = cost
                        best_swap = j

                if best_swap is not None:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with multiple seeds
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if time.time() < deadline:
            local_search(seed, max_rounds=150)
            trim_code()

    # Final polish: one more round of aggressive trimming and small swaps
    trim_code()
    if time.time() < deadline:
        local_search(99999, max_rounds=50)
        trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
