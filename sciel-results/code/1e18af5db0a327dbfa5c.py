import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    # Fallback for other parameters
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    # Precompute powers of 3 (little-endian: digit i has weight 3^i)
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute digits for each index
    digits_table = [[0] * 7 for _ in range(N)]
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits_table[i][pos] = tmp % 3
            tmp //= 3

    # Precompute ball masks: for each word index, bitmask of indices within Hamming distance 1
    ball_masks = [0] * N
    for i in range(N):
        mask = 1 << i
        for pos in range(7):
            orig = digits_table[i][pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= (1 << new_i)
        ball_masks[i] = mask

    # Precompute ball lists for local search (list of indices in each ball)
    ball_lists = [[] for _ in range(N)]
    for i in range(N):
        m = ball_masks[i]
        lst = []
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            lst.append(idx)
            m ^= lsb
        ball_lists[i] = lst

    full_mask = (1 << N) - 1

    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        while uncovered:
            best_idx = -1
            best_count = -1

            # Scan all candidates; use random tie-breaking by shuffling order
            candidates = list(range(N))
            rng.shuffle(candidates)

            for i in candidates:
                if i in selected_set:
                    continue
                cnt = (uncovered & ball_masks[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    if best_count == uncovered.bit_count():
                        break

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]

            if time.time() > deadline - 10.0:
                break

        return selected

    def local_search(code, seed, time_budget):
        """Aggressive local search: deletion, 1-for-1 swap, 2-for-1 exchange."""
        rng = random.Random(seed)
        code = list(code)
        code_set = set(code)

        # cnt[i] = number of codewords in code that cover word i
        cnt = [0] * N
        for c in code:
            for w in ball_lists[c]:
                cnt[w] += 1

        best_code = code[:]
        best_size = len(code)
        elapsed_budget = 0.0

        while time.time() < time_budget:
            improved = False

            # Phase 1: Deletion - remove any codeword that doesn't uniquely cover any word
            to_remove = []
            for c in code:
                is_essential = False
                for w in ball_lists[c]:
                    if cnt[w] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    to_remove.append(c)

            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    for w in ball_lists[c]:
                        cnt[w] -= 1
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                improved = True
                continue

            # Phase 2: 1-for-1 Swap
            # Pick a random codeword c, find words it uniquely covers,
            # find candidates that cover all those words, swap.
            swapped = False
            for _ in range(min(30, len(code))):
                if time.time() > time_budget:
                    break
                c = rng.choice(code)
                unique_words = [w for w in ball_lists[c] if cnt[w] == 1]

                if not unique_words:
                    # Redundant, should have been caught in deletion
                    code.remove(c)
                    code_set.remove(c)
                    for w in ball_lists[c]:
                        cnt[w] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                    improved = True
                    swapped = True
                    break

                # Find candidates that cover all unique_words
                # A candidate c' must be in the intersection of balls of all unique_words
                # i.e., for each w in unique_words, c' must be in ball_lists[w]
                # Equivalently: c' must cover w, meaning w is in ball_lists[c']
                # We need c' such that for all w in unique_words, w in ball_lists[c']
                # This means c' is in the set of codewords whose ball contains all unique_words.

                # Build the set of valid candidates:
                # c' is valid if for all w in unique_words, c' covers w
                # c' covers w iff w is in ball_lists[c'] iff c' is in the "reverse ball" of w
                # Let's precompute reverse balls: for each word w, which codewords cover it?
                # Actually, ball_lists[c] gives words covered by c.
                # We need: for all w in unique_words, c' in {c : w in ball_lists[c]}
                # This is the intersection over w in unique_words of the set of codewords covering w.

                # For efficiency, let's just check candidates from the ball of the first unique word
                # The candidate must be within distance 1 of each unique word.
                # So candidate must be in the intersection of balls(unique_words).

                cand_mask = ball_masks[unique_words[0]]
                for w in unique_words[1:]:
                    cand_mask &= ball_masks[w]
                    if not cand_mask:
                        break

                if not cand_mask:
                    continue

                # Collect candidates not already in code
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if idx not in code_set:
                        candidates.append(idx)
                    m ^= lsb

                if not candidates:
                    continue

                c_prime = rng.choice(candidates)

                # Verify: after removing c and adding c_prime, all words still covered
                # Words uniquely covered by c must be covered by c_prime (already ensured by construction)
                # Words covered by c but not c_prime are still covered by other codewords (cnt[w] >= 2 before removal)
                # Words covered by c_prime but not c are new coverage (fine)

                # Perform swap
                code.remove(c)
                code_set.remove(c)
                for w in ball_lists[c]:
                    cnt[w] -= 1

                code.append(c_prime)
                code_set.add(c_prime)
                for w in ball_lists[c_prime]:
                    cnt[w] += 1

                # Verify validity: all words must have cnt >= 1
                valid = True
                for w in ball_lists[c] + ball_lists[c_prime]:
                    if cnt[w] < 1:
                        valid = False
                        break

                if valid:
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                    improved = True
                    swapped = True
                    break
                else:
                    # Undo swap
                    code.remove(c_prime)
                    code_set.remove(c_prime)
                    for w in ball_lists[c_prime]:
                        cnt[w] -= 1

                    code.append(c)
                    code_set.add(c)
                    for w in ball_lists[c]:
                        cnt[w] += 1

            if swapped:
                continue

            # Phase 3: 2-for-1 Exchange
            # Try to replace two codewords c1, c2 with one codeword c3
            # c3 must cover all words that c1 and c2 uniquely cover (together)
            if len(code) >= 2:
                # Pick two random codewords
                for _ in range(min(20, len(code) * (len(code) - 1) // 2)):
                    if time.time() > time_budget:
                        break
                    i1, i2 = rng.sample(range(len(code)), 2)
                    c1, c2 = code[i1], code[i2]

                    # Find words uniquely covered by c1 or c2 (i.e., cnt[w] == 1 and w in ball[c1] or ball[c2])
                    unique_to_pair = set()
                    for w in ball_lists[c1]:
                        if cnt[w] == 1:
                            unique_to_pair.add(w)
                    for w in ball_lists[c2]:
                        if cnt[w] == 1:
                            unique_to_pair.add(w)

                    if not unique_to_pair:
                        # Both are redundant together, just remove one
                        code.pop(i2)
                        code_set.remove(c2)
                        for w in ball_lists[c2]:
                            cnt[w] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        improved = True
                        break

                    # Find candidates c3 that cover all words in unique_to_pair
                    # c3 must be in intersection of balls of all words in unique_to_pair
                    unique_list = list(unique_to_pair)
                    cand_mask = ball_masks[unique_list[0]]
                    for w in unique_list[1:]:
                        cand_mask &= ball_masks[w]
                        if not cand_mask:
                            break

                    if not cand_mask:
                        continue

                    # Also, c3 must not be c1 or c2
                    cand_mask &= ~(1 << c1)
                    cand_mask &= ~(1 << c2)

                    if not cand_mask:
                        continue

                    candidates = []
                    m = cand_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx not in code_set:
                            candidates.append(idx)
                        m ^= lsb

                    if not candidates:
                        continue

                    c3 = rng.choice(candidates)

                    # Perform 2-for-1: remove c1, c2, add c3
                    # Save state for rollback
                    old_cnt = cnt[:]

                    code.remove(c1)
                    code_set.remove(c1)
                    code.remove(c2)
                    code_set.remove(c2)
                    for w in ball_lists[c1]:
                        cnt[w] -= 1
                    for w in ball_lists[c2]:
                        cnt[w] -= 1

                    code.append(c3)
                    code_set.add(c3)
                    for w in ball_lists[c3]:
                        cnt[w] += 1

                    # Verify validity
                    valid = True
                    for w in range(N):
                        if cnt[w] < 1:
                            valid = False
                            break

                    if valid:
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        improved = True
                        break
                    else:
                        # Rollback
                        cnt[:] = old_cnt
                        code.remove(c3)
                        code_set.remove(c3)
                        code.append(c1)
                        code_set.add(c1)
                        code.append(c2)
                        code_set.add(c2)

            if not improved:
                # Perturbation: randomly swap in a new codeword and remove a random one
                # Then re-run local search from the perturbed state
                if len(code) > 0:
                    c_out = rng.choice(code)
                    # Find a random candidate not in code
                    all_candidates = [i for i in range(N) if i not in code_set]
                    if all_candidates:
                        c_in = rng.choice(all_candidates)

                        # Perform swap
                        code.remove(c_out)
                        code_set.remove(c_out)
                        for w in ball_lists[c_out]:
                            cnt[w] -= 1

                        code.append(c_in)
                        code_set.add(c_in)
                        for w in ball_lists[c_in]:
                            cnt[w] += 1

                        # If invalid, revert
                        valid = True
                        for w in range(N):
                            if cnt[w] < 1:
                                valid = False
                                break

                        if not valid:
                            # Revert
                            code.remove(c_in)
                            code_set.remove(c_in)
                            for w in ball_lists[c_in]:
                                cnt[w] -= 1

                            code.append(c_out)
                            code_set.add(c_out)
                            for w in ball_lists[c_out]:
                                cnt[w] += 1

        return best_code

    # Main search: multiple restarts with greedy init + local search
    best_overall = None
    best_size = float('inf')

    seed = 42
    restart_num = 0
    while time.time() < deadline - 5.0:
        restart_num += 1
        # Greedy initialization
        init_code = greedy_init(seed)
        seed += 1

        if not init_code:
            break

        # Local search with time budget
        ls_budget = min(deadline - 3.0, time.time() + 20.0)
        result = local_search(init_code, seed, ls_budget)
        seed += 1

        if len(result) < best_size:
            best_size = len(result)
            best_overall = result[:]

        if best_size <= 170:
            break

    # Fallback
    if not best_overall:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_overall = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Convert to index form
        idx_list = []
        for word in best_overall:
            idx = 0
            for pos, d in enumerate(word):
                idx += d * p3[pos]
            idx_list.append(idx)
        best_overall = idx_list

    # Convert back to lists of integers (most-significant digit first)
    result = []
    for idx in best_overall:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        # Convert to most-significant first
        digits.reverse()
        result.append(digits)

    return result


def _construct_general(q, n, radius, deadline):
    """Simple greedy + prune for general parameters."""
    total = q ** n
    words = list(range(total))

    def decode(val):
        return [val // (q ** (n - i - 1)) % q for i in range(n)]

    def encode(digits):
        val = 0
        for i, d in enumerate(digits):
            val += d * (q ** (n - i - 1))
        return val

    # Precompute balls
    balls = {}
    for w in words:
        digs = decode(w)
        neighbors = {w}
        for pos in range(n):
            original = digs[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digs = digs[:]
                    new_digs[pos] = new_val
                    neighbors.add(encode(new_digs))
        balls[w] = neighbors

    # Greedy
    uncovered = set(words)
    selected = []
    while uncovered and time.time() < deadline - 5.0:
        best_w = max(words, key=lambda w: len(balls[w] & uncovered))
        selected.append(best_w)
        uncovered -= balls[best_w]

    # Prune
    if selected:
        counts = {w: 0 for w in words}
        for c in selected:
            for w in balls[c]:
                counts[w] += 1

        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                can_remove = all(counts[w] > 1 for w in balls[c])
                if can_remove:
                    for w in balls[c]:
                        counts[w] -= 1
                    selected.pop(i)
                    changed = True

    return [decode(w) for w in selected]

# EVOLVE-BLOCK-END
