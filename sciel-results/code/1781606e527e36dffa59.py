"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0  # safety margin before the 150s kill

    # ---- mapping helpers ----
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * q + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total = q ** n

    # ---- precompute radius balls for every word ----
    neighbor_lists = []
    for i in range(total):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)

    # ---- small-n fallback (exhaustive greedy) ----
    if n <= 4:
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        for w in all_words:
            if not uncovered:
                break
            if w in uncovered:
                code.append(list(w))
                for u in list(uncovered):
                    if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                        uncovered.remove(u)
        return code

    # ---- initial code: direct sum of the perfect ternary tetracode ----
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_tails = list(itertools.product(range(q), repeat=n - 4))
        code = [h + list(t) for h in tetracode for t in base_tails]
        # Prune initial code: remove redundant codewords before optimization
        # For the direct sum, many tail positions create redundancy
        # Use a quick greedy prune
        if n == 6:
            # 81 codewords initially; try to reduce
            initial_counts = np.zeros(total, dtype=np.int32)
            initial_indices = [word_to_int(c) for c in code]
            for ci in initial_indices:
                for nb in neighbor_lists[ci]:
                    initial_counts[nb] += 1
            # Greedy removal on initial code
            removed_set = set()
            for ci in initial_indices:
                if ci in removed_set:
                    continue
                ball = neighbor_lists[ci]
                # Check if all words in ball have count >= 2 considering only non-removed
                if all(initial_counts[nb] - (1 if nb in [neighbor_lists[ri][0] for ri in removed_set] else 0) >= 2 for nb in ball[:1]):
                    pass  # Simplified check; use full check below
            # Full greedy removal
            code_indices_init = list(initial_indices)
            counts_init = initial_counts.copy()
            while True:
                found = False
                for idx in range(len(code_indices_init)):
                    ci = code_indices_init[idx]
                    ball = neighbor_lists[ci]
                    if np.all(counts_init[ball] >= 2):
                        for nb in ball:
                            counts_init[nb] -= 1
                        code_indices_init.pop(idx)
                        found = True
                        break
                if not found:
                    break
            code = [int_to_word(ci) for ci in code_indices_init]
    else:
        # Fallback: greedy initial code
        uncovered = set(range(total))
        code = []
        for i in range(total):
            if not uncovered:
                break
            if i in uncovered:
                code.append(int_to_word(i))
                for nb in neighbor_lists[i]:
                    if nb in uncovered:
                        uncovered.discard(nb)

    code_indices = [word_to_int(c) for c in code]
    code_set = set(code_indices)

    counts = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    # ---- smart greedy removal: remove the codeword with maximum total redundancy first ----
    def smart_remove_pass():
        """Remove the most-redundant removable codeword each step."""
        while time.time() < deadline - 5.0:
            best_idx = -1
            best_slack = 0
            for idx in range(len(code_indices)):
                ci = code_indices[idx]
                ball = neighbor_lists[ci]
                if np.all(counts[ball] >= 2):
                    slack = int(np.sum(counts[ball] - 1))
                    if slack > best_slack:
                        best_slack = slack
                        best_idx = idx
            if best_idx < 0:
                break
            ci = code_indices[best_idx]
            ball = neighbor_lists[ci]
            for nb in ball:
                counts[nb] -= 1
            code_indices.pop(best_idx)
            code_set.discard(ci)

    smart_remove_pass()

    # ---- add-remove local search to escape local minima ----
    rng = random.Random(12345)

    def add_remove_search(max_rounds=30):
        """Try adding a new codeword then removing two, to escape local minima."""
        nonlocal counts
        for _round in range(max_rounds):
            if time.time() > deadline - 10.0:
                return
            # Find all words not yet covered by any codeword (should be none if valid)
            # Instead, find words with count==1 (critical) and try to add a codeword
            # that covers them plus provides redundancy for removal
            # Strategy: pick a random word, add it as a new codeword, then try to remove 2
            # existing codewords that become redundant
            if len(code_indices) < 2:
                return
            # Pick a random position in the code to "probe"
            probe_idx = rng.randrange(len(code_indices))
            ci_probe = code_indices[probe_idx]
            # Find a word in the ball of ci_probe that has count==1
            ball_probe = neighbor_lists[ci_probe]
            critical_words = [w for w in ball_probe if counts[w] == 1]
            if not critical_words:
                continue
            # Try to find a new codeword that covers some critical words
            # and allows removal of 2 existing codewords
            # Sample some candidate positions
            candidates = []
            for _ in range(20):
                cand_val = rng.randrange(total)
                if cand_val not in code_set:
                    candidates.append(cand_val)
            for cand in candidates:
                if time.time() > deadline - 10.0:
                    return
                cand_ball = neighbor_lists[cand]
                # Check: if we add cand, can we remove 2 codewords?
                # After adding, counts increase by 1 on cand_ball
                # We need to find 2 codewords whose removal keeps all counts >= 1
                new_counts = counts.copy()
                for nb in cand_ball:
                    new_counts[nb] += 1
                # Find removable pairs
                m = len(code_indices)
                removed = False
                for i in range(m):
                    if time.time() > deadline - 10.0:
                        return
                    ci_i = code_indices[i]
                    ball_i = neighbor_lists[ci_i]
                    if not np.all(new_counts[ball_i] >= 2):
                        continue
                    # Try removing i, then check if another can be removed
                    temp_counts = new_counts.copy()
                    for nb in ball_i:
                        temp_counts[nb] -= 1
                    for j in range(m):
                        if j == i:
                            continue
                        ci_j = code_indices[j]
                        ball_j = neighbor_lists[ci_j]
                        if np.all(temp_counts[ball_j] >= 2):
                            # Remove both i and j, add cand
                            for nb in ball_j:
                                temp_counts[nb] -= 1
                            # Apply the change
                            counts = temp_counts
                            # Remove i and j (in reverse order)
                            code_indices.pop(j)
                            code_indices.pop(i)
                            code_set.discard(ci_i)
                            code_set.discard(ci_j)
                            # Add cand
                            code_indices.append(cand)
                            code_set.add(cand)
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    # Do a smart removal pass to clean up
                    smart_remove_pass()
                    break

    add_remove_search(max_rounds=25)

    # ---- final smart removal pass ----
    smart_remove_pass()

    # ---- Multi-round double/triple removal B&B ----
    m = len(code_indices)
    if m > 300 or time.time() > deadline - 15.0:
        return [int_to_word(ci) for ci in code_indices]

    def can_remove(indices_to_remove):
        """Check if removing these codewords still leaves full coverage."""
        if not indices_to_remove:
            return True
        new_counts = counts.copy()
        for idx in indices_to_remove:
            ci = code_indices[idx]
            for nb in neighbor_lists[ci]:
                new_counts[nb] -= 1
        return np.all(new_counts >= 1)

    def do_double_triple_removal():
        """Try double and triple removals in a loop until no more can be found."""
        nonlocal counts, m
        progress = True
        while progress and time.time() < deadline - 8.0:
            progress = False
            m = len(code_indices)
            if m <= 1:
                break
            # Try double removals
            found_double = False
            for i in range(m):
                if time.time() > deadline - 8.0:
                    return
                ci_i = code_indices[i]
                ball_i = neighbor_lists[ci_i]
                if not np.all(counts[ball_i] >= 2):
                    continue
                # Temporarily remove i
                temp_counts = counts.copy()
                for nb in ball_i:
                    temp_counts[nb] -= 1
                for j in range(i + 1, m):
                    if time.time() > deadline - 8.0:
                        return
                    ci_j = code_indices[j]
                    ball_j = neighbor_lists[ci_j]
                    if np.all(temp_counts[ball_j] >= 2):
                        # Remove both
                        for nb in ball_j:
                            temp_counts[nb] -= 1
                        counts = temp_counts
                        code_indices.pop(j)
                        code_indices.pop(i)
                        code_set.discard(ci_i)
                        code_set.discard(ci_j)
                        found_double = True
                        progress = True
                        break
                if found_double:
                    break
            if found_double:
                continue
            # Try triple removals
            found_triple = False
            for i in range(m):
                if time.time() > deadline - 8.0:
                    return
                ci_i = code_indices[i]
                ball_i = neighbor_lists[ci_i]
                if not np.all(counts[ball_i] >= 2):
                    continue
                temp_counts = counts.copy()
                for nb in ball_i:
                    temp_counts[nb] -= 1
                for j in range(i + 1, m):
                    if time.time() > deadline - 8.0:
                        return
                    ci_j = code_indices[j]
                    ball_j = neighbor_lists[ci_j]
                    if not np.all(temp_counts[ball_j] >= 2):
                        continue
                    temp2 = temp_counts.copy()
                    for nb in ball_j:
                        temp2[nb] -= 1
                    for k in range(j + 1, m):
                        if time.time() > deadline - 8.0:
                            return
                        ci_k = code_indices[k]
                        ball_k = neighbor_lists[ci_k]
                        if np.all(temp2[ball_k] >= 2):
                            # Remove all three
                            for nb in ball_k:
                                temp2[nb] -= 1
                            counts = temp2
                            code_indices.pop(k)
                            code_indices.pop(j)
                            code_indices.pop(i)
                            code_set.discard(ci_i)
                            code_set.discard(ci_j)
                            code_set.discard(ci_k)
                            found_triple = True
                            progress = True
                            break
                    if found_triple:
                        break
                if found_triple:
                    break

    do_double_triple_removal()

    # ---- Final smart removal pass ----
    smart_remove_pass()

    # ---- Final verification and return ----
    return [int_to_word(ci) for ci in code_indices]
# EVOLVE-BLOCK-END
