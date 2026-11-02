import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute ball masks as numpy boolean arrays (much faster than bit ints)
    # ball_mask[i] is a boolean array of length total_words
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i, w in enumerate(all_words):
        ball_masks[i, i] = True
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = w[:pos] + (new_val,) + w[pos + 1:]
                j = word_to_idx[new_word]
                ball_masks[i, j] = True

    # Greedy set cover with efficient marginal gain tracking
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    code_set = set()
    used = np.zeros(total_words, dtype=bool)

    # Symmetry breaking: fix first codeword to all zeros
    first_idx = word_to_idx[(0,) * n]
    code.append(list(all_words[first_idx]))
    code_set.add(all_words[first_idx])
    used[first_idx] = True
    uncovered[ball_masks[first_idx]] = False

    # Greedy: pick word with max marginal coverage
    while uncovered.any() and time.time() < deadline:
        # Marginal gain for each unused word
        gains = np.sum(uncovered[None, :] & ball_masks, axis=1)
        gains[used] = -1  # Exclude used words

        best_idx = int(np.argmax(gains))
        if gains[best_idx] <= 0:
            break

        code.append(list(all_words[best_idx]))
        code_set.add(all_words[best_idx])
        used[best_idx] = True
        uncovered[ball_masks[best_idx]] = False

    # Simulated Annealing to reduce code size
    # Objective: minimize len(code) while maintaining coverage
    def compute_coverage_counts(code_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in code_list:
            idx = word_to_idx[tuple(cw)]
            counts[ball_masks[idx]] += 1
        return counts

    def is_valid(code_list):
        if not code_list:
            return False
        counts = compute_coverage_counts(code_list)
        return (counts >= 1).all()

    # SA: try removing codewords, accepting worse solutions with probability
    sa_code = code[:]
    sa_temp = 50.0
    sa_cooling = 0.97
    sa_min_temp = 0.5
    best_sa_code = sa_code[:]
    best_sa_len = len(sa_code)
    rng = np.random.default_rng(42)

    sa_start = time.time()
    while sa_temp > sa_min_temp and time.time() < deadline - 5.0:
        # Try removing a random codeword
        if len(sa_code) > 1:
            k = int(rng.integers(0, len(sa_code)))
            trial = sa_code[:k] + sa_code[k + 1:]
            if is_valid(trial):
                sa_code = trial
                if len(sa_code) < best_sa_len:
                    best_sa_len = len(sa_code)
                    best_sa_code = sa_code[:]
            else:
                # Accept with probability exp(-delta/temp), delta = 1 (size increased by 1 effectively)
                if rng.random() < np.exp(-1.0 / sa_temp):
                    sa_code = trial

        # Try swapping a codeword with another
        if len(sa_code) > 1 and time.time() < deadline - 5.0:
            k = int(rng.integers(0, len(sa_code)))
            old_word = tuple(sa_code[k])
            old_idx = word_to_idx[old_word]

            # Find candidate replacements
            # Prefer words that cover essential words of the removed codeword
            counts = compute_coverage_counts(sa_code)
            essential = (counts == 1) & ball_masks[old_idx]
            essential_indices = np.where(essential)[0]

            if len(essential_indices) == 0:
                # Can just remove
                trial = sa_code[:k] + sa_code[k + 1:]
                sa_code = trial
                if len(sa_code) < best_sa_len:
                    best_sa_len = len(sa_code)
                    best_sa_code = sa_code[:]
            else:
                # Find a word that covers all essential words
                candidates = []
                for i in range(total_words):
                    if used[i] or i == old_idx:
                        continue
                    if ball_masks[i, essential_indices].all():
                        candidates.append(i)

                if candidates:
                    new_idx = candidates[int(rng.integers(0, len(candidates)))]
                    trial = sa_code[:]
                    trial[k] = list(all_words[new_idx])
                    if is_valid(trial):
                        sa_code = trial
                        if len(sa_code) < best_sa_len:
                            best_sa_len = len(sa_code)
                            best_sa_code = sa_code[:]

        sa_temp *= sa_cooling

    if len(best_sa_code) < len(code):
        code = best_sa_code

    # Final polish: greedy removal pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        counts = compute_coverage_counts(code)
        for k in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            c_idx = word_to_idx[tuple(code[k])]
            if (counts[ball_masks[c_idx]] == 1).any():
                continue
            # Safe to remove
            code.pop(k)
            improved = True
            break

    # Verify and return
    final_counts = compute_coverage_counts(code)
    if not (final_counts >= 1).all():
        # Fallback: should not happen, but safety
        code = code  # Return as is

    return code
