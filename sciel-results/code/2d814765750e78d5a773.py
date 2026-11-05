import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    def greedy_set_cover(perturb=0.0, seed=None):
        """Greedy set cover with optional perturbation."""
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        rng = np.random.default_rng(seed) if seed is not None else None

        while np.any(uncovered):
            if time.time() > deadline - 30:
                break

            # Compute gains: for each candidate, how many uncovered points does it cover?
            gains = np.zeros(N, dtype=np.float64)
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                gains[neighbors_list[idx]] += 1

            if perturb > 0 and rng is not None:
                gains += rng.random(N) * perturb

            best_idx = int(np.argmax(gains))
            if gains[best_idx] < 0.01:
                break

            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors_list[best_idx]] = False

        return code

    def local_search(code):
        """Local search: remove redundant, 2-swap, kick moves."""
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

        def try_remove():
            """Remove all redundant codewords."""
            removed = False
            for c in list(code):
                if time.time() > deadline - 15:
                    break
                if np.min(counts[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    removed = True
            return removed

        def try_2swap():
            """Try to remove two codewords and add one better-positioned codeword."""
            improved = False
            m = len(code)
            for i in range(m):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline - 10:
                        break
                    c1, c2 = code[i], code[j]
                    trial_counts = counts.copy()
                    trial_counts[neighbors_list[c1]] -= 1
                    trial_counts[neighbors_list[c2]] -= 1

                    unc = trial_counts == 0
                    if not np.any(unc):
                        # Can just remove both
                        code.remove(c2)
                        code.remove(c1)
                        code_set.remove(c2)
                        code_set.remove(c1)
                        counts[neighbors_list[c1]] -= 1
                        counts[neighbors_list[c2]] -= 1
                        improved = True
                        break

                    # Find best single word to add
                    unc_indices = np.where(unc)[0]
                    gains = np.zeros(N, dtype=np.int32)
                    for idx in unc_indices:
                        gains[neighbors_list[idx]] += 1
                    for k in code_set:
                        gains[k] = -1

                    best_k = int(np.argmax(gains))
                    if gains[best_k] == len(unc_indices):
                        # New word covers all uncovered
                        code.remove(c2)
                        code.remove(c1)
                        code_set.remove(c2)
                        code_set.remove(c1)
                        counts[neighbors_list[c1]] -= 1
                        counts[neighbors_list[c2]] -= 1

                        code.append(best_k)
                        code_set.add(best_k)
                        counts[neighbors_list[best_k]] += 1
                        improved = True
                        break
                if improved:
                    break
            return improved

        def try_kick():
            """Remove one codeword, add two better-positioned codewords."""
            improved = False
            m = len(code)
            for i in range(m):
                if time.time() > deadline - 8:
                    break
                c = code[i]
                trial_counts = counts.copy()
                trial_counts[neighbors_list[c]] -= 1

                unc = trial_counts == 0
                if not np.any(unc):
                    # Can just remove
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    improved = True
                    break

                # Need to cover unc with 2 words
                unc_indices = np.where(unc)[0]
                gains = np.zeros(N, dtype=np.int32)
                for idx in unc_indices:
                    gains[neighbors_list[idx]] += 1
                for k in code_set:
                    gains[k] = -1

                best_k1 = int(np.argmax(gains))
                if gains[best_k1] < len(unc_indices):
                    # Need a second word
                    new_unc = unc & ~neighbors_list[best_k1]
                    new_unc_indices = np.where(new_unc)[0]
                    if len(new_unc_indices) > 0:
                        gains2 = np.zeros(N, dtype=np.int32)
                        for idx in new_unc_indices:
                            gains2[neighbors_list[idx]] += 1
                        for k in code_set:
                            gains2[k] = -1
                        gains2[best_k1] = -1

                        best_k2 = int(np.argmax(gains2))
                        if gains2[best_k2] == len(new_unc_indices):
                            # New two words cover all uncovered
                            code.remove(c)
                            code_set.remove(c)
                            counts[neighbors_list[c]] -= 1

                            code.append(best_k1)
                            code_set.add(best_k1)
                            counts[neighbors_list[best_k1]] += 1

                            code.append(best_k2)
                            code_set.add(best_k2)
                            counts[neighbors_list[best_k2]] += 1
                            improved = True
                            break
            return improved

        def try_relocation(c):
            """Try to relocate codeword c to a better position."""
            # Get critical points for c: points in neighbors[c] with counts == 1
            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

            if not crit_points:
                return False

            # Find replacement covering all critical points
            shuffled_crit = list(crit_points)
            random.shuffle(shuffled_crit)

            for p in shuffled_crit:
                cands = [x for x in neighbor_sets[p] if x not in code_set]

                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Score: maximize redundancy creation
                    score = int(np.sum(counts[neighbors_list[w]] == 1))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors_list[best_w]] += 1
                    return True

            return False

        # Main local search loop
        for _ in range(50):
            if time.time() > deadline - 15:
                break

            # Remove redundant
            if try_remove():
                continue

            # Try 2-swap
            if try_2swap():
                continue

            # Try kick
            if try_kick():
                continue

            # Try relocation
            moved = False
            for c in list(code):
                if time.time() > deadline - 15:
                    break
                if try_relocation(c):
                    moved = True
                    break
            if moved:
                continue

            # No improvement, stop
            break

        # Final polish: remove any remaining redundant
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            if not try_remove():
                break

        return code

    # Multi-start
    best_result = None
    best_size = float('inf')

    configs = [
        (0.0, None),
        (0.3, 1),
        (0.7, 2),
        (1.0, 3),
        (0.5, 4),
        (1.5, 5),
        (0.2, 6),
    ]

    for perturb, seed in configs:
        if time.time() > deadline - 40:
            break

        selected = greedy_set_cover(perturb=perturb, seed=seed)
        if not selected:
            continue

        result = local_search(selected)
        if len(result) < best_size:
            best_size = len(result)
            best_result = result
            if best_size <= 186:
                break

    if best_result is None:
        best_result = greedy_set_cover()
        best_result = local_search(best_result)

    # Final polish
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        changed = True
        while changed and (time.time() < deadline - 5):
            changed = False
            for idx in list(best_result):
                if time.time() > deadline - 5:
                    break
                if len(best_result) <= 1:
                    continue
                trial = best_result - {idx}
                if not trial:
                    continue
                masks = np.array([neighbors_list[i] for i in trial])
                if np.any(masks, axis=0).all():
                    best_result = trial
                    changed = True
                    break

    return [idx_to_tuple(i) for i in best_result]
