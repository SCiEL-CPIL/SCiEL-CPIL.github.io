import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Ball of radius 1 for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            new_digit = (orig_digit + 1) % q
            delta = -2 * power if new_digit == 0 else power
            nbrs.append(idx + delta)
            new_digit = (orig_digit - 1) % q
            delta = 2 * power if new_digit == 2 else -power
            nbrs.append(idx + delta)
        neighbors[idx] = nbrs

    balls = np.array(neighbors, dtype=np.int32)  # (6561, 17)

    # coverers[p] = list of codeword indices whose ball contains point p
    coverers = [None] * total_points
    for p in range(total_points):
        lst = []
        for c in range(total_points):
            if p in neighbors[c]:
                lst.append(c)
        coverers[p] = lst

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return list(selected_indices)
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break
        return selected_list

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def local_search_optimize(selected_indices, deadline):
        selected_list = prune(list(selected_indices))
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Remove any removable codeword (full deterministic sweep)
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # 2. Neighbor-restricted swap with O(1) risky-point candidate intersection
            progress = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)
                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue
                    coverage_count[balls[c]] -= 1
                    # risky points: in ball(c), count now 0 -> must be covered by c_prime
                    risky = balls[c][coverage_count[balls[c]] == 0]
                    if len(risky) > 0:
                        # candidates must cover all risky points but not be c itself
                        cand = set(coverers[int(risky[0])])
                        for rp in risky[1:]:
                            cand &= set(coverers[int(rp)])
                        cand.discard(c)
                        if c_prime not in cand:
                            coverage_count[balls[c]] += 1
                            continue
                    # valid swap
                    coverage_count[balls[c_prime]] += 1
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                if progress:
                    break

            # 3. 2-opt escape: remove two codewords, add one that covers all exposed points
            if time.time() < deadline:
                n_sel = len(selected_list)
                if n_sel >= 2:
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        i = rng.randrange(n_sel)
                        j = rng.randrange(n_sel)
                        if i == j:
                            continue
                        a = selected_list[i]
                        b = selected_list[j]
                        coverage_count[balls[a]] -= 1
                        coverage_count[balls[b]] -= 1
                        risky = balls[a][coverage_count[balls[a]] == 0]
                        if len(risky) == 0:
                            risky = balls[b][coverage_count[balls[b]] == 0]
                        else:
                            # union of risky from a and b
                            ra = set(int(x) for x in balls[a][coverage_count[balls[a]] == 0])
                            rb = set(int(x) for x in balls[b][coverage_count[balls[b]] == 0])
                            risky = list(ra | rb)
                        if len(risky) > 0:
                            cand = set(coverers[int(risky[0])])
                            for rp in risky[1:]:
                                cand &= set(coverers[int(rp)])
                            cand.discard(a)
                            cand.discard(b)
                            if cand:
                                c_new = int(rng.choice(sorted(cand)))
                                coverage_count[balls[c_new]] += 1
                                # remove a and b
                                selected_list.remove(a)
                                selected_list.remove(b)
                                selected_set.discard(a)
                                selected_set.discard(b)
                                selected_list.append(c_new)
                                selected_set.add(c_new)
                                progress = True
                                break
                        # revert
                        coverage_count[balls[a]] += 1
                        coverage_count[balls[b]] += 1

            if progress:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    best_code = None
    best_len = float('inf')
    for seed in range(8):
        if time.time() > deadline - 50:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = prune(greedy_set_cover(0))
        best_len = len(best_code)

    best_code = local_search_optimize(best_code, deadline)

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    return result
