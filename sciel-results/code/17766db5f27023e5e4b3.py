import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0

    total_points = q ** n          # 6561
    ball_size = 1 + n * (q - 1)    # 17
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        neighbors[idx, 0] = idx
        nb = 1
        for pos in range(n):
            orig = word[pos]
            pw = powers[pos]
            # +1
            nd = (orig + 1) % q
            delta = -2 * pw if nd == 0 else pw
            neighbors[idx, nb] = idx + delta
            nb += 1
            # -1
            nd = (orig - 1) % q
            delta = 2 * pw if nd == 2 else -pw
            neighbors[idx, nb] = idx + delta
            nb += 1

    # Reverse mapping: for each point, which balls contain it (== neighbors of that point)
    # (symmetric in Hamming space)
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        return counts

    def reduce_code(code, counts):
        current_code = list(code)
        current_counts = counts.copy()
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                if np.all(current_counts[neighbors[c]] > 1):
                    current_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    # --- Fast greedy initialization with potential tracking ---
    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        # order for tie-breaking / deterministic-ish
        order = rng.permutation(total_points)
        # map index position in order for tie-break
        while True:
            if time.time() > deadline - 35.0:
                break
            maxp = potential.max()
            if maxp <= 0:
                break
            # candidates with max potential
            cands = np.where(potential == maxp)[0]
            if len(cands) == 1:
                c = int(cands[0])
            else:
                # random tie-break
                c = int(cands[rng.integers(len(cands))])
            selected.append(c)
            potential[c] = 0
            for x in neighbors[c]:
                if not is_covered[x]:
                    is_covered[x] = True
                    # for every ball covering x, reduce potential
                    potential[neighbors[x]] -= 1
                    np.maximum(potential, 0, out=potential)
            if is_covered.all():
                break
        return selected

    # --- Defect-minimization local search with targeted swap ---
    def local_search(code, counts, deadline):
        code = list(code)
        counts = counts.copy()
        code_set = set(code)
        best_len = len(code)
        best_code = code[:]
        rng = np.random.default_rng(7)
        no_improve = 0
        max_no_improve = 8000

        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            # Pass 1: try pure removals (fast)
            for _ in range(40):
                if not code:
                    break
                if time.time() > deadline:
                    break
                c = code[rng.integers(len(code))]
                risky = neighbors[c][counts[neighbors[c]] == 1]
                if len(risky) == 0:
                    counts[neighbors[c]] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    break

            # Pass 2: targeted swap for a codeword that has risky points
            if not improved and code:
                # pick a codeword with fewest risky points (easier to cover)
                # sample a few, choose min-risky
                sample = rng.choice(len(code), size=min(20, len(code)), replace=False)
                best_c = None
                best_risky = None
                best_risky_set = None
                for s in sample:
                    c = code[s]
                    risky = neighbors[c][counts[neighbors[c]] == 1]
                    if best_risky is None or len(risky) < len(best_risky):
                        best_c = c
                        best_risky = risky
                        best_risky_set = set(risky.tolist())
                if best_risky is not None and len(best_risky) > 0:
                    # Intersect neighbor lists of risky points to find candidate
                    # that covers all risky points
                    if len(best_risky) == 1:
                        cand = neighbors[best_risky[0]]
                    else:
                        # intersection via set
                        common = set(neighbors[best_risky[0]].tolist())
                        for r in best_risky[1:]:
                            common &= set(neighbors[r].tolist())
                            if not common:
                                break
                        cand = np.array(sorted(common), dtype=np.int32)
                    if len(cand) > 0:
                        # choose candidate maximizing additional coverage (points with count 0 or low)
                        # score = number of risky it covers (all) + extra uncovered/low-count
                        best_new = None
                        best_score = -1
                        # limit candidates for speed
                        if len(cand) > 60:
                            cand = cand[rng.choice(len(cand), size=60, replace=False)]
                        for np_c in cand:
                            if np_c in code_set:
                                continue
                            # new coverage set
                            new_ball = neighbors[np_c]
                            # count how many risky it covers (should be all by construction)
                            # extra benefit: points in new_ball with count 0 (none, covered) -> use count==1 as "risky elsewhere"
                            # We want to reduce total risky count. Compute risky in new_ball:
                            # points in new_ball that would become count 0 after swap are bad.
                            # After swap: counts change by -1 on neighbors[best_c], +1 on new_ball.
                            # A point p in new_ball with current count 1 would become 2 (good, was risky of best_c? no)
                            # A point p in neighbors[best_c] not in new_ball with count 1 becomes 0 (BAD - new uncovered)
                            # So we must ensure all points in neighbors[best_c] with count==1 are in new_ball (== risky covered)
                            # risky = exactly those count==1 points in neighbors[best_c]; we required new covers all risky. Good.
                            # Additional benefit: new_ball may cover other count==1 points elsewhere? No, those are risky of their own codewords,
                            # but covering them with an extra copy is fine (count 1->2). Doesn't hurt.
                            # So any valid candidate works; prefer one that is "central" (covers many count==1 points to be robust).
                            cnt1 = np.sum(counts[new_ball] == 1)
                            if cnt1 > best_score:
                                best_score = cnt1
                                best_new = int(np_c)
                        if best_new is not None:
                            counts[neighbors[best_c]] -= 1
                            counts[neighbors[best_new]] += 1
                            code.remove(best_c)
                            code.append(best_new)
                            code_set.discard(best_c)
                            code_set.add(best_new)
                            improved = True

            if improved:
                no_improve = 0
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
            else:
                no_improve += 1
                # periodic full reduce to escape
                if no_improve % 30 == 0:
                    new_code, new_counts = reduce_code(code, counts)
                    if len(new_code) < len(code):
                        code = new_code
                        counts = new_counts
                        code_set = set(code)
                        no_improve = 0
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]

        return best_code, get_counts(best_code)

    # --- Main ---
    best_code = None
    best_len = float('inf')

    # Multiple greedy seeds, keep best after reduce
    for seed in [1, 2, 3]:
        if time.time() > deadline - 45.0:
            break
        sel = fast_greedy(seed)
        counts = get_counts(sel)
        sel, counts = reduce_code(sel, counts)
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel
            best_counts = counts

    if best_code is None:
        sel = fast_greedy(1)
        best_code = sel
        best_counts = get_counts(sel)

    # Local search from best
    best_code, best_counts = local_search(best_code, best_counts, deadline)

    # Final reduce
    best_code, best_counts = reduce_code(best_code, best_counts)

    # Convert to words
    result = [idx_to_word(int(idx)) for idx in best_code]
    return result
