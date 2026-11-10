import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    points = list(range(n))

    # All possible triples
    all_triples = list(combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    def compute_alpha(triples):
        """Compute independence number using bitmask subset enumeration."""
        if not triples:
            return n

        num_t = len(triples)
        if num_t == 0:
            return n

        # For each triple, store as bitmask of points
        t_point_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_point_masks.append(m)

        # point_hit[p] = bitmask of triples that contain point p
        point_hit = [0] * n
        for idx, tm in enumerate(t_point_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_hit[p] |= (1 << idx)

        # Branch and bound for minimum hitting set
        best_hs = [num_t]

        def solve(rem, cur):
            if cur >= best_hs[0]:
                return
            if rem == 0:
                best_hs[0] = cur
                return
            fb = rem & (-rem)
            fi = fb.bit_length() - 1
            ft = t_point_masks[fi]
            pts = [p for p in range(n) if ft & (1 << p)]
            pts.sort(key=lambda p: bin(rem & point_hit[p]).count('1'), reverse=True)
            for p in pts:
                new_rem = rem & ~point_hit[p]
                solve(new_rem, cur + 1)
                if best_hs[0] <= cur + 1:
                    return

        solve((1 << num_t) - 1, 0)
        return n - best_hs[0]

    def find_max_independent_set(triples):
        """Find the actual maximum independent set (not just its size)."""
        if not triples:
            return set(points)

        t_point_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_point_masks.append(m)

        num_t = len(triples)

        point_hit = [0] * n
        for idx, tm in enumerate(t_point_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_hit[p] |= (1 << idx)

        best_hs = [num_t]
        best_hs_set = [set(points)]

        def solve(rem, cur, cur_set):
            if cur >= best_hs[0]:
                return
            if rem == 0:
                best_hs[0] = cur
                best_hs_set[0] = set(cur_set)
                return
            fb = rem & (-rem)
            fi = fb.bit_length() - 1
            ft = t_point_masks[fi]
            pts = [p for p in range(n) if ft & (1 << p)]
            pts.sort(key=lambda p: bin(rem & point_hit[p]).count('1'), reverse=True)
            for p in pts:
                new_rem = rem & ~point_hit[p]
                cur_set.add(p)
                solve(new_rem, cur + 1, cur_set)
                cur_set.remove(p)
                if best_hs[0] <= cur + 1:
                    return

        solve((1 << num_t) - 1, 0, set())
        hs_set = best_hs_set[0]
        return set(points) - hs_set

    def get_used_pairs(triples):
        used = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    used.add(key)
        return used

    def can_add(triple, used_pairs, existing_triples):
        if triple in existing_triples:
            return False
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = triple[i], triple[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    return False
        return True

    def greedy_construct():
        """Greedy: add triples that cover the most uncovered pairs."""
        triples = []
        used_pairs = set()
        remaining_triples = all_triples[:]
        random.shuffle(remaining_triples)

        for t in remaining_triples:
            if can_add(t, used_pairs, set(map(tuple, triples))):
                triples.append(t)
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        used_pairs.add(key)

        return triples

    def targeted_improvement(triples, deadline):
        """
        Structural approach: identify the independent set, then find triples
        that hit it while maintaining linearity. Repeat until no improvement.
        """
        current = triples[:]
        cur_alpha = compute_alpha(current)

        while time.time() < deadline - 5:
            if cur_alpha <= 4:
                break

            ind_set = find_max_independent_set(current)

            if not ind_set:
                break

            hitting_triples = []
            current_set = set(map(tuple, current))
            used_pairs = get_used_pairs(current)

            for t in all_triples:
                t_set = set(t)
                overlap = t_set & ind_set
                if len(overlap) >= 2 and can_add(t, used_pairs, current_set):
                    hitting_triples.append(t)

            if not hitting_triples:
                for t in all_triples:
                    t_set = set(t)
                    overlap = t_set & ind_set
                    if len(overlap) == 1 and can_add(t, used_pairs, current_set):
                        hitting_triples.append(t)

            if not hitting_triples:
                break

            improved = False
            random.shuffle(hitting_triples)
            for t in hitting_triples:
                if time.time() > deadline - 5:
                    break

                test = current + [t]
                test_alpha = compute_alpha(test)
                if test_alpha < cur_alpha:
                    current = test
                    cur_alpha = test_alpha
                    current_set = set(map(tuple, current))
                    used_pairs = get_used_pairs(current)
                    improved = True
                    break

            if not improved:
                if len(current) > 0:
                    found_swap = False
                    for idx in range(len(current)):
                        if time.time() > deadline - 5:
                            break
                        removed = current[idx]
                        test = current[:idx] + current[idx+1:]
                        test_used = get_used_pairs(test)
                        test_set = set(map(tuple, test))

                        for t in all_triples:
                            t_set = set(t)
                            overlap = t_set & ind_set
                            if len(overlap) >= 2 and can_add(t, test_used, test_set):
                                new_test = test + [t]
                                new_alpha = compute_alpha(new_test)
                                if new_alpha < cur_alpha:
                                    current = new_test
                                    cur_alpha = new_alpha
                                    current_set = set(map(tuple, current))
                                    used_pairs = get_used_pairs(current)
                                    found_swap = True
                                    break
                        if found_swap:
                            break

                    if not found_swap:
                        break

        return current, cur_alpha

    def local_search_swap(triples, deadline):
        """Standard swap-based local search."""
        current = triples[:]
        cur_alpha = compute_alpha(current)
        best = current[:]
        best_alpha = cur_alpha

        max_iters = 3000
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            if len(current) < 2:
                break

            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            test = current[:idx] + current[idx+1:]
            test_used = get_used_pairs(test)
            test_set = set(map(tuple, test))

            cands = []
            for t in all_triples:
                if can_add(t, test_used, test_set):
                    cands.append(t)

            if not cands:
                continue

            added = random.choice(cands)
            new_current = test + [added]
            new_alpha = compute_alpha(new_current)

            if new_alpha <= cur_alpha:
                current = new_current
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best = current[:]

        return best, best_alpha

    best_triples = []
    best_alpha = n

    for restart in range(40):
        if time.time() > deadline - 20:
            break

        triples = greedy_construct()
        if not triples:
            continue

        triples, alpha = targeted_improvement(triples, deadline)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]

        if best_alpha <= 4:
            break

    if time.time() < deadline - 10 and best_triples:
        best_triples, best_alpha = local_search_swap(best_triples, deadline)

    while time.time() < deadline - 8:
        triples = greedy_construct()
        if not triples:
            continue
        triples, alpha = targeted_improvement(triples, deadline)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        if best_alpha <= 4:
            break

    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    used_pairs = set()
    valid = True
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    valid = False
                    break
                used_pairs.add(key)
        if not valid:
            break
    if valid:
        print("Valid linear system")
