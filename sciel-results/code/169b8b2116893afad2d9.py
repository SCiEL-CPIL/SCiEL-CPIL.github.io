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
    triple_set = set(all_triples)

    def get_alpha_fast(triples):
        """Compute independence number: largest subset with no triple entirely in it."""
        if not triples:
            return n

        # Convert triples to bitmasks
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)

        num_triples = len(t_masks)

        # For small systems, use direct approach: check all subsets
        # alpha = max |S| such that no triple is subset of S
        # Equivalently: n - min hitting set

        # Use branch-and-bound for minimum hitting set
        # A hitting set hits every triple (contains at least one point from each)
        # alpha = n - min_hitting_set_size

        # Precompute: for each point, which triples it hits (as bitmask)
        point_hit = [0] * n
        for idx, tm in enumerate(t_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_hit[p] |= (1 << idx)

        # Initial upper bound: greedy hitting set
        remaining = (1 << num_triples) - 1
        greedy_size = 0
        temp_remaining = remaining
        while temp_remaining:
            # Pick point that hits most remaining triples
            best_p = -1
            best_count = -1
            for p in range(n):
                cnt = bin(temp_remaining & point_hit[p]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_p = p
            if best_count == 0:
                break
            temp_remaining &= ~point_hit[best_p]
            greedy_size += 1

        best = [greedy_size]

        # Branch and bound
        def solve(rem_mask, cur_size):
            if cur_size >= best[0]:
                return
            if rem_mask == 0:
                best[0] = cur_size
                return

            # Pick a triple with fewest available points
            # Find first set bit
            first_bit = rem_mask & (-rem_mask)
            first_idx = first_bit.bit_length() - 1
            first_triple = t_masks[first_idx]

            # Get points in this triple
            pts_in_triple = [p for p in range(n) if first_triple & (1 << p)]

            # Order points by how many remaining triples they hit (descending)
            pts_in_triple.sort(key=lambda p: bin(rem_mask & point_hit[p]).count('1'), reverse=True)

            for p in pts_in_triple:
                new_rem = rem_mask & ~point_hit[p]
                solve(new_rem, cur_size + 1)
                if best[0] <= cur_size + 1:
                    return

        solve((1 << num_triples) - 1, 0)
        return n - best[0]

    def is_linear(triples):
        """Check if any pair is repeated."""
        seen_pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in seen_pairs:
                        return False
                    seen_pairs.add(key)
        return True

    def local_search(triples, deadline):
        """Try to improve alpha by swapping triples."""
        if not triples:
            return triples, n

        current_alpha = get_alpha_fast(triples)
        best_alpha = current_alpha
        best_triples = triples[:]

        # Precompute used pairs
        def compute_used_pairs(trip_list):
            used = set()
            for t in trip_list:
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        used.add(key)
            return used

        used_pairs = compute_used_pairs(triples)

        # Candidate triples: those with no pair conflict with current set
        def get_valid_candidates(used_pairs_set):
            cands = []
            for t in all_triples:
                if t in triple_set and t in [tuple(x) for x in triples]:
                    continue
                can_add = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        if key in used_pairs_set:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    cands.append(t)
            return cands

        # Simulated annealing
        temperature = 1.0
        cooling = 0.997
        current = triples[:]
        cur_alpha = current_alpha

        max_steps = 15000
        for step in range(max_steps):
            if time.time() > deadline:
                break

            if step % 30 == 0:
                temperature *= cooling

            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]

            # Compute used pairs without the removed triple
            new_used_pairs = compute_used_pairs([t for i, t in enumerate(current) if i != idx])

            # Find valid candidates to add
            valid_cands = []
            for t in all_triples:
                if t == removed:
                    continue
                if t in [tuple(x) for x in current]:
                    continue
                can_add = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        if key in new_used_pairs:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    valid_cands.append(t)

            if not valid_cands:
                continue

            added = random.choice(valid_cands)
            new_triples = current[:idx] + current[idx + 1:] + [added]
            new_alpha = get_alpha_fast(new_triples)

            # Accept if better, or with probability exp(-delta/temperature)
            delta = new_alpha - cur_alpha
            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
            else:
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current = new_triples
                    cur_alpha = new_alpha

        # Final polish: try to add more triples if it doesn't increase alpha
        while time.time() < deadline - 2:
            used_pairs_final = compute_used_pairs(best_triples)
            added_any = False
            for t in all_triples:
                if t in [tuple(x) for x in best_triples]:
                    continue
                can_add = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        if key in used_pairs_final:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    test_triples = best_triples + [t]
                    test_alpha = get_alpha_fast(test_triples)
                    if test_alpha <= best_alpha:
                        best_triples = test_triples
                        best_alpha = test_alpha
                        used_pairs_final = compute_used_pairs(best_triples)
                        added_any = True
                        break

            if not added_any:
                break

        return best_triples, best_alpha

    # Main: multiple random restarts with local search
    best_triples = []
    best_alpha = n

    def greedy_construct_smart():
        """Greedy construction that prioritizes triples covering high-degree points."""
        triples = []
        used_pairs = set()
        point_count = [0] * n

        shuffled = all_triples[:]
        random.shuffle(shuffled)

        for t in shuffled:
            valid = True
            for i in range(3):
                for j in range(i + 1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                triples.append(t)
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        used_pairs.add(key)
                for p in t:
                    point_count[p] += 1

        return triples

    # Run greedy + local search multiple times
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break

        # Greedy start
        triples = greedy_construct_smart()
        if not triples:
            continue

        # Local search
        triples, alpha = local_search(triples, deadline)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]

        if best_alpha <= 4:
            break

    # If we still have time, try more restarts
    while time.time() < deadline - 10:
        triples = greedy_construct_smart()
        if not triples:
            continue
        triples, alpha = local_search(triples, deadline)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]

    # Convert to list of lists
    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    # Verify linearity
    used_pairs = set()
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    break
                used_pairs.add(key)
    else:
        print("Valid linear system")
