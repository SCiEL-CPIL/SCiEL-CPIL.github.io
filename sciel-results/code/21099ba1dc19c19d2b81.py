import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s

    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    num_triples_all = len(all_triples)

    # Precompute pair keys for each triple
    triple_pairs = {}
    for t in all_triples:
        pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pairs.add(key)
        triple_pairs[t] = pairs

    # For each pair, the unique triple containing it
    pair_to_triple = {}
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triple[p] = t

    # For each point, triples containing it
    point_to_triples = {p: [] for p in points}
    for t in all_triples:
        for p in t:
            point_to_triples[p].append(t)

    # Precompute bitmask for each triple
    triple_masks = {}
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[t] = m

    def get_alpha_fast(triples):
        """
        Fast branch-and-bound independence number.
        Returns the size of the largest subset of points containing no triple.
        """
        if not triples:
            return n

        t_masks = [triple_masks[t] for t in triples]
        num_t = len(t_masks)

        # For each point, bitmask of triples containing it
        point_tmask = [0] * n
        for i, tm in enumerate(t_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_tmask[p] |= (1 << i)

        # Sort points by degree (number of triples containing them) descending
        # Points in more triples are "more constrained"
        order = sorted(range(n), key=lambda p: bin(point_tmask[p]).count('1'), reverse=True)

        # B&B state: (idx, current_mask, current_count)
        # idx: index in order we're considering
        # We decide for each point in order: include or not
        # Upper bound: current_count + (n - idx) remaining points
        # Prune if upper bound <= best

        best = [0]
        # Try to find a large independent set via greedy first
        # Greedy: include points in order if they don't complete any triple
        greedy_mask = 0
        greedy_count = 0
        for p in order:
            # Check if adding p to greedy_mask creates any triple
            new_mask = greedy_mask | (1 << p)
            ok = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    ok = False
                    break
            if ok:
                greedy_mask = new_mask
                greedy_count += 1
        best[0] = greedy_count

        # B&B
        # Use iterative approach with stack to avoid recursion depth issues
        # State: (pos, mask, count)
        # pos: next index in order to consider
        stack = [(0, 0, 0)]
        while stack:
            pos, mask, count = stack.pop()
            # Upper bound
            if count + (n - pos) <= best[0]:
                continue
            if pos == n:
                if count > best[0]:
                    best[0] = count
                continue

            p = order[pos]
            # Option 1: don't include p
            stack.append((pos + 1, mask, count))

            # Option 2: include p (if valid)
            new_mask = mask | (1 << p)
            valid = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    valid = False
                    break
            if valid:
                # Upper bound check for this branch
                if count + 1 + (n - pos - 1) > best[0]:
                    stack.append((pos + 1, new_mask, count + 1))

        return best[0]

    def get_alpha_with_set(triples):
        """Returns (alpha, max_independent_set) where max_independent_set is a set of points."""
        if not triples:
            return n, set(points)

        t_masks = [triple_masks[t] for t in triples]

        order = sorted(range(n), key=lambda p: sum(1 for tm in t_masks if tm & (1 << p)), reverse=True)

        best = [0]
        best_set = [set()]

        # Greedy first
        greedy_mask = 0
        greedy_count = 0
        for p in order:
            new_mask = greedy_mask | (1 << p)
            ok = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    ok = False
                    break
            if ok:
                greedy_mask = new_mask
                greedy_count += 1
        best[0] = greedy_count
        best_set[0] = set(p for p in range(n) if greedy_mask & (1 << p))

        # B&B with set tracking
        stack = [(0, 0, 0, frozenset())]
        while stack:
            pos, mask, count, current_set = stack.pop()
            if count + (n - pos) <= best[0]:
                continue
            if pos == n:
                if count > best[0]:
                    best[0] = count
                    best_set[0] = set(current_set)
                continue

            p = order[pos]
            # Don't include
            stack.append((pos + 1, mask, count, current_set))

            # Include
            new_mask = mask | (1 << p)
            valid = True
            for tm in t_masks:
                if (new_mask & tm) == tm:
                    valid = False
                    break
            if valid and count + 1 + (n - pos - 1) > best[0]:
                stack.append((pos + 1, new_mask, count + 1, current_set | {p}))

        return best[0], best_set[0]

    def is_valid_addition(triple, used_pairs):
        """Check if triple can be added without violating linearity."""
        for p in triple_pairs[triple]:
            if p in used_pairs:
                return False
        return True

    def greedy_construct():
        triples = []
        used_pairs = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            if is_valid_addition(t, used_pairs):
                triples.append(t)
                used_pairs.update(triple_pairs[t])
        return triples

    def local_search(triples, deadline):
        """Local search with targeted moves."""
        if not triples:
            return triples, n, set(points)

        current_alpha, current_iset = get_alpha_with_set(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        best_iset = current_iset

        used_pairs = set()
        for t in triples:
            used_pairs.update(triple_pairs[t])

        current = triples[:]
        cur_alpha = current_alpha
        cur_iset = current_iset

        max_steps = 3000
        temperature = 1.0
        cooling = 0.99

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if step % 50 == 0:
                temperature *= cooling

            # Find candidate triples that can be added
            candidates = []
            for t in all_triples:
                if t in current:
                    continue
                if is_valid_addition(t, used_pairs):
                    candidates.append(t)

            if not candidates:
                break

            # Move type 1: targeted add - add a triple that hits the independent set
            # Move type 2: swap - remove one, add another
            move_type = random.random()

            if move_type < 0.4 and cur_iset:
                # Targeted: find triples that intersect cur_iset in 2+ points
                # (adding such a triple would reduce independence)
                # Actually, we want to add a triple that "hits" the independent set
                # A triple hits S if it shares at least one point with S
                # But adding a triple only reduces alpha if it's a subset of some max IS
                # Better: find triples that share 2 points with cur_iset
                targeted = []
                for t in candidates:
                    overlap = len(set(t) & cur_iset)
                    if overlap >= 2:
                        targeted.append(t)
                if targeted:
                    added = random.choice(targeted)
                else:
                    added = random.choice(candidates)
                new_triples = current + [added]
                new_used_pairs = used_pairs | triple_pairs[added]
            else:
                # Swap: remove a random triple, add a candidate
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                # Recompute used_pairs without removed
                new_used_pairs = set()
                for t in current:
                    if t != removed:
                        new_used_pairs.update(triple_pairs[t])
                # Find valid candidates
                valid_cands = []
                for t in candidates:
                    if t == removed:
                        continue
                    if is_valid_addition(t, new_used_pairs):
                        valid_cands.append(t)
                if not valid_cands:
                    continue
                added = random.choice(valid_cands)
                new_triples = current[:idx] + current[idx+1:] + [added]
                new_used_pairs = new_used_pairs | triple_pairs[added]

            new_alpha, new_iset = get_alpha_with_set(new_triples)
            delta = new_alpha - cur_alpha

            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                cur_iset = new_iset
                used_pairs = new_used_pairs
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
                    best_iset = cur_iset
            else:
                # Accept worse with probability
                if temperature > 0.01:
                    prob = pow(2.718281828, -delta / max(temperature, 0.01))
                    if random.random() < prob:
                        current = new_triples
                        cur_alpha = new_alpha
                        cur_iset = new_iset
                        used_pairs = new_used_pairs

            if best_alpha <= 4:
                break

        return best_triples, best_alpha, best_iset

    # Main
    best_triples = []
    best_alpha = n
    best_iset = set(points)

    # Multiple restarts
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break

        triples = greedy_construct()
        if not triples:
            continue

        triples, alpha, iset = local_search(triples, deadline - 3)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            best_iset = iset

        if best_alpha <= 4:
            break

    # Final polish: if we have time, do more targeted improvements
    while time.time() < deadline - 2:
        if not best_triples:
            break

        used_pairs = set()
        for t in best_triples:
            used_pairs.update(triple_pairs[t])

        candidates = []
        for t in all_triples:
            if t in best_triples:
                continue
            if is_valid_addition(t, used_pairs):
                candidates.append(t)

        if not candidates:
            break

        # Try adding triples that hit the best IS
        improved = False
        for t in candidates:
            if time.time() > deadline - 2:
                break
            overlap = len(set(t) & best_iset)
            if overlap >= 2:
                new_triples = best_triples + [t]
                new_alpha, new_iset = get_alpha_with_set(new_triples)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples
                    best_iset = new_iset
                    improved = True
                    used_pairs.update(triple_pairs[t])
                    # Recompute candidates
                    candidates = []
                    for tt in all_triples:
                        if tt in best_triples:
                            continue
                        if is_valid_addition(tt, used_pairs):
                            candidates.append(tt)
                    break

        if not improved:
            # Try a few random adds
            random.shuffle(candidates)
            for t in candidates[:5]:
                if time.time() > deadline - 2:
                    break
                new_triples = best_triples + [t]
                new_alpha, new_iset = get_alpha_with_set(new_triples)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples
                    best_iset = new_iset
                    used_pairs.update(triple_pairs[t])
                    break

    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    used_pairs = set()
    valid = True
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print(f"INVALID: Shared pair {key} in {t}")
                    valid = False
                used_pairs.add(key)
    if valid:
        print("Valid linear system")
