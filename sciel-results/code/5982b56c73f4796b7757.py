"""Linear triple systems on 16 points with small independence number."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    random.seed(42)

    N = 16
    FULL_MASK = (1 << N) - 1
    ALL_TRIPLES = list(itertools.combinations(range(N), 3))
    num_triples = len(ALL_TRIPLES)

    # Precompute triple bitmasks
    triple_masks = []
    for t in ALL_TRIPLES:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)

    # Precompute conflict: conflict[i] is a bitmask of triples j such that i and j share >= 2 points
    # For speed, we build a list of lists of conflicting indices
    # Two triples conflict if they share a pair.
    # We can precompute for each pair of points, which triples contain it.
    pair_to_triples = {}
    for i, t in enumerate(ALL_TRIPLES):
        for k in range(3):
            for l in range(k+1, 3):
                p = (min(t[k], t[l]), max(t[k], t[l]))
                if p not in pair_to_triples:
                    pair_to_triples[p] = []
                pair_to_triples[p].append(i)

    # Build conflict list: conflict[i] = set of j != i that conflict with i
    conflict = [set() for _ in range(num_triples)]
    for p, tlist in pair_to_triples.items():
        for idx_a in range(len(tlist)):
            for idx_b in range(idx_a + 1, len(tlist)):
                a = tlist[idx_a]
                b = tlist[idx_b]
                conflict[a].add(b)
                conflict[b].add(a)

    def independence_number(triples):
        """Find the largest subset of {0..15} containing no triple. Exact via bitmask DP."""
        if not triples:
            return N

        # Build a 'bad' array: bad[S] = True if S contains any triple
        # Use the supersets marking technique for speed
        bad = bytearray(1 << N)  # 65536 bytes, very fast to allocate and check
        for t in triples:
            m = triple_masks[ALL_TRIPLES.index(t)] if t in ALL_TRIPLES else ((1 << t[0]) | (1 << t[1]) | (1 << t[2]))
            # Actually, let's just compute mask directly to avoid index lookup
            # But t is always from ALL_TRIPLES or generated. Let's just compute.
            # For speed in the inner loop of search, we should pass masks.
            # But for now, correctness first.
            pass

        # More efficient: precompute masks for the specific triples list
        tmasks = []
        for t in triples:
            tmasks.append((1 << t[0]) | (1 << t[1]) | (1 << t[2]))

        bad = bytearray(1 << N)
        for m in tmasks:
            free = FULL_MASK ^ m
            sub = free
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & free

        best = 0
        for S in range(1 << N):
            if not bad[S]:
                pc = S.bit_count()
                if pc > best:
                    best = pc
        return best

    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            ti = set(triples[i])
            for j in range(i + 1, len(triples)):
                tj = set(triples[j])
                if len(ti & tj) > 1:
                    return False
        return True

    def random_greedy_packing(seed_offset=0):
        """Build a dense linear triple system via randomized greedy packing."""
        rng = random.Random(42 + seed_offset)
        used_pairs = set()
        triples = []

        candidates = list(range(num_triples))
        rng.shuffle(candidates)

        for idx in candidates:
            t = ALL_TRIPLES[idx]
            pairs = [
                frozenset((t[0], t[1])),
                frozenset((t[0], t[2])),
                frozenset((t[1], t[2]))
            ]
            can_add = True
            for p in pairs:
                if p in used_pairs:
                    can_add = False
                    break
            if can_add:
                for p in pairs:
                    used_pairs.add(p)
                triples.append(t)

        return triples

    def local_search_sa(triples, max_time=60):
        """
        Simulated Annealing local search.
        State: set of triples (linear).
        Moves:
          1. Add a random valid triple.
          2. Remove a random triple.
          3. Swap: remove a random triple, add a different random valid triple.
        Energy: independence_number.
        We want to minimize energy.
        """
        rng = random.Random(1000)
        current_triples = triples[:]
        current_score = independence_number(current_triples)

        best_triples = current_triples[:]
        best_score = current_score

        timeout = time.time() + max_time

        # Temperature schedule
        T0 = 5.0
        Tend = 0.01
        # Estimate number of steps we can do.
        # Each step involves independence_number which is ~0.05-0.1s?
        # Let's measure.
        # 2^16 = 65536 iterations in python loop is ~0.01s.
        # So independence_number is fast.
        # We can do many steps.

        steps = 0
        while time.time() < timeout:
            steps += 1
            # Adaptive temperature
            # Simple linear decay or exponential
            # Let's use a fixed number of steps per temperature level
            # Or just use a simple cooling schedule based on steps
            T = max(Tend, T0 * (0.999 ** steps))

            # Generate a move
            move_type = rng.random()
            new_triples = current_triples[:]
            valid_move = False

            if move_type < 0.4:
                # Add a random triple
                # Pick a random triple index
                idx = rng.randint(0, num_triples - 1)
                t = ALL_TRIPLES[idx]
                # Check if it can be added (no conflict)
                # We need to check against all current triples
                # For speed, we can check pairs
                can_add = True
                t_set = set(t)
                for ct in current_triples:
                    if len(t_set & set(ct)) > 1:
                        can_add = False
                        break
                if can_add:
                    new_triples.append(t)
                    valid_move = True
                else:
                    # Try to find a valid triple to add
                    # Sample a few
                    for _ in range(10):
                        idx = rng.randint(0, num_triples - 1)
                        t = ALL_TRIPLES[idx]
                        t_set = set(t)
                        can_add = True
                        for ct in current_triples:
                            if len(t_set & set(ct)) > 1:
                                can_add = False
                                break
                        if can_add:
                            new_triples.append(t)
                            valid_move = True
                            break

            elif move_type < 0.7:
                # Remove a random triple
                if len(current_triples) > 0:
                    idx = rng.randint(0, len(current_triples) - 1)
                    new_triples.pop(idx)
                    valid_move = True
            else:
                # Swap: remove one, add another
                if len(current_triples) > 0:
                    idx = rng.randint(0, len(current_triples) - 1)
                    removed = new_triples.pop(idx)
                    # Find a replacement
                    added = False
                    for _ in range(20):
                        nidx = rng.randint(0, num_triples - 1)
                        t = ALL_TRIPLES[nidx]
                        t_set = set(t)
                        can_add = True
                        for ct in new_triples:
                            if len(t_set & set(ct)) > 1:
                                can_add = False
                                break
                        if can_add:
                            new_triples.append(t)
                            added = True
                            valid_move = True
                            break
                    if not added:
                        # Revert removal
                        new_triples.append(removed)
                        valid_move = False

            if not valid_move:
                continue

            new_score = independence_number(new_triples)
            delta = new_score - current_score

            if delta < 0 or (T > 0 and rng.random() < 2.71828 ** (-delta / T)):
                current_triples = new_triples
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_triples = current_triples[:]

        return best_triples, best_score

    def targeted_local_search(triples, max_time=60):
        """Targeted search: find large independent sets and add triples to break them."""
        rng = random.Random(2000)
        best_triples = triples[:]
        best_score = independence_number(best_triples)

        current_triples = triples[:]
        current_score = best_score

        timeout = time.time() + max_time

        for iteration in range(10000):
            if time.time() > timeout:
                break

            # Find a large independent set
            # Use a heuristic: greedy maximal independent set
            # Or brute force if current_score is small
            # Let's use a simple greedy to find *a* large independent set
            # Start with all points, remove points that form triples? No.
            # Greedy: add points one by one if they don't complete a triple.
            # This finds a maximal independent set, not necessarily maximum.
            # But it's fast.

            # Better: use the exact independence_number but also return the set?
            # Let's just use a random subset of size current_score to find an independent set.
            # If we find one, we try to break it.

            alpha = current_score
            if alpha < 4:
                break

            found_iset = None
            # Try random subsets of size alpha
            for attempt in range(50):
                subset = set(rng.sample(range(N), alpha))
                is_independent = True
                for t in current_triples:
                    if set(t).issubset(subset):
                        is_independent = False
                        break
                if is_independent:
                    found_iset = subset
                    break

            if found_iset is None:
                # Try smaller
                for size in range(alpha - 1, 3, -1):
                    for attempt in range(20):
                        subset = set(rng.sample(range(N), size))
                        is_independent = True
                        for t in current_triples:
                            if set(t).issubset(subset):
                                is_independent = False
                                break
                        if is_independent:
                            found_iset = subset
                            break
                    if found_iset:
                        break

            if found_iset is None:
                continue

            # Try to add a triple that is a subset of found_iset
            iset_list = list(found_iset)
            candidates = list(itertools.combinations(iset_list, 3))
            rng.shuffle(candidates)

            added = False
            for new_triple in candidates:
                # Check if it conflicts with existing triples
                valid = True
                nt_set = set(new_triple)
                for t in current_triples:
                    if len(nt_set & set(t)) > 1:
                        valid = False
                        break
                if valid:
                    current_triples.append(new_triple)
                    new_score = independence_number(current_triples)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_triples = current_triples[:]
                    added = True
                    break

            if not added:
                # Try removing a random triple and adding a better one
                if len(current_triples) > 5:
                    idx = rng.randint(0, len(current_triples) - 1)
                    removed = current_triples.pop(idx)

                    for new_triple in candidates:
                        valid = True
                        nt_set = set(new_triple)
                        for t in current_triples:
                            if len(nt_set & set(t)) > 1:
                                valid = False
                                break
                        if valid:
                            current_triples.append(new_triple)
                            new_score = independence_number(current_triples)
                            if new_score < current_score:
                                current_score = new_score
                                if new_score < best_score:
                                    best_score = new_score
                                    best_triples = current_triples[:]
                            break
                    else:
                        current_triples.append(removed)

        return best_triples, best_score

    # Phase 1: Generate initial systems with different seeds
    best_triples = None
    best_score = N + 1

    for seed_offset in range(20):
        if time.time() > deadline - 30:
            break
        initial = random_greedy_packing(seed_offset)
        if not initial:
            continue
        score = independence_number(initial)
        if score < best_score:
            best_score = score
            best_triples = initial[:]

    if best_triples is None:
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        best_score = independence_number(best_triples)

    # Phase 2: Local search (Simulated Annealing)
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = local_search_sa(best_triples, max_time=min(remaining_time, 60))

    # Phase 3: Targeted local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = targeted_local_search(best_triples, max_time=remaining_time)

    # Final check
    if not is_linear(best_triples):
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    return [list(t) for t in best_triples]
