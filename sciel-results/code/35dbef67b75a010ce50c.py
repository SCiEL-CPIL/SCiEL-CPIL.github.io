import itertools
import time
import random

def construct(q, n, radius):
    # This solution is specialized for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to tetracode direct sum for other cases (though task is fixed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(itertools.product(range(3), repeat=n))[0]] # Dummy

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute neighbors for all 729 words
    neighbors = []
    for c in range(729):
        digs = [(c // (3**i)) % 3 for i in range(6)]
        ns = [c]
        for i in range(6):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                weight = 3**i
                old_val = digs[i] * weight
                new_val = new_dig * weight
                ns.append(c - old_val + new_val)
        neighbors.append(ns)

    # Precompute neighbor sets as frozensets for fast intersection
    neighbor_sets = [frozenset(ns) for ns in neighbors]

    # Initial set: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codes = []
    for h in tetracode:
        for t in itertools.product(range(3), repeat=2):
            code_digits = h + list(t)
            val = 0
            for i, d in enumerate(code_digits):
                val += d * (3**i)
            initial_codes.append(val)

    def make_state(codes):
        """Create a fresh mutable state from a list of codeword indices."""
        code_set = set(codes)
        current_codes = list(code_set)
        cov = [0] * 729
        coverers = [set() for _ in range(729)]
        for c in current_codes:
            for w in neighbors[c]:
                cov[w] += 1
                coverers[w].add(c)
        return current_codes, code_set, cov, coverers

    def remove_code(state, c):
        current_codes, code_set, cov, coverers = state
        for w in neighbors[c]:
            cov[w] -= 1
            coverers[w].discard(c)
        code_set.discard(c)
        if c in current_codes:
            current_codes.remove(c)

    def add_code(state, c):
        current_codes, code_set, cov, coverers = state
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
        code_set.add(c)
        current_codes.append(c)

    def is_removable(state, c):
        cov = state[2]
        for w in neighbors[c]:
            if cov[w] == 1:
                return False
        return True

    def try_removals(state):
        """Remove all removable codewords. Returns number removed."""
        current_codes, code_set, cov, coverers = state
        removed_count = 0
        changed = True
        while changed:
            changed = False
            for c in current_codes[:]:
                removable = True
                for w in neighbors[c]:
                    if cov[w] == 1:
                        removable = False
                        break
                if removable:
                    for w in neighbors[c]:
                        cov[w] -= 1
                        coverers[w].discard(c)
                    code_set.discard(c)
                    if c in current_codes:
                        current_codes.remove(c)
                    removed_count += 1
                    changed = True
        return removed_count

    def try_swaps(state, max_candidates=200):
        """Try to find a beneficial swap. Returns True if a swap was made."""
        current_codes, code_set, cov, coverers = state
        # Find codewords with exclusive words
        candidates_for_swap = []
        for c in current_codes:
            exclusive = []
            for w in neighbors[c]:
                if cov[w] == 1:
                    exclusive.append(w)
            if exclusive:
                candidates_for_swap.append((len(exclusive), c, exclusive))

        if not candidates_for_swap:
            return False

        # Sort by size of exclusive set (smallest first)
        candidates_for_swap.sort(key=lambda x: x[0])

        for size, c, exclusive in candidates_for_swap:
            # Find codewords that cover ALL exclusive words
            common = set(coverers[exclusive[0]])
            for w in exclusive[1:]:
                common &= coverers[w]
                if not common:
                    break

            if not common:
                continue

            common -= code_set
            if not common:
                continue

            # Evaluate candidates
            best_new = None
            best_gain = -1
            candidate_list = list(common)
            if len(candidate_list) > max_candidates:
                # Prioritize candidates that cover more low-coverage words
                def score_candidate(nc):
                    s = 0
                    for w in neighbors[nc]:
                        if cov[w] == 0:
                            s += 10
                        elif cov[w] == 1:
                            s += 5
                        elif cov[w] == 2:
                            s += 1
                    return s
                candidate_list.sort(key=score_candidate, reverse=True)
                candidate_list = candidate_list[:max_candidates]

            for new_c in candidate_list:
                gain = 0
                for w in neighbors[new_c]:
                    if cov[w] <= 2:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_c

            if best_new is not None:
                # Perform swap
                for w in neighbors[c]:
                    cov[w] -= 1
                    coverers[w].discard(c)
                code_set.discard(c)
                if c in current_codes:
                    current_codes.remove(c)

                for w in neighbors[best_new]:
                    cov[w] += 1
                    coverers[w].add(best_new)
                code_set.add(best_new)
                current_codes.append(best_new)
                return True

        return False

    def full_polish(state):
        """Run removals and swaps until no more improvements."""
        while True:
            if time.time() > deadline - 3:
                break
            progress = False
            if try_removals(state) > 0:
                progress = True
            if try_swaps(state):
                progress = True
            if not progress:
                break

    # Multi-restart local search
    best_size = 81  # tetracode direct sum size
    best_codes = list(set(initial_codes))

    rng = random.Random(42)

    while time.time() < deadline - 5:
        # Initialize state
        state = make_state(best_codes)

        # Full polish
        full_polish(state)

        current_size = len(state[0])
        if current_size < best_size:
            best_size = current_size
            best_codes = state[0][:]

        # If we've hit the lower bound, stop
        if best_size <= 71:
            break

        # Check time
        if time.time() > deadline - 10:
            break

        # Perturb: add random codewords
        num_perturb = rng.randint(2, 6)
        for _ in range(num_perturb):
            for attempt in range(50):
                r = rng.randint(0, 728)
                if r not in state[1]:
                    add_code(state, r)
                    break

        # Polish after perturbation
        full_polish(state)

        current_size = len(state[0])
        if current_size < best_size:
            best_size = current_size
            best_codes = state[0][:]

    # Final polish on best
    if time.time() < deadline - 2:
        final_state = make_state(best_codes)
        full_polish(final_state)
        if len(final_state[0]) < best_size:
            best_size = len(final_state[0])
            best_codes = final_state[0][:]

    # Convert best_codes back to list of lists
    result = []
    for c in best_codes:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)

    return result
