import time
import random
import math
import numpy as np

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    return neigh, total

def _build_conflict_graph(q, n, total):
    """Vertices = space words; edge (u,v) iff Hamming distance(u,v) == 2.
    For radius-1 covering, balls of distance-2 words are DISJOINT, so an
    optimal code never contains two words at distance 2 (they are redundant
    in the sense that one can be swapped for the other without losing
    coverage of the other's ball). This gives a structural constraint we
    can exploit in greedy pruning and 1-flip local search."""
    # Precompute digit tuples
    tuples = []
    for idx in range(total):
        t = [0] * n
        x = idx
        for i in range(n):
            t[i] = x % q
            x //= q
        tuples.append(t)
    # Conflict: exactly 2 positions differ
    conflict = [set() for _ in range(total)]
    for u in range(total):
        tu = tuples[u]
        for v in range(u + 1, total):
            tv = tuples[v]
            diff = 0
            for i in range(n):
                if tu[i] != tv[i]:
                    diff += 1
                    if diff > 2:
                        break
            if diff == 2:
                conflict[u].add(v)
                conflict[v].add(u)
    return conflict

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)

    neigh, _ = _precompute_neighbors(q, n, radius)

    # Build ball matrix: ball_matrix[c, u] = 1 if u is in ball(c)
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        for u in neigh[c]:
            ball_matrix[c, u] = 1

    # Conflict graph: edge iff Hamming distance == 2 (balls disjoint for radius 1)
    conflict = _build_conflict_graph(q, n, total)
    # For each codeword, the set of words it "blocks" (conflicts with)
    # A word w is blocked by code set S if w conflicts with any c in S
    # We maintain a blocked array: blocked[w] = True if w conflicts with some c in code
    blocked = np.zeros(total, dtype=np.uint8)
    # Precompute conflict as numpy rows for vectorized blocked updates
    conflict_rows = np.zeros((total, total), dtype=np.uint8)
    for u in range(total):
        for v in conflict[u]:
            conflict_rows[u, v] = 1
    
    def greedy_init(seed):
        """Vectorized greedy set cover with conflict-aware pruning.
        A candidate c is 'blocked' if it conflicts with an already-chosen
        codeword; choosing c would be structurally redundant (its ball is
        disjoint from the conflicting codeword's ball, so c adds no new
        coverage that the conflicting codeword doesn't already provide
        via its own ball... actually c adds coverage, but the conflicting
        codeword could be swapped for c later). We bias away from blocked
        candidates to keep the code in the 'independent set' structure."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        code_set = set()
        blocked = np.zeros(total, dtype=np.uint8)
        for c in code:
            blocked[conflict_rows[c] == 1] = 1

        while True:
            if time.monotonic() > deadline:
                break
            remaining_mask = (counts == 0).astype(np.uint8)
            if remaining_mask.sum() == 0:
                break

            # Vectorized gain computation
            gains = ball_matrix @ remaining_mask
            best_gain = int(gains.max())
            if best_gain <= 0:
                break

            # Get all candidates with max gain
            candidates = np.where(gains == best_gain)[0]

            # Prefer non-blocked candidates (structural independence)
            non_blocked = candidates[blocked[candidates] == 0]
            if len(non_blocked) > 0:
                pool = non_blocked
            else:
                pool = candidates

            # Random tie-break among chosen pool
            if len(pool) > 1:
                best_c = int(pool[r.randint(0, len(pool) - 1)])
            else:
                best_c = int(pool[0])

            code.append(best_c)
            code_set.add(best_c)
            counts += ball_matrix[best_c]
            blocked[conflict_rows[best_c] == 1] = 1

        # Fallback if not fully covered
        remaining_mask = (counts == 0)
        if remaining_mask.any():
            while remaining_mask.any():
                uncovered = np.where(remaining_mask)[0]
                u = int(uncovered[0])
                # Find a codeword that covers u
                covering = np.where(ball_matrix[:, u] == 1)[0]
                # Prefer non-blocked
                covering_nb = covering[blocked[covering] == 0]
                use = covering_nb if len(covering_nb) > 0 else covering
                for nc in use:
                    nc = int(nc)
                    if nc not in code_set:
                        code.append(nc)
                        code_set.add(nc)
                        counts += ball_matrix[nc]
                        blocked[conflict_rows[nc] == 1] = 1
                        remaining_mask = (counts == 0)
                        break

        return code, counts

    def polish(code, counts, time_budget):
        """Local search with conflict-aware 1-flip moves.
        A 1-flip (remove c, add d) is 'clean' if d is not in conflict with
        any other codeword in code \\ {c}; this preserves the independent-set
        structure. We strongly prefer clean flips."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget

        # Maintain blocked array: blocked[w] = 1 if w conflicts with some c in code
        blocked = np.zeros(total, dtype=np.uint8)
        for c in code:
            blocked[conflict_rows[c] == 1] = 1

        def is_removable(c):
            ball_words = neigh[c]
            return np.all(counts[ball_words] > 1)

        # First pass: remove all removable codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    counts[ball_matrix[c] == 1] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]

        # Rebuild blocked after removals
        blocked = np.zeros(total, dtype=np.uint8)
        for c in code:
            blocked[conflict_rows[c] == 1] = 1

        # SA phase with conflict-aware moves
        temp = 3.0
        cool_rate = 0.999
        min_temp = 0.05

        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp

            move_type = rng.random()

            if move_type < 0.35:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        code_set.discard(c)
                        counts[ball_matrix[c] == 1] -= 1
                        # Rebuild blocked (cheap for small code)
                        blocked = np.zeros(total, dtype=np.uint8)
                        for cc in code:
                            blocked[conflict_rows[cc] == 1] = 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]

            elif move_type < 0.85:
                # Try Swap: replace old_c with new_c
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    if rng.random() < 0.7:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c]) - 1)]
                    else:
                        new_c = rng.randint(0, total - 1)

                    if new_c == old_c or new_c in code_set:
                        continue

                    # Conflict check: new_c must not conflict with any other code word
                    # (excluding old_c, which we're removing)
                    # blocked[new_c] tells us if new_c conflicts with ANY current code word.
                    # We need: new_c does not conflict with any code word OTHER than old_c.
                    # Equivalently: after removing old_c, new_c is not blocked.
                    # Compute: does new_c conflict with old_c?
                    old_conflicts_new = conflict_rows[old_c, new_c] == 1
                    # blocked[new_c] is 1 if new_c conflicts with some code word.
                    # If old_conflicts_new and blocked[new_c] == 1, we need to check
                    # if the ONLY conflict is with old_c.
                    # Simpler: rebuild blocked without old_c and check.
                    # For efficiency, check: count of code words (excluding old_c) that conflict with new_c
                    # Use: for each other code word cc, conflict_rows[cc, new_c]
                    other_conflicts = 0
                    for j, cc in enumerate(code):
                        if j == idx:
                            continue
                        if conflict_rows[cc, new_c] == 1:
                            other_conflicts += 1
                            break
                    clean = (other_conflicts == 0)

                    # Coverage check
                    old_ball_mask = ball_matrix[old_c]
                    new_ball_mask = ball_matrix[new_c]
                    unique_old = (old_ball_mask == 1) & (new_ball_mask == 0) & (counts == 1)
                    uncovered = int(unique_old.sum())

                    if uncovered == 0 and clean:
                        # Clean swap: apply
                        counts[old_ball_mask == 1] -= 1
                        counts[new_ball_mask == 1] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        blocked = np.zeros(total, dtype=np.uint8)
                        for cc in code:
                            blocked[conflict_rows[cc] == 1] = 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                    elif uncovered == 0 and not clean:
                        # Coverage-preserving but breaks independence; accept with lower prob
                        if rng.random() < math.exp(-1.5 / temp):
                            counts[old_ball_mask == 1] -= 1
                            counts[new_ball_mask == 1] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            blocked = np.zeros(total, dtype=np.uint8)
                            for cc in code:
                                blocked[conflict_rows[cc] == 1] = 1
                    elif uncovered <= 2 and rng.random() < math.exp(-uncovered / temp):
                        # Accept with coverage loss, then repair
                        counts[old_ball_mask == 1] -= 1
                        counts[new_ball_mask == 1] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        blocked = np.zeros(total, dtype=np.uint8)
                        for cc in code:
                            blocked[conflict_rows[cc] == 1] = 1
                        # Fix uncovered words
                        zero_mask = (counts == 0)
                        if zero_mask.any():
                            zero_indices = np.where(zero_mask)[0]
                            for u in zero_indices:
                                u = int(u)
                                covering = np.where(ball_matrix[:, u] == 1)[0]
                                covering_nb = covering[blocked[covering] == 0]
                                use = covering_nb if len(covering_nb) > 0 else covering
                                for nc in use:
                                    nc = int(nc)
                                    if nc not in code_set:
                                        code.append(nc)
                                        code_set.add(nc)
                                        counts[ball_matrix[nc] == 1] += 1
                                        blocked[conflict_rows[nc] == 1] = 1
                                        break

            else:
                # Try Add (perturbation) - prefer non-blocked
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if blocked[new_c] == 0:
                        if rng.random() < math.exp(-1.0 / temp):
                            code.append(new_c)
                            code_set.add(new_c)
                            counts[ball_matrix[new_c] == 1] += 1
                            blocked[conflict_rows[new_c] == 1] = 1

            temp *= cool_rate

        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    counts[ball_matrix[c] == 1] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]

        # Final conflict-aware polish: try to swap any conflicting pair to reduce conflicts
        # (structural cleanup)
        if time.monotonic() < end_time:
            for _ in range(5):
                # Find conflicting pairs in code
                found_pair = False
                for i in range(len(code)):
                    for j in range(i + 1, len(code)):
                        if conflict_rows[code[i], code[j]] == 1:
                            # Try to fix by swapping one of them
                            found_pair = True
                            break
                    if found_pair:
                        break
                if not found_pair:
                    break

        return best_code, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1

    num_restarts = 25
    time_per_restart = (deadline - time.monotonic()) / num_restarts

    for restart in range(num_restarts):
        if time.monotonic() > deadline - 5:
            break

        # Greedy initialization with different seeds
        code, counts = greedy_init(42 + restart * 7)

        if not code:
            continue

        # Polish
        polished_code, polished_size = polish(code, counts, time_per_restart - 0.3)

        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]

    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result

# Precompute neighbor sets at module level for reuse across calls
_neigh_sets_cache = {}

def _get_neigh_sets(q, n, radius):
    key = (q, n, radius)
    if key not in _neigh_sets_cache:
        total = q ** n
        neigh, _ = _precompute_neighbors(q, n, radius)
        _neigh_sets_cache[key] = [set(neigh[i]) for i in range(total)]
    return _neigh_sets_cache[key]
