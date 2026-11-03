import itertools
import random
import time
import math

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    N = q**n
    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    neighbors = [[] for _ in range(N)]
    neighbors_set = [set() for _ in range(N)]

    for idx, w in enumerate(words):
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        neighbors[idx] = list(set(neighbors[idx]))
        neighbors_set[idx] = set(neighbors[idx])

    # Precompute distance-2 graph: words at Hamming distance exactly 2
    # These are "conflicts" - two codewords at distance 2 can potentially be merged
    dist2_neighbors = [[] for _ in range(N)]
    for idx, w in enumerate(words):
        for i in range(n):
            for j in range(i+1, n):
                for d1 in range(1, q):
                    for d2 in range(1, q):
                        new_w = list(w)
                        new_w[i] = (new_w[i] + d1) % q
                        new_w[j] = (new_w[j] + d2) % q
                        dist2_neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        dist2_neighbors[idx] = list(set(dist2_neighbors[idx]))

    def greedy_init():
        code = []
        code_set = set()
        covered = [False] * N

        all_words = list(range(N))
        random.shuffle(all_words)

        while not all(covered):
            best_w = -1
            best_count = 0
            for w in all_words:
                if w in code_set:
                    continue
                cnt = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_w = w

            if best_w == -1:
                break

            code.append(best_w)
            code_set.add(best_w)
            for u in neighbors[best_w]:
                covered[u] = True

        return code, code_set

    def tetracode_init():
        """Start from the tetracode construction (81 words) and use it as a structured seed."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        code_set = set()
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                w = tuple(h + list(tail))
                idx = tuple_to_idx(w)
                code.append(idx)
                code_set.add(idx)
        return code, code_set

    def dist2_local_search(code, code_set, time_limit):
        """
        Structural local search using distance-2 conflicts.
        Key insight: if two codewords c1 and c2 are at distance 2, we can try
        to replace them with a single word that covers the union of their balls
        (minus what's already covered by other codewords).
        """
        cover_count = [0] * N
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1

        best_code = list(code)
        best_size = len(best_code)
        start = time.time()

        while time.time() < start + time_limit:
            improved = False

            # Phase 1: Remove redundant codewords
            i = 0
            while i < len(code):
                c = code[i]
                uncovered = []
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        uncovered.append(nb)

                if not uncovered:
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break
                else:
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                    i += 1

            if improved:
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue

            # Phase 2: Distance-2 merge attempt
            # Find pairs of codewords at distance 2 and try to merge
            merged = False
            for i in range(len(code)):
                if time.time() > start + time_limit:
                    break
                c1 = code[i]
                for j in range(i+1, len(code)):
                    if time.time() > start + time_limit:
                        break
                    c2 = code[j]

                    # Check if c1 and c2 are at distance 2
                    if c2 not in neighbors_set[c1] and c1 not in neighbors_set[c2]:
                        # They're not at distance 1, check if distance 2
                        # Quick check: see if they share a common neighbor
                        common = neighbors_set[c1] & neighbors_set[c2]
                        if not common:
                            continue

                        # Try to find a replacement word that covers all uniquely-covered
                        # neighbors of both c1 and c2
                        unique1 = [nb for nb in neighbors[c1] if cover_count[nb] == 1]
                        unique2 = [nb for nb in neighbors[c2] if cover_count[nb] == 1]

                        if not unique1 and not unique2:
                            # Both are redundant in some sense, just remove one
                            # Actually if cover_count is 1 for all their neighbors,
                            # removing one leaves the other to cover
                            # Let's just try removing the one with fewer unique neighbors
                            if len(unique1) <= len(unique2):
                                # Remove c1
                                for nb in neighbors[c1]:
                                    cover_count[nb] -= 1
                                code.pop(i)
                                code_set.remove(c1)
                                merged = True
                                break
                            else:
                                # Remove c2
                                for nb in neighbors[c2]:
                                    cover_count[nb] -= 1
                                code.pop(j)
                                code_set.remove(c2)
                                merged = True
                                break

                        # Try to find a single word that covers all unique neighbors
                        if unique1 or unique2:
                            all_unique = set(unique1) | set(unique2)
                            # Candidate: any word in the intersection of balls of all unique neighbors
                            # A word w can replace both c1 and c2 if for every u in all_unique,
                            # w is in neighbors[u]
                            # This means w must be within distance 1 of every uniquely-covered neighbor

                            # Get candidates from the first unique neighbor's ball
                            if all_unique:
                                first_u = list(all_unique)[0]
                                candidates = list(neighbors_set[first_u])
                                random.shuffle(candidates)

                                for c_prime in candidates[:30]:
                                    if c_prime in code_set and c_prime != c1 and c_prime != c2:
                                        continue

                                    # Check if c_prime covers all unique neighbors
                                    ok = True
                                    for u in all_unique:
                                        if c_prime not in neighbors_set[u]:
                                            ok = False
                                            break

                                    if ok:
                                        # Verify: after removing c1 and c2 and adding c_prime,
                                        # nothing becomes uncovered
                                        # Temporarily simulate
                                        temp_cover = cover_count[:]
                                        for nb in neighbors[c1]:
                                            temp_cover[nb] -= 1
                                        for nb in neighbors[c2]:
                                            temp_cover[nb] -= 1
                                        for nb in neighbors[c_prime]:
                                            temp_cover[nb] += 1

                                        valid = True
                                        for nb in range(N):
                                            if temp_cover[nb] < 0:
                                                valid = False
                                                break

                                        if valid:
                                            # Apply the merge
                                            for nb in neighbors[c1]:
                                                cover_count[nb] -= 1
                                            for nb in neighbors[c2]:
                                                cover_count[nb] -= 1
                                            for nb in neighbors[c_prime]:
                                                cover_count[nb] += 1

                                            # Remove c1 and c2, add c_prime
                                            if c_prime not in code_set:
                                                code.remove(c1)
                                                code.remove(c2)
                                                code_set.remove(c1)
                                                code_set.remove(c2)
                                                code.append(c_prime)
                                                code_set.add(c_prime)
                                            else:
                                                # c_prime already in code, just remove c1 and c2
                                                code.remove(c1)
                                                code.remove(c2)
                                                code_set.remove(c1)
                                                code_set.remove(c2)

                                            merged = True
                                            break

                                if merged:
                                    break

                if merged:
                    break

            if merged:
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue

            # Phase 3: Single swap
            for i in range(len(code)):
                if time.time() > start + time_limit:
                    break
                c = code[i]

                uncovered = []
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        uncovered.append(nb)

                if not uncovered:
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break

                u0 = uncovered[0]
                candidates = list(neighbors_set[u0])
                random.shuffle(candidates)

                valid_swaps = []
                for c_prime in candidates[:50]:
                    if c_prime in code_set:
                        continue
                    ok = True
                    for u in uncovered[1:]:
                        if c_prime not in neighbors_set[u]:
                            ok = False
                            break
                    if ok:
                        valid_swaps.append(c_prime)

                if valid_swaps:
                    c_prime = random.choice(valid_swaps)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c_prime]:
                        cover_count[nb] += 1
                    code[i] = c_prime
                    code_set.remove(c)
                    code_set.add(c_prime)
                    improved = True
                    break

            if not improved:
                break

        return best_code

    def simulated_annealing(code, code_set, time_limit):
        """SA with distance-2 merge moves."""
        cover_count = [0] * N
        uncovered_count = 0
        for c in code:
            for nb in neighbors[c]:
                if cover_count[nb] == 0:
                    uncovered_count += 1
                cover_count[nb] += 1

        best_code = list(code)
        best_size = len(code)

        current_obj = len(code) + 10 * uncovered_count
        T = 5.0
        cooling = 0.9997
        start = time.time()

        while time.time() < start + time_limit:
            move_type = random.random()

            if move_type < 0.30 and len(code) > 1:
                # Remove a codeword
                i = random.randint(0, len(code) - 1)
                c = code[i]
                new_uncovered = 0
                affected = []
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        new_uncovered += 1
                    affected.append(nb)

                if new_uncovered == 0:
                    code.pop(i)
                    code_set.remove(c)
                    new_obj = len(code) + 10 * uncovered_count
                    delta = new_obj - current_obj
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_obj = new_obj
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                    else:
                        code.insert(i, c)
                        code_set.add(c)
                        for nb in affected:
                            cover_count[nb] += 1
                else:
                    for nb in affected:
                        cover_count[nb] += 1
                T *= cooling

            elif move_type < 0.60 and len(code) > 1:
                # Distance-2 merge: pick two codewords at distance 2, try to merge
                i = random.randint(0, len(code) - 1)
                j = random.randint(0, len(code) - 1)
                if i == j:
                    T *= cooling
                    continue

                c1 = code[i]
                c2 = code[j]

                # Check they're at distance 2 (not 1, not 0)
                if c1 == c2 or c2 in neighbors_set[c1]:
                    T *= cooling
                    continue

                common = neighbors_set[c1] & neighbors_set[c2]
                if not common:
                    T *= cooling
                    continue

                # Try to find a replacement
                unique1 = [nb for nb in neighbors[c1] if cover_count[nb] == 1]
                unique2 = [nb for nb in neighbors[c2] if cover_count[nb] == 1]
                all_unique = set(unique1) | set(unique2)

                if all_unique:
                    first_u = list(all_unique)[0]
                    candidates = list(neighbors_set[first_u])
                    random.shuffle(candidates)

                    for c_prime in candidates[:20]:
                        if c_prime in code_set and c_prime != c1 and c_prime != c2:
                            continue

                        ok = True
                        for u in all_unique:
                            if c_prime not in neighbors_set[u]:
                                ok = False
                                break

                        if ok:
                            # Simulate
                            temp_cover = cover_count[:]
                            for nb in neighbors[c1]:
                                temp_cover[nb] -= 1
                            for nb in neighbors[c2]:
                                temp_cover[nb] -= 1
                            for nb in neighbors[c_prime]:
                                temp_cover[nb] += 1

                            valid = True
                            new_uncov = 0
                            for nb in range(N):
                                if temp_cover[nb] < 0:
                                    valid = False
                                    break
                                if temp_cover[nb] == 0:
                                    new_uncov += 1

                            if valid:
                                old_uncov = uncovered_count
                                uncovered_count = new_uncov

                                # Apply
                                for nb in neighbors[c1]:
                                    cover_count[nb] -= 1
                                for nb in neighbors[c2]:
                                    cover_count[nb] -= 1
                                for nb in neighbors[c_prime]:
                                    cover_count[nb] += 1

                                # Remove c1, c2 from code
                                idx1 = code.index(c1)
                                idx2 = code.index(c2)
                                if idx1 > idx2:
                                    idx1, idx2 = idx2, idx1
                                code.pop(idx2)
                                code.pop(idx1)
                                code_set.remove(c1)
                                code_set.remove(c2)

                                if c_prime not in code_set:
                                    code.append(c_prime)
                                    code_set.add(c_prime)

                                new_obj = len(code) + 10 * uncovered_count
                                delta = new_obj - current_obj
                                if delta <= 0 or random.random() < math.exp(-delta / T):
                                    current_obj = new_obj
                                    if len(code) < best_size:
                                        best_size = len(code)
                                        best_code = list(code)
                                else:
                                    # Revert
                                    for nb in neighbors[c_prime]:
                                        cover_count[nb] -= 1
                                    for nb in neighbors[c2]:
                                        cover_count[nb] += 1
                                    for nb in neighbors[c1]:
                                        cover_count[nb] += 1

                                    code.append(c1)
                                    code.append(c2)
                                    code_set.add(c1)
                                    code_set.add(c2)
                                    if c_prime in code_set and c_prime != c1 and c_prime != c2:
                                        # This shouldn't happen since we checked
                                        pass
                                    uncovered_count = old_uncov

                                break

            elif move_type < 0.85 and len(code) > 0:
                # Swap a codeword
                i = random.randint(0, len(code) - 1)
                c = code[i]

                unique_neighbors = [nb for nb in neighbors[c] if cover_count[nb] == 1]

                if not unique_neighbors:
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    code_set.remove(c)
                    new_obj = len(code) + 10 * uncovered_count
                    delta = new_obj - current_obj
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_obj = new_obj
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                    else:
                        code.insert(i, c)
                        code_set.add(c)
                        for nb in neighbors[c]:
                            cover_count[nb] += 1
                    T *= cooling
                    continue

                candidates = list(neighbors_set[unique_neighbors[0]])
                random.shuffle(candidates)

                swapped = False
                for c_prime in candidates[:30]:
                    if c_prime in code_set:
                        continue

                    new_uncovered = 0
                    temp_affected = []
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                        if cover_count[nb] == 0:
                            new_uncovered += 1
                        temp_affected.append(nb)
                    for nb in neighbors[c_prime]:
                        cover_count[nb] += 1
                        temp_affected.append(nb)

                    actual_new_uncovered = 0
                    for nb in temp_affected:
                        if cover_count[nb] == 0:
                            actual_new_uncovered += 1

                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                    for nb in neighbors[c_prime]:
                        cover_count[nb] -= 1

                    if actual_new_uncovered == 0:
                        for nb in neighbors[c]:
                            cover_count[nb] -= 1
                        for nb in neighbors[c_prime]:
                            cover_count[nb] += 1
                        code[i] = c_prime
                        code_set.remove(c)
                        code_set.add(c_prime)
                        new_obj = len(code) + 10 * uncovered_count
                        delta = new_obj - current_obj
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            current_obj = new_obj
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = list(code)
                        else:
                            for nb in neighbors[c]:
                                cover_count[nb] += 1
                            for nb in neighbors[c_prime]:
                                cover_count[nb] -= 1
                            code[i] = c
                            code_set.remove(c_prime)
                            code_set.add(c)
                        swapped = True
                        break

                if not swapped:
                    T *= cooling
            else:
                T *= cooling

            if T < 0.1:
                T = 0.1

        return best_code

    start_time = time.time()
    deadline = 140.0
    global_best = None
    global_best_size = float('inf')

    # Use multiple initialization strategies
    init_strategies = [tetracode_init, greedy_init]

    while time.time() < deadline:
        time_budget = deadline - time.time()
        if time_budget < 15:
            break

        # Cycle through initialization strategies
        strat_idx = int((time.time() - start_time) % 2)
        code, code_set = init_strategies[strat_idx]()

        # Distance-2 local search
        ls_time = min(8.0, time_budget - 10.0)
        if ls_time > 0:
            code = dist2_local_search(code, code_set, ls_time)
            code_set = set(code)

        # Simulated annealing with distance-2 merges
        sa_time = min(25.0, time_budget - 5.0)
        if sa_time > 0:
            best_code = simulated_annealing(code, code_set, sa_time)

            if len(best_code) < global_best_size:
                global_best_size = len(best_code)
                global_best = best_code

    if global_best is None:
        code, _ = tetracode_init()
        global_best = code

    return [list(idx_to_tuple(c)) for c in global_best]
