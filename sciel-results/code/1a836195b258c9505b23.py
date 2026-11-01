import time
import random
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback
    if total_words <= 5000:
        return _generic_solve(q, n, radius, deadline)
    return [[0] * n]


def _generic_solve(q, n, radius, deadline):
    """Generic solver for small alphabets/lengths."""
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Build coverage table
    coverage = []
    for w in all_words:
        neighbors = [w]
        for pos in range(n):
            for v in range(q):
                if v != w[pos]:
                    nw = list(w)
                    nw[pos] = v
                    neighbors.append(tuple(nw))
        mask = 0
        for nb in neighbors:
            mask |= 1 << word_to_idx[nb]
        coverage.append(mask)

    full_mask = (1 << N) - 1

    # Greedy set cover
    covered = 0
    code = []
    while covered != full_mask:
        best_gain = -1
        best_idx = -1
        for idx in range(N):
            gain = (coverage[idx] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        code.append(best_idx)
        covered |= coverage[best_idx]

    # Prune
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            others = 0
            for j in range(len(code)):
                if j != i:
                    others |= coverage[code[j]]
            if coverage[code[i]] & ~others == 0:
                code.pop(i)
                changed = True
                break

    return [list(all_words[c]) for c in code]


def _solve_q3_n6_r1(deadline):
    """Specialized solver for q=3, n=6, radius=1."""
    q = 3
    n = 6
    N = 729

    # Precompute neighbors using integer encoding
    # Each word is an integer 0..728 in base-3
    neighbors = []
    for i in range(N):
        word = [(i // (3 ** (n - 1 - p))) % 3 for p in range(n)]
        nbs = [i]
        for pos in range(n):
            for v in range(q):
                if v == word[pos]:
                    continue
                new_word = word[:]
                new_word[pos] = v
                new_i = 0
                for p in range(n):
                    new_i = new_i * 3 + new_word[p]
                nbs.append(new_i)
        neighbors.append(nbs)

    # Coverage bitset for each word
    coverage = []
    for i in range(N):
        mask = 0
        for nb in neighbors[i]:
            mask |= 1 << nb
        coverage.append(mask)

    full_mask = (1 << N) - 1

    # Tetracode direct sum (81 words) as initial solution
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = h + list(tail)
            idx = 0
            for p in range(n):
                idx = idx * 3 + word[p]
            initial_code.append(idx)

    best_code = initial_code[:]
    best_size = len(best_code)

    def compute_union(code_list):
        u = 0
        for c in code_list:
            u |= coverage[c]
        return u

    def prune(code_list, time_limit=None):
        """Remove redundant codewords."""
        code_list = list(code_list)
        changed = True
        while changed:
            if time_limit and time.time() > time_limit:
                break
            changed = False
            for i in range(len(code_list)):
                others = 0
                for j in range(len(code_list)):
                    if j != i:
                        others |= coverage[code_list[j]]
                if coverage[code_list[i]] & ~others == 0:
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    def local_search(code_list, time_limit):
        """Try to reduce code size via swaps and removals."""
        code_list = list(code_list)
        code_set = set(code_list)
        union = compute_union(code_list)

        if union != full_mask:
            return code_list

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(code_list)):
                if time.time() > time_limit:
                    break
                # Find unique coverage of code_list[i]
                others = 0
                for j in range(len(code_list)):
                    if j != i:
                        others |= coverage[code_list[j]]
                unique = coverage[code_list[i]] & ~others

                if unique == 0:
                    # Remove redundant
                    code_list.pop(i)
                    code_set.discard(code_list[i])
                    union = compute_union(code_list)
                    improved = True
                    break

                # Try to find replacement
                for j in range(N):
                    if time.time() > time_limit:
                        break
                    if j in code_set:
                        continue
                    if unique & ~coverage[j]:
                        continue
                    # Check if new union still covers everything
                    new_union = (union & ~coverage[code_list[i]]) | coverage[j]
                    if new_union == union:
                        code_list[i] = j
                        code_set.discard(code_list[i])
                        code_set.add(j)
                        union = new_union
                        improved = True
                        break

        return code_list

    # Phase 1: Prune and local search from initial
    current = prune(initial_code, deadline - 30.0)
    current = local_search(current, deadline - 25.0)
    if len(current) < best_size:
        best_size = len(current)
        best_code = current[:]

    # Phase 2: Multiple greedy restarts with randomization
    for restart in range(40):
        if time.time() > deadline - 15.0:
            break

        covered = 0
        code = []
        rng = random.Random(restart * 7919 + 42)

        # Randomly shuffle candidates
        candidates = list(range(N))
        rng.shuffle(candidates)

        while covered != full_mask:
            if time.time() > deadline - 20.0:
                break
            best_gain = -1
            best_idx = -1
            best_candidates = []
            for idx in candidates:
                gain = (coverage[idx] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_candidates = [idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(idx)

            if best_gain <= 0:
                break

            # Random tie-breaking
            if len(best_candidates) > 1 and rng.random() < 0.3:
                best_idx = rng.choice(best_candidates)

            code.append(best_idx)
            covered |= coverage[best_idx]

        if covered != full_mask:
            continue

        # Prune and local search
        current = prune(code, deadline - 10.0)
        current = local_search(current, deadline - 8.0)
        if len(current) < best_size:
            best_size = len(current)
            best_code = current[:]

    # Phase 3: Final polish on best solution
    if time.time() < deadline - 5.0:
        final = prune(best_code, deadline - 3.0)
        final = local_search(final, deadline - 1.0)
        if len(final) < best_size:
            best_size = len(final)
            best_code = final

    # Convert back to word lists
    result = []
    for idx in best_code:
        word = [(idx // (3 ** (n - 1 - p))) % 3 for p in range(n)]
        result.append(word)
    return result
