# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0

    # Generate smooth numbers up to 128 with exponent vectors
    smooth = []
    smooth_exp = {}
    for x in range(1, 129):
        y = x
        a = b = c = 0
        while y % 2 == 0:
            y //= 2
            a += 1
        while y % 3 == 0:
            y //= 3
            b += 1
        while y % 5 == 0:
            y //= 5
            c += 1
        if y == 1:
            smooth.append(x)
            smooth_exp[x] = (a, b, c)

    smooth_set = set(smooth)
    n_smooth = len(smooth)

    # Precompute incompatibility: incompat[i][j] = True if smooth[i] and smooth[j] are incompatible (one divides the other)
    # For antichain, we need NO pair to be comparable
    incompat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        ei = smooth_exp[smooth[i]]
        for j in range(i + 1, n_smooth):
            ej = smooth_exp[smooth[j]]
            if (ei[0] <= ej[0] and ei[1] <= ej[1] and ei[2] <= ej[2]) or \
               (ej[0] <= ei[0] and ej[1] <= ei[1] and ej[2] <= ei[2]):
                incompat[i][j] = True
                incompat[j][i] = True

    # Precompute for each index, the set of indices it's compatible with (bitmask for speed)
    # But n_smooth is ~50, so we can use a list of compatible indices
    compat_list = [[] for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and not incompat[i][j]:
                compat_list[i].append(j)

    def find_rep(n):
        # k=1
        if n in smooth_set:
            return [n]

        # k=2: try largest a first
        for i in range(n_smooth - 1, -1, -1):
            a = smooth[i]
            if a >= n:
                continue
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                # Find index of b
                # a < b, check if a divides b
                ei = smooth_exp[a]
                ej = smooth_exp[b]
                if not (ei[0] <= ej[0] and ei[1] <= ej[1] and ei[2] <= ej[2]):
                    return [a, b]

        # k=3: try combinations with largest first
        # Use backtracking with pruning
        def backtrack(target, min_idx, current_sum, current_list, current_indices, max_depth):
            if current_sum == target:
                return current_list[:]
            if current_sum > target:
                return None
            if len(current_list) >= max_depth:
                return None

            remaining = target - current_sum
            # Try terms from largest to smallest
            for idx in range(min_idx - 1, -1, -1):
                term = smooth[idx]
                if term > remaining:
                    continue
                if current_sum + term > target:
                    continue

                # Check compatibility with all existing terms
                valid = True
                for existing_idx in current_indices:
                    if incompat[idx][existing_idx]:
                        valid = False
                        break

                if valid:
                    current_list.append(term)
                    current_indices.append(idx)
                    res = backtrack(target, idx, current_sum + term, current_list, current_indices, max_depth)
                    if res:
                        return res
                    current_list.pop()
                    current_indices.pop()
            return None

        # Try k=3, 4, 5 with backtracking
        for k in range(3, 8):
            res = backtrack(n, n_smooth, 0, [], [], k)
            if res:
                return res

        # k>=8: use randomized search
        pool = [x for x in smooth if x < n]
        pool_set = set(pool)
        import random
        rng = random.Random(n)

        for k in range(8, min(31, len(pool) + 1)):
            for attempt in range(500):
                if time.time() > deadline - 3:
                    break
                terms = rng.sample(pool, k)
                if sum(terms) == n:
                    # Check antichain
                    valid = True
                    for i in range(len(terms)):
                        for j in range(i + 1, len(terms)):
                            a, b = terms[i], terms[j]
                            if a > b:
                                a, b = b, a
                            if b % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return sorted(terms)
            if time.time() > deadline - 3:
                break

        # Fallback: exhaustive backtracking without depth limit
        res = backtrack(n, n_smooth, 0, [], [], 30)
        if res:
            return res

        # Absolute fallback
        return [1]

    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 5:
            # Emergency: fill remaining with any valid representation
            for m in range(n, 129):
                if m in smooth_set:
                    answers.append([m])
                else:
                    # Quick find
                    found = find_rep(m)
                    answers.append(found)
            break

        found = find_rep(n)
        if found is None:
            found = [n] if n in smooth_set else [1]
        answers.append(found)

    return answers
# EVOLVE-BLOCK-END
