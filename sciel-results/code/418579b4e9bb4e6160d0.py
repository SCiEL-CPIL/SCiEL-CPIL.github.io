"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0

    def get_sums_set(S):
        sums = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            sums.add(si + si)
            for j in range(i + 1, n):
                sums.add(si + S[j])
        return sums

    def is_sidon(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def fast_is_maximal(S):
        """Check if S is inclusion-maximal Sidon in [1,N]."""
        if not S:
            return False
        sums = get_sums_set(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            for a in S:
                if x + a in sums:
                    break
            else:
                return False
        return True

    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            if 2 * x in sums:
                continue
            conflict = False
            for a in S:
                if x + a in sums:
                    conflict = True
                    break
            if not conflict:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S

    def count_unique_sums(S, element_idx):
        """Count how many sums are uniquely contributed by S[element_idx]."""
        S = list(S)
        elem = S[element_idx]
        S_without = S[:element_idx] + S[element_idx + 1:]
        sums_with = get_sums_set(S)
        sums_without = get_sums_set(S_without)
        return len(sums_with - sums_without)

    def prune_for_maximality(S):
        """Remove elements while maintaining maximality. Prioritize elements with fewest unique sums."""
        S = list(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(S)
            if n == 0:
                break
            # Score each element by number of unique sums it contributes
            scores = []
            for i in range(n):
                unique = count_unique_sums(S, i)
                scores.append((unique, i))
            # Try removing from lowest unique-sum count first
            scores.sort()
            for unique, i in scores:
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i + 1:]
                if len(new_S) == 0:
                    continue
                if fast_is_maximal(new_S):
                    S = new_S
                    changed = True
                    break
        return S

    def fast_greedy(order):
        S = []
        sums = set()
        for x in order:
            if 2 * x in sums:
                continue
            conflict = False
            for a in S:
                if x + a in sums:
                    conflict = True
                    break
            if not conflict:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S

    # Phase 1: Build initial Sidon set using ascending greedy
    S = greedy_construct(range(1, N + 1))
    S = prune_for_maximality(S)
    best_S = list(S)
    best_size = len(S)

    # Phase 2: Randomized restarts with various orderings
    rng = random.Random(42)
    base_order = list(range(1, N + 1))

    for trial in range(80000):
        if time.time() > deadline:
            break

        t10 = trial % 10
        if t10 == 0:
            order = base_order[:]
            rng.shuffle(order)
        elif t10 == 1:
            order = base_order[::-1]
        elif t10 == 2:
            order = list(range(N, 0, -1))
            half = len(order) // 2
            for i in range(half - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif t10 == 3:
            order = []
            left, right = 1, N
            while left <= right:
                order.append(left)
                if left != right:
                    order.append(right)
                left += 1
                right -= 1
        elif t10 == 4:
            order = base_order[:]
            rng.shuffle(order)
            low = [x for x in order if x <= N // 2]
            high = [x for x in order if x > N // 2]
            new_order = []
            li, hi = 0, 0
            for i in range(N):
                if li < len(low):
                    new_order.append(low[li])
                    li += 1
                if hi < len(high):
                    new_order.append(high[hi])
                    hi += 1
            order = new_order
        elif t10 == 5:
            seed_val = trial * 6151 + 7
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1))
        elif t10 == 6:
            seed_val = trial * 8191 + 13
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1), reverse=True)
        elif t10 == 7:
            order = list(range(1, N + 1))
            rng.shuffle(order)
            for _ in range(rng.randint(1, 8)):
                idx = rng.randint(0, N - 1)
                order.insert(0, order.pop(idx))
        elif t10 == 8:
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                max_j = max(0, i - i // 3)
                j = rng.randint(0, max_j)
                order[i], order[j] = order[j], order[i]
        else:
            rng2 = random.Random(trial * 104729 + 7)
            order = list(range(1, N + 1))
            rng2.shuffle(order)

        S = fast_greedy(order)
        if S:
            S_pruned = prune_for_maximality(S)
            if len(S_pruned) < best_size:
                best_size = len(S_pruned)
                best_S = list(S_pruned)

    # Phase 3: Aggressive local search - try removing elements from best
    S = list(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(S)
        if n == 0:
            break
        # Score by unique sums
        scores = []
        for i in range(n):
            unique = count_unique_sums(S, i)
            scores.append((unique, i))
        scores.sort()
        for unique, i in scores:
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i + 1:]
            if not new_S:
                continue
            if fast_is_maximal(new_S):
                S = new_S
                improved = True
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                break

    # Phase 4: Try removing + replacing to get smaller sets
    S = list(best_S)
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        n = len(S)
        if n <= 1:
            break
        # Try each removal
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i + 1:]
            if not S_without:
                continue
            sums_without = get_sums_set(S_without)
            S_without_set = set(S_without)

            # Try to find a replacement that keeps maximality
            found = False
            for x in range(1, N + 1):
                if x in S_without_set:
                    continue
                if 2 * x in sums_without:
                    continue
                blocked = False
                for a in S_without:
                    if x + a in sums_without:
                        blocked = True
                        break
                if blocked:
                    continue
                new_S = sorted(S_without + [x])
                if fast_is_maximal(new_S):
                    S = new_S
                    improved = True
                    found = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
            if improved or found:
                break

    # Phase 5: Final aggressive removal pass with scoring
    S = list(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(S)
        if n == 0:
            break
        scores = []
        for i in range(n):
            unique = count_unique_sums(S, i)
            scores.append((unique, i))
        scores.sort()
        for unique, i in scores:
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i + 1:]
            if not new_S:
                continue
            if fast_is_maximal(new_S):
                S = new_S
                improved = True
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                break

    # Final verification and repair
    S = list(best_S)
    if not S or not is_sidon(S):
        S = greedy_construct(range(1, N + 1))
        S = prune_for_maximality(S)

    if not fast_is_maximal(S):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            sums = get_sums_set(S)
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if 2 * x in sums:
                    continue
                can_add = True
                for a in S:
                    if x + a in sums:
                        can_add = False
                        break
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    sums.add(2 * x)
                    for a in S[:-1]:
                        sums.add(x + a)
                    changed = True
                    break
        S = sorted(S)

    return S

# EVOLVE-BLOCK-END
