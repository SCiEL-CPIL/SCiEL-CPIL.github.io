import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    def can_add_greedy(S, x):
        """Check if x can be added to S. When iterating descending, all elements in S are > x.
        Only constraint: no pair b,c in S with b<=c such that x divides b+c."""
        s_list = sorted(S)
        m = len(s_list)
        for i in range(m):
            b = s_list[i]
            for j in range(i, m):
                c = s_list[j]
                if (b + c) % x == 0:
                    return False
        return True

    def is_valid_set(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does NOT divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_full(S, x):
        """Full check: x can be added to S satisfying a<b<=c, a ∤ (b+c)."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                continue
            else:
                # a < x
                # Case 1: x is 'c'. Need a < b <= x for b in S.
                for j in range(m):
                    b = S_list[j]
                    if a < b <= x:
                        if (b + x) % a == 0:
                            return False
                # Case 2: x is 'b'. Need a < x <= c for c in S.
                for j in range(m):
                    c = S_list[j]
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
        # Case 3: x is 'a'. Need x < b <= c for b, c in S.
        for i in range(m):
            b = S_list[i]
            if b > x:
                if (b + b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        return True

    def local_search(S, rng, time_limit):
        """Try to improve S by swapping elements out for better ones."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S)
            # Try removing each element and replacing with a smaller one
            for idx in range(len(S_list)):
                if time.time() > time_limit:
                    return S
                a = S_list[idx]
                S.remove(a)
                # Try to add a smaller element
                for x in range(1, a):
                    if x not in S:
                        if can_add_full(S, x):
                            S.add(x)
                            improved = True
                            break
                if not improved:
                    S.add(a)  # put it back
        return S

    def local_search_swap(S, rng, time_limit):
        """Try to improve S by swapping one element for another (keeping size)."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S)
            for idx in range(len(S_list)):
                if time.time() > time_limit:
                    return S
                a = S_list[idx]
                S.remove(a)
                # Try to add a different element (not necessarily smaller)
                for x in range(1, n + 1):
                    if x not in S and x != a:
                        if can_add_full(S, x):
                            S.add(x)
                            if x > a:  # prefer larger elements? or just any improvement
                                improved = True
                            else:
                                improved = True
                            break
                if not improved:
                    S.add(a)
        return S

    best_size = 0
    best_S = set()

    # Strategy 1: Descending greedy (inspiration 1)
    if time.time() < deadline - 5.0:
        S = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if can_add_greedy(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 2: Start with upper half, greedily add from below
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 1000 + 42)

        base_start = n - rng.randint(40, 80)
        S = set(range(base_start, n + 1))

        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(1, base_start) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 3: Start with a random subset, greedily add
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 2000 + 99)

        S = set(rng.sample(range(1, n + 1), rng.randint(20, 40)))

        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 4: Start with all odd numbers, then add evens
    if time.time() < deadline - 5.0:
        S = set(range(1, n + 1, 2))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(2, n + 1, 2) if x not in S]
        rng = random.Random(12345)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 5: Start with numbers > n/3, then add smaller
    if time.time() < deadline - 5.0:
        S = set(range(86, n + 1))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        candidates = [x for x in range(1, 86) if x not in S]
        rng = random.Random(54321)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_full(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Strategy 6: Descending greedy with different starting points
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 3000 + 777)

        # Start with a random subset of upper elements
        upper_size = rng.randint(30, 60)
        upper_start = n - upper_size + 1
        S = set(range(upper_start, n + 1))

        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_full(valid_S, x):
                    valid_S.add(x)
            S = valid_S

        # Descending greedy from n-1 down to 1
        for x in range(n - 1, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if x not in S and can_add_greedy(S, x):
                S.add(x)

        # Local search refinement
        if time.time() < deadline - 5.0:
            S = local_search(S, rng, deadline - 3.0)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    if best_size == 0:
        return list(range(129, 257))

    if not is_valid_set(best_S):
        S = set(range(129, 257))
        return sorted(S)

    return sorted(best_S)
