import time
import random
import bisect

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    def check_valid(S_list):
        """Check if S_list (sorted) satisfies the property."""
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def is_valid_add(S_list, m):
        """Check if adding m to S_list (sorted) maintains the property."""
        pos = bisect.bisect_left(S_list, m)
        L = len(S_list)

        # Case 1: m is smallest (a=m), need m ∤ (b+c) for all b,c in S with b > m, c >= b
        for j in range(pos, L):
            b = S_list[j]
            for k in range(j, L):
                c = S_list[k]
                if (b + c) % m == 0:
                    return False

        # Case 2: m is middle (b=m), need a ∤ (m+c) for all a < m, c >= m
        for i in range(0, pos):
            a = S_list[i]
            for k in range(pos, L):
                c = S_list[k]
                if (m + c) % a == 0:
                    return False

        # Case 3: m is largest (c=m), need a ∤ (b+m) for all a < b <= m
        for i in range(0, pos):
            a = S_list[i]
            for j in range(i + 1, pos):
                b = S_list[j]
                if (b + m) % a == 0:
                    return False

        return True

    # Strategy: Start with upper tail, try multiple strategies to add elements
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)

    # Precompute: for each a in 1..256, which pairs (b,c) with b<=c, b>a have a|(b+c)
    # This is expensive but we can use it to speed up checks

    # Multiple random restarts with different addition orders
    random.seed(42)
    for restart in range(80):
        if time.time() > deadline:
            break

        S = list(base_set)
        candidates = list(range(1, 171))

        if restart % 3 == 0:
            # Ascending
            candidates.sort()
        elif restart % 3 == 1:
            # Descending
            candidates.sort(reverse=True)
        else:
            # Random
            random.shuffle(candidates)

        for m in candidates:
            if time.time() > deadline:
                break
            if is_valid_add(S, m):
                pos = bisect.bisect_left(S, m)
                S.insert(pos, m)

        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    # Local search: try to add more elements, multiple passes
    improved = True
    pass_count = 0
    while improved and time.time() < deadline and pass_count < 10:
        improved = False
        pass_count += 1
        S = list(best_S)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]

        # Try adding in order of "easiest" first (larger numbers tend to be easier)
        not_in_S.sort(reverse=True)

        for add_m in not_in_S:
            if time.time() > deadline:
                break
            if is_valid_add(S, add_m):
                pos = bisect.bisect_left(S, add_m)
                S.insert(pos, add_m)
                best_S = list(S)
                best_size = len(S)
                improved = True
                in_S = set(S)
                not_in_S = [m for m in range(1, n + 1) if m not in in_S]
                not_in_S.sort(reverse=True)
                break

    # Try removing and re-adding different elements (swap search)
    if time.time() < deadline - 10:
        for swap_attempt in range(20):
            if time.time() > deadline:
                break
            S = list(best_S)
            in_S = set(S)
            not_in_S = [m for m in range(1, n + 1) if m not in in_S]
            found_improvement = False

            # Try removing one element and adding two
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                removed = S[i]
                S_temp = S[:i] + S[i+1:]
                # Try to add two elements
                added_count = 0
                for add_m in not_in_S:
                    if time.time() > deadline:
                        break
                    if is_valid_add(S_temp, add_m):
                        pos = bisect.bisect_left(S_temp, add_m)
                        S_temp.insert(pos, add_m)
                        added_count += 1
                        if added_count >= 2:
                            break
                if added_count >= 2:
                    best_S = sorted(S_temp)
                    best_size = len(best_S)
                    found_improvement = True
                    break

            if found_improvement:
                break

    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        # Fallback to known valid set
        return list(range(171, 257))

    return S_final
