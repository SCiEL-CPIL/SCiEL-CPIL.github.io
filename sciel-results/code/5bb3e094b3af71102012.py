import time
import random
import bisect

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

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

    # Start with upper tail [171, 256]
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)

    # Precompute conflict info: for each m in 1..256, which existing elements block it
    # Use a smarter approach: try all subsets of a "candidate pool" using greedy + local search

    random.seed(12345)

    # Strategy 1: Multiple greedy passes with different orderings
    for restart in range(150):
        if time.time() > deadline:
            break

        S = list(base_set)
        candidates = list(range(1, 171))

        if restart % 4 == 0:
            candidates.sort()
        elif restart % 4 == 1:
            candidates.sort(reverse=True)
        elif restart % 4 == 2:
            random.shuffle(candidates)
        else:
            # Weighted random: prefer larger numbers
            weights = [c for c in candidates]
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

    # Strategy 2: Local search - try to add elements not in best_S
    improved = True
    pass_count = 0
    while improved and time.time() < deadline and pass_count < 15:
        improved = False
        pass_count += 1
        S = list(best_S)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
        not_in_S.sort(reverse=True)  # Try larger first (easier to add)

        for add_m in not_in_S:
            if time.time() > deadline:
                break
            if is_valid_add(S, add_m):
                pos = bisect.bisect_left(S, add_m)
                S.insert(pos, add_m)
                best_S = list(S)
                best_size = len(S)
                improved = True
                break  # Restart the pass

    # Strategy 3: Swap search - remove one, add two
    if time.time() < deadline - 15:
        for swap_attempt in range(30):
            if time.time() > deadline:
                break
            S = list(best_S)
            in_S = set(S)
            not_in_S = [m for m in range(1, n + 1) if m not in in_S]
            not_in_S.sort(reverse=True)
            found_improvement = False

            for i in range(len(S)):
                if time.time() > deadline:
                    break
                removed = S[i]
                S_temp = S[:i] + S[i+1:]
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

    # Strategy 4: Try different base intervals
    if time.time() < deadline - 10:
        for start_val in range(160, 172):
            if time.time() > deadline:
                break
            base = list(range(start_val, n + 1))
            S = list(base)
            candidates = list(range(1, start_val))
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

    # Final local search
    improved = True
    pass_count = 0
    while improved and time.time() < deadline and pass_count < 10:
        improved = False
        pass_count += 1
        S = list(best_S)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
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
                break

    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        return list(range(171, 257))

    return S_final
