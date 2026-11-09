import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    max_elements = 48
    n = 192

    def is_safe_incremental(a, S_list, S_set_local):
        """Check if adding a to S_list keeps validity. Only checks new constraints involving a."""
        L = len(S_list)
        if L >= 3:
            # Check 1: a doesn't divide sum of any 3 distinct elements in S_list
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % a] += 1

            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False

        # Check 2: For each existing element b in S_list, b doesn't divide sum of a + 2 distinct others
        for b in S_list:
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue

            target_rem = (-a) % b
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1

            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    return False

        return True

    def is_valid_set(S_list):
        """Check if S_list is valid: no element divides sum of 3 distinct others."""
        for idx, a in enumerate(S_list):
            others = [v for i, v in enumerate(S_list) if i != idx]
            L_len = len(others)
            if L_len < 3:
                continue

            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % a] += 1

            for i in range(L_len):
                u = others[i]
                for j in range(i + 1, L_len):
                    v = others[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True

    def greedy(order, max_time):
        """Greedy construction with given order of candidates."""
        S = []
        S_set = set()
        for x in order:
            if len(S) >= max_elements:
                break
            if time.time() - start_time > max_time:
                break
            if is_safe_incremental(x, S, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, max_iter=50):
        """Try to improve set S by adding elements or swapping."""
        S_list = list(S)
        S_set = set(S_list)
        best_S = list(S_set)
        best_size = len(best_S)

        for _ in range(max_iter):
            if time.time() - start_time > deadline - 2:
                break
            improved = False

            # Try to add an element
            for x in range(1, n + 1):
                if x in S_set:
                    continue
                if is_safe_incremental(x, S_list, S_set):
                    S_list.append(x)
                    S_set.add(x)
                    best_S = list(S_set)
                    best_size += 1
                    improved = True
                    break

            if improved:
                continue

            # Try to swap: remove one, add another
            for i in range(len(S_list)):
                if time.time() - start_time > deadline - 2:
                    break
                remove = S_list[i]
                S_temp = [v for j, v in enumerate(S_list) if j != i]
                S_temp_set = set(S_temp)

                for x in range(1, n + 1):
                    if x in S_temp_set:
                        continue
                    if is_safe_incremental(x, S_temp, S_temp_set):
                        S_list = S_temp + [x]
                        S_set = S_temp_set | {x}
                        best_S = list(S_set)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return best_S

    # Pass 1: greedy from largest to smallest
    S = greedy(range(n, 0, -1), deadline - 50)

    # Pass 2: try to add remaining numbers (from small to large) that fit
    if len(S) < max_elements and time.time() - start_time < deadline - 30:
        for x in range(1, n + 1):
            if len(S) >= max_elements:
                break
            if time.time() - start_time > deadline - 25:
                break
            if x not in set(S):
                if is_safe_incremental(x, S, set(S)):
                    S.append(x)

    # Local search on initial result
    if time.time() - start_time < deadline - 10:
        S = local_search(S, max_iter=30)

    # Randomized restarts
    random.seed(42)
    best_result = list(S)
    best_size = len(best_result)

    remaining_time = deadline - (time.time() - start_time)
    num_restarts = min(20, max(1, int(remaining_time / 2)))

    for trial in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break

        order = list(range(1, n + 1))
        random.shuffle(order)
        result = greedy(order, deadline - 3)

        if len(result) > best_size and time.time() - start_time < deadline - 8:
            result = local_search(result, max_iter=15)
            if len(result) > best_size:
                best_result = list(result)
                best_size = len(best_result)

    # Final validation
    if not is_valid_set(best_result):
        # Fallback to greedy large-to-small
        best_result = greedy(range(n, 0, -1), deadline - 2)

    return best_result[:max_elements]
