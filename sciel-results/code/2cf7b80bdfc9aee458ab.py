import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def verify_set(s):
        n = len(s)
        if n < 3:
            return True
        for i, a in enumerate(s):
            others = [x for j, x in enumerate(s) if j != i]
            m = len(others)
            if m < 2:
                continue
            max_s = sum(others)
            if max_s == 0:
                continue
            dp = [np.zeros(max_s + 1, dtype=bool) for _ in range(m + 1)]
            dp[0][0] = True
            for x in others:
                for k in range(m - 1, -1, -1):
                    if x <= max_s:
                        dp[k + 1][x:] |= dp[k][:(max_s + 1) - x]
            for k in range(2, m + 1):
                target = k * a
                if target <= max_s and dp[k][target]:
                    return False
        return True

    max_sum = 48 * 512
    max_k = 48
    dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_k + 1)]
    dp[0][0] = True

    selected = []
    current_size = 0

    for candidate in range(1, 513):
        if time.time() > deadline - 10:
            break
        if len(selected) >= 48:
            break

        valid = True
        for k in range(2, current_size + 1):
            target = k * candidate
            if target <= max_sum and dp[k][target]:
                valid = False
                break

        if not valid:
            continue

        current_size += 1
        for j in range(current_size - 1, -1, -1):
            if j + 1 <= max_k:
                dp[j + 1][candidate:] |= dp[j][:(max_sum + 1) - candidate]
        selected.append(candidate)

    if time.time() > deadline - 2:
        return selected

    if not verify_set(selected):
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 0.5:
                break
            test = selected[:i] + selected[i + 1:]
            if verify_set(test):
                selected = test
                break

    return selected
