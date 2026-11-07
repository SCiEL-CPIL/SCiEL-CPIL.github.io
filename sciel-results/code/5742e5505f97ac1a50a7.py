import time
import random

def construct():
    start_time = time.time()
    max_val = 256
    max_size = 60
    deadline = start_time + 85.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def is_valid(dp, max_k):
        for i in range(max_k + 1):
            di = dp[i]
            if di == 0:
                continue
            for j in range(i + 1, max_k + 1):
                if di & dp[j]:
                    return False
        return True

    def try_add(dp, x, current_size):
        new_dp = dp[:]
        max_k_new = current_size + 1
        for k in range(1, max_k_new + 1):
            new_dp[k] |= (dp[k-1] << x)
        if is_valid(new_dp, max_k_new):
            return new_dp
        return None

    def run_order(order):
        nonlocal best_set, best_size
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        current_size = 0
        for x in order:
            if current_size >= max_size:
                break
            new_dp = try_add(dp, x, current_size)
            if new_dp is not None:
                S.append(x)
                dp = new_dp
                current_size += 1
        if current_size > best_size:
            best_size = current_size
            best_set = S[:]
            if best_size == max_size:
                return True
        return False

    # Deterministic orders
    for order in [list(range(1, max_val + 1)), list(range(max_val, 0, -1))]:
        if time.time() > deadline:
            break
        if run_order(order):
            break
        if best_size == max_size:
            break

    # Random restarts
    while time.time() < deadline:
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        if run_order(candidates):
            break
        if best_size == max_size:
            break

    return [int(x) for x in best_set] if best_set else []
