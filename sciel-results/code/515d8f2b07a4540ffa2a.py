import time
import random

def construct():
    start_time = time.time()
    max_val = 256
    max_size = 60
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    best_set = []
    best_size = 0
    
    def is_valid(new_dp, max_k):
        for i in range(max_k + 1):
            if new_dp[i] == 0:
                continue
            for j in range(i + 1, max_k + 1):
                if new_dp[i] & new_dp[j]:
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
    orders = [
        list(range(1, max_val + 1)),
        list(range(max_val, 0, -1)),
        list(range(1, max_val + 1, 2)) + list(range(2, max_val + 1, 2)),
        list(range(2, max_val + 1, 2)) + list(range(1, max_val + 1, 2)),
        [i for i in range(1, max_val + 1) if i % 3 != 0] + [i for i in range(1, max_val + 1) if i % 3 == 0],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        if run_order(order):
            break
        if best_size == max_size:
            break

    # Random restarts
    rng = random.Random(42)
    while time.time() < deadline:
        strategy = rng.randint(0, 3)
        if strategy == 0:
            candidates = list(range(1, max_val + 1))
            rng.shuffle(candidates)
        elif strategy == 1:
            small = list(range(1, 30))
            rest = list(range(30, max_val + 1))
            rng.shuffle(small)
            rng.shuffle(rest)
            candidates = small + rest
        elif strategy == 2:
            large = list(range(max_val - 29, max_val + 1))
            rest = list(range(1, max_val - 29))
            rng.shuffle(large)
            rng.shuffle(rest)
            candidates = large + rest
        else:
            candidates = list(range(1, max_val + 1))
            rng.shuffle(candidates)
        
        if run_order(candidates):
            break
        if best_size == max_size:
            break

    # Try to extend best set with remaining elements
    if best_size < max_size and time.time() < deadline:
        used = set(best_set)
        remaining = [x for x in range(1, max_val + 1) if x not in used]
        dp = [0] * (max_size + 1)
        dp[0] = 1
        current_size = 0
        for x in best_set:
            new_dp = try_add(dp, x, current_size)
            if new_dp is not None:
                dp = new_dp
                current_size += 1
        
        for x in remaining:
            if time.time() > deadline:
                break
            if current_size >= max_size:
                break
            new_dp = try_add(dp, x, current_size)
            if new_dp is not None:
                best_set.append(x)
                dp = new_dp
                current_size += 1
                best_size = current_size

    return best_set
