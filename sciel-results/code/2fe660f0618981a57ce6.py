import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    ap_pairs = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # Case 1: x is first (x, a, b)
        # b = 2a - x
        # a > x, b <= N
        # a <= (N + x) // 2
        max_a = min(N, (N + x) // 2)
        for a in range(x + 1, max_a + 1):
            b = 2 * a - x
            ap_pairs[x].append((a, b))
            
        # Case 2: x is middle (a, x, b)
        # b = 2x - a
        # a < x, b <= N
        # a >= 2x - N
        min_a = max(1, 2 * x - N)
        for a in range(min_a, x):
            b = 2 * x - a
            ap_pairs[x].append((a, b))
            
        # Case 3: x is last (a, b, x)
        # a = 2b - x
        # b < x, a >= 1
        # b >= (x + 1) // 2
        min_b = max(1, (x + 1) // 2)
        for b in range(min_b, x):
            a = 2 * b - x
            ap_pairs[x].append((a, b))

    # Greedy Init
    in_set = [False] * (N + 1)
    selected = []
    for x in range(1, N + 1):
        valid = True
        # Check if x forms AP with existing a, b
        # Since x is largest, only case "a, b, x" matters?
        # No, we check all pairs in selected.
        # But as established, only "x is last" is possible for new x in increasing order.
        # x is last => a = 2b - x.
        for b in selected:
            a = 2 * b - x
            if a > 0 and in_set[a]:
                valid = False
                break
        if valid:
            in_set[x] = True
            selected.append(x)
            
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    S = sum(recip[x] for x in selected)
    K = 0
    best_S = S
    best_in_set = in_set.copy()
    
    LAMBDA = 2.0
    T = 1.0
    cooling = 0.9995
    steps_per_check = 1000
    
    def count_aps(x):
        c = 0
        for a, b in ap_pairs[x]:
            if in_set[a] and in_set[b]:
                c += 1
        return c

    step = 0
    while time.time() < deadline:
        x = random.randint(1, N)
        if in_set[x]:
            c = count_aps(x)
            dS = -recip[x]
            dK = -c
            dE = -dS + LAMBDA * dK
            if dE < 0 or random.random() < math.exp(-dE / T):
                in_set[x] = False
                S += dS
                K += dK
                if K == 0 and S > best_S:
                    best_S = S
                    best_in_set = in_set.copy()
        else:
            c = count_aps(x)
            dS = recip[x]
            dK = c
            dE = -dS + LAMBDA * dK
            if dE < 0 or random.random() < math.exp(-dE / T):
                in_set[x] = True
                S += dS
                K += dK
                if K == 0 and S > best_S:
                    best_S = S
                    best_in_set = in_set.copy()
                    
        step += 1
        if step % steps_per_check == 0:
            T *= cooling
            if T < 0.01:
                T = 0.01

    result = [i for i in range(1, N + 1) if best_in_set[i]]
    if not result:
        return [1]
    return result

# EVOLVE-BLOCK-END
