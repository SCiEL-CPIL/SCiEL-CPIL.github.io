import time
import random

def construct():
    best_set = []
    best_size = 0
    deadline = time.time() + 110  # 10 seconds safety margin

    def try_greedy(order):
        S = []
        counts = [0] * 2001
        is_full = [False] * 2001
        for x in order:
            valid = True
            # Check 2x
            if is_full[2 * x]:
                valid = False
            else:
                # Check x+y for y in S
                # Using local variable for speed
                c = counts
                f = is_full
                for y in S:
                    if f[x + y]:
                        valid = False
                        break
            if valid:
                S.append(x)
                # Update counts and is_full
                c[2 * x] += 1
                if c[2 * x] == 2:
                    f[2 * x] = True
                for y in S[:-1]:
                    s = x + y
                    c[s] += 1
                    if c[s] == 2:
                        f[s] = True
        return S

    # 1. Standard Greedy
    S = try_greedy(range(1, 1001))
    if len(S) > best_size:
        best_set = S
        best_size = len(S)

    # 2. Reverse Greedy
    S = try_greedy(range(1000, 0, -1))
    if len(S) > best_size:
        best_set = S
        best_size = len(S)

    # 3. Random Greedy Restarts
    # Use a fixed seed for reproducibility in local tests, but random in production?
    # The evaluator might run this multiple times. Random is fine.
    while time.time() < deadline:
        order = list(range(1, 1001))
        random.shuffle(order)
        S = try_greedy(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)

    return best_set
