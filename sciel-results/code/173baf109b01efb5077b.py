import math
import time

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 55

    # Precompute compatibility matrix
    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(2, n + 1):
        for j in range(i + 1, n + 1):
            compat[i][j] = compat[j][i] = math.gcd(i, j) > 1

    # Start with multiples of 3 (largest single-prime-factor set containing 255)
    current_set = set(range(3, n + 1, 3))

    all_numbers = list(range(2, n + 1))

    # Local search: try 1-swap improvements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for to_remove in list(current_set):
            if time.time() > deadline:
                break
            if to_remove == 255:
                continue
            trial = current_set - {to_remove}
            for to_add in all_numbers:
                if to_add in current_set:
                    continue
                if all(compat[to_add][y] for y in trial):
                    current_set = trial | {to_add}
                    improved = True
                    break

    return sorted(current_set)
