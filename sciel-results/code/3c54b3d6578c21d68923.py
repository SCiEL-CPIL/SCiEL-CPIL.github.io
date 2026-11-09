import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_set = None
    best_size = N + 1
    
    def is_valid_addition(x, S, used_sums):
        if 2 * x in used_sums:
            return False
        for s in S:
            if x + s in used_sums:
                return False
        return True

    def greedy_build(order):
        S = []
        used_sums = set()
        for x in order:
            if is_valid_addition(x, S, used_sums):
                S.append(x)
                # Add sums
                used_sums.add(2 * x)
                for s in S:
                    used_sums.add(x + s)
        return S

    # 1. Ascending
    s = greedy_build(range(1, N + 1))
    if len(s) < best_size:
        best_size = len(s)
        best_set = s

    # 2. Descending
    s = greedy_build(range(N, 0, -1))
    if len(s) < best_size:
        best_size = len(s)
        best_set = s

    # 3. Randomized Greedy
    # To speed up, we can pre-shuffle the list and iterate.
    base_candidates = list(range(1, N + 1))
    
    while time.time() < deadline:
        random.shuffle(base_candidates)
        s = greedy_build(base_candidates)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            # If we find a very small set, we might want to try to improve it further?
            # But we can't reduce it directly.
            # We can try to generate sets similar to it?
            # For now, just keep searching.

    return best_set
