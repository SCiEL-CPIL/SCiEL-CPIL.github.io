import numpy as np
import sys
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    time_limit = 600  # seconds, leave 30s margin for 630s total

    # Forbidden differences mod 205
    forbidden_205 = set()
    for x in range(1, m):
        forbidden_205.add(pow(x, 2, m))
    forbidden_205.discard(0)

    # Nonzero squares mod 41
    squares_41 = set()
    for x in range(1, 41):
        squares_41.add(pow(x, 2, 41))
    squares_41.discard(0)

    # Build adjacency for Paley(41)
    n = 41
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = (i - j) % n
            if d in squares_41:
                adj[i].add(j)
                adj[j].add(i)

    # Simulated annealing to find large independent set in Paley(41)
    def is_valid_set(s):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                d = (s[i] - s[j]) % n
                if d in squares_41:
                    return False
        return True

    # Start with greedy
    def greedy_order(order):
        s = set()
        for v in order:
            can_add = True
            for u in s:
                if (v - u) % n in squares_41:
                    can_add = False
                    break
            if can_add:
                s.add(v)
        return s

    # Multiple greedy with different orderings
    best_set = set()
    best_size = 0

    for seed in range(50):
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        s = greedy_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    print(f"Best greedy size: {best_size}, set: {sorted(best_set)}", file=sys.stderr)

    # Simulated annealing to improve
    random.seed(123)
    current_set = set(best_set)
    current_size = len(current_set)
    best_sa_set = set(current_set)
    best_sa_size = current_size

    T_start = 1.0
    T_end = 0.01
    iterations = 200000

    for it in range(iterations):
        if time.time() - start_time > time_limit - 30:
            break

        T = T_start * (T_end / T_start) ** (it / iterations)

        # Choose a move: add, remove, or swap
        move_type = random.random()
        if move_type < 0.3:
            # Try to add a vertex
            candidates = [v for v in range(n) if v not in current_set]
            if not candidates:
                continue
            v = random.choice(candidates)
            conflicts = [u for u in current_set if (v - u) % n in squares_41]
            if not conflicts:
                new_set = set(current_set)
                new_set.add(v)
                delta = 1
            else:
                # Can't add without removing
                continue
        elif move_type < 0.6:
            # Try to remove a vertex
            if not current_set:
                continue
            v = random.choice(list(current_set))
            new_set = set(current_set)
            new_set.discard(v)
            delta = -1
        else:
            # Try to swap: remove one, add another
            if not current_set:
                continue
            v_remove = random.choice(list(current_set))
            candidates = [v for v in range(n) if v not in current_set and v != v_remove]
            if not candidates:
                continue
            v_add = random.choice(candidates)
            # Check if v_add conflicts with current_set minus v_remove
            temp_set = set(current_set)
            temp_set.discard(v_remove)
            conflicts = [u for u in temp_set if (v_add - u) % n in squares_41]
            if not conflicts:
                new_set = set(temp_set)
                new_set.add(v_add)
                delta = 0
            else:
                continue

        if delta > 0:
            current_set = new_set
            current_size += delta
            if current_size > best_sa_size:
                best_sa_size = current_size
                best_sa_set = set(current_set)
        elif delta < 0:
            current_set = new_set
            current_size += delta
        else:
            # delta == 0, accept with some probability
            if random.random() < 0.5:
                current_set = new_set

    if best_sa_size > best_size:
        best_size = best_sa_size
        best_set = set(best_sa_set)

    print(f"Best SA size: {best_size}, set: {sorted(best_set)}", file=sys.stderr)

    # Construct mod 205 answer using CRT
    # Residues 0 and 2 mod 5 (since 2 is a non-square mod 5)
    result = set()
    for k in best_set:
        result.add(5 * k % 205)
        result.add((2 + 5 * k) % 205)

    result_arr = np.array(sorted(result), dtype=np.int64)

    # Verify
    valid = True
    for i in range(len(result_arr)):
        for j in range(i+1, len(result_arr)):
            d1 = (result_arr[i] - result_arr[j]) % 205
            d2 = (result_arr[j] - result_arr[i]) % 205
            if d1 in forbidden_205 or d2 in forbidden_205:
                valid = False
                break
        if not valid:
            break

    if not valid:
        print("VALIDATION FAILED, using trivial set", file=sys.stderr)
        return np.array([0], dtype=np.int64)

    print(f"Final result size: {len(result_arr)}", file=sys.stderr)
    return result_arr
