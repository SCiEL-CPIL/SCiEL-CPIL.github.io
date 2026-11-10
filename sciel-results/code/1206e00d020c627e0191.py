import random
import time
import math

def construct():
    n = 20
    max_val = 10000
    start_time = time.time()
    time_limit = 55  # Safety margin for 90s wall-clock limit

    def compute_sumprod(A):
        """Compute full sumset and productset, return (sums, prods, max_size)."""
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return sums, prods, max(len(sums), len(prods))

    def incremental_swap_eval(A, sums, prods, idx, new_val):
        """
        Compute exact new |A+A| and |A*A| after replacing A[idx] with new_val.
        Returns (new_sums, new_prods, new_max) or None if new_val collides.
        """
        old = A[idx]
        if new_val == old:
            return None
        # Check collision with other elements
        for k in range(n):
            if k != idx and A[k] == new_val:
                return None

        new_sums = set(sums)
        new_prods = set(prods)

        # Remove pairs involving old
        for k in range(n):
            if k == idx:
                continue
            s_val = old + A[k]
            p_val = old * A[k]
            # Only remove if not present in any other pair
            # For sums: check if s_val can be formed by another pair
            # For efficiency, we just remove and re-add; if another pair has same sum, it'll be re-added
            new_sums.discard(s_val)
            new_prods.discard(p_val)

        # Remove diagonal old+old
        s_diag = old + old
        p_diag = old * old
        # Check if s_diag exists in another pair
        found_s = False
        for i in range(n):
            for j in range(i, n):
                if i == idx or j == idx:
                    continue
                if A[i] + A[j] == s_diag:
                    found_s = True
                    break
            if found_s:
                break
        if not found_s:
            new_sums.discard(s_diag)

        found_p = False
        for i in range(n):
            for j in range(i, n):
                if i == idx or j == idx:
                    continue
                if A[i] * A[j] == p_diag:
                    found_p = True
                    break
            if found_p:
                break
        if not found_p:
            new_prods.discard(p_diag)

        # Add pairs involving new_val
        for k in range(n):
            if k == idx:
                continue
            new_sums.add(new_val + A[k])
            new_prods.add(new_val * A[k])

        # Add diagonal new_val+new_val
        new_sums.add(new_val + new_val)
        new_prods.add(new_val * new_val)

        return new_sums, new_prods, max(len(new_sums), len(new_prods))

    # Initial candidates
    candidates = []

    # 1. Random
    for _ in range(10):
        A = random.sample(range(1, max_val + 1), n)
        candidates.append(A)

    # 2. Arithmetic Progressions
    for start in range(1, 50):
        for step in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]:
            A = [start + i * step for i in range(n)]
            if A[-1] <= max_val:
                candidates.append(A)

    # 3. Geometric-like (powers)
    for base in [2, 3, 4, 5]:
        for start in [1, 2, 3]:
            A = [start * (base ** i) for i in range(n)]
            if A[-1] <= max_val:
                candidates.append(A)

    # 4. Smooth numbers (products of small primes)
    smooth = []
    for a in range(0, 6):
        for b in range(0, 6):
            for c in range(0, 5):
                for d in range(0, 4):
                    v = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if 1 <= v <= max_val:
                        smooth.append(v)
    smooth = sorted(set(smooth))
    if len(smooth) >= n:
        # Take first n, or evenly spaced
        candidates.append(smooth[:n])
        if len(smooth) > n:
            step = len(smooth) // n
            candidates.append([smooth[i * step] for i in range(n)])

    # 5. Multiplicative AP (geometric with integer ratios)
    for ratio in [2, 3, 4]:
        for base in range(1, 10):
            A = [base * (ratio ** i) for i in range(n)]
            if A[-1] <= max_val:
                candidates.append(A)

    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    best_sums = None
    best_prods = None

    for A in candidates:
        if len(set(A)) != n:
            continue
        if any(x < 1 or x > max_val for x in A):
            continue
        sums, prods, score = compute_sumprod(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
            best_sums = sums
            best_prods = prods

    if best_set is None:
        best_set = list(range(1, n + 1))
        best_sums, best_prods, best_score = compute_sumprod(best_set)

    # Local search with incremental evaluation
    current_set = best_set[:]
    current_sums = best_sums
    current_prods = best_prods
    current_score = best_score

    T = 50.0
    T_min = 0.5
    alpha = 0.9995
    no_improve = 0
    max_no_improve = 5000

    while time.time() - start_time < time_limit:
        # Generate a random swap candidate
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]

        # Choose new value: mix of random and local perturbation
        r = random.random()
        if r < 0.4:
            # Local perturbation
            magnitude = random.choice([1, 2, 3, 5, 10, 20, 50, 100])
            sign = random.choice([-1, 1])
            new_val = old_val + sign * magnitude
        elif r < 0.7:
            # Random within range
            new_val = random.randint(1, max_val)
        else:
            # Random near current value
            new_val = random.randint(max(1, old_val - 200), min(max_val, old_val + 200))

        if new_val < 1 or new_val > max_val:
            continue

        # Quick collision check
        if new_val in current_set and new_val != old_val:
            continue

        # Incremental evaluation
        result = incremental_swap_eval(current_set, current_sums, current_prods, idx, new_val)
        if result is None:
            continue

        new_sums, new_prods, new_score = result

        # Acceptance
        if new_score < current_score:
            current_set[idx] = new_val
            current_sums = new_sums
            current_prods = new_prods
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
                best_sums = current_sums
                best_prods = current_prods
        else:
            delta = new_score - current_score
            if T > T_min:
                if random.random() < math.exp(-delta / T):
                    current_set[idx] = new_val
                    current_sums = new_sums
                    current_prods = new_prods
                    current_score = new_score
            no_improve += 1

        T = max(T_min, T * alpha)

        # If stuck for too long, restart from best
        if no_improve > max_no_improve:
            current_set = best_set[:]
            current_sums = best_sums
            current_prods = best_prods
            current_score = best_score
            no_improve = 0
            T = 50.0

    return best_set
