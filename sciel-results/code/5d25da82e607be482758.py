import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Generate 2,3-smooth numbers <= 10000
    smooth_23 = set()
    for i in range(0, 14):
        for j in range(0, 10):
            val = (1 << i) * (3 ** j)
            if val <= 10000:
                smooth_23.add(val)
    smooth_23 = sorted(smooth_23)

    # Generate 2,3,5-smooth numbers <= 10000
    smooth_235 = set()
    for i in range(0, 14):
        for j in range(0, 10):
            for k in range(0, 7):
                val = (1 << i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    smooth_235.add(val)
    smooth_235 = sorted(smooth_235)

    initial_candidates = []

    # Family 1: Subsets of 2,3-smooth numbers (most structured for products)
    # Take first 20 smallest
    if len(smooth_23) >= 20:
        initial_candidates.append(smooth_23[:20])
    # Try various windows
    for offset in range(0, min(50, len(smooth_23) - 20 + 1)):
        cand = smooth_23[offset:offset + 20]
        if len(set(cand)) == 20 and max(cand) <= 10000:
            initial_candidates.append(cand)

    # Family 2: Subsets of 2,3,5-smooth numbers
    if len(smooth_235) >= 20:
        initial_candidates.append(smooth_235[:20])
    for offset in range(0, min(30, len(smooth_235) - 20 + 1)):
        cand = smooth_235[offset:offset + 20]
        if len(set(cand)) == 20 and max(cand) <= 10000:
            initial_candidates.append(cand)

    # Family 3: Powers of 2
    A_powers2 = [1 << i for i in range(20)]
    if max(A_powers2) <= 10000:
        initial_candidates.append(A_powers2)

    # Family 4: Powers of 3
    A_powers3 = [3 ** i for i in range(1, 21)]
    if max(A_powers3) <= 10000:
        initial_candidates.append(A_powers3)

    # Family 5: Mixed powers - 2^i * 3^j with i+j <= 6 (triangular)
    tri_candidates = []
    for i in range(0, 7):
        for j in range(0, 7 - i):
            val = (1 << i) * (3 ** j)
            if val <= 10000:
                tri_candidates.append(val)
    tri_candidates = sorted(set(tri_candidates))
    if len(tri_candidates) >= 20:
        initial_candidates.append(tri_candidates[:20])

    # Family 6: Arithmetic progressions
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 5]:
            cand = [start + i * step for i in range(20)]
            if all(x <= 10000 for x in cand) and len(set(cand)) == 20:
                initial_candidates.append(sorted(cand))

    # Family 7: Geometric progressions with small bases
    for base in [2, 3, 4, 5]:
        for start_exp in range(0, 5):
            cand = [base ** (start_exp + i) for i in range(20)]
            if max(cand) <= 10000 and len(set(cand)) == 20:
                initial_candidates.append(sorted(cand))

    # Family 8: 2^i * 3^j grid subsets
    grid_cands = []
    for max_i in [4, 5, 6]:
        for max_j in [4, 5, 6]:
            vals = []
            for i in range(0, max_i + 1):
                for j in range(0, max_j + 1):
                    v = (1 << i) * (3 ** j)
                    if v <= 10000:
                        vals.append(v)
            vals = sorted(set(vals))
            if len(vals) >= 20:
                grid_cands.append(vals[:20])
    initial_candidates.extend(grid_cands)

    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    for cand in initial_candidates:
        s = compute_metrics(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]

    if not best_set:
        best_set = list(range(1, 21))
        best_score = compute_metrics(best_set)

    current_set = best_set[:]
    current_score = best_score

    # Create a diverse pool for replacement centered on smooth numbers
    pool = set(smooth_235[:200])  # Smallest 200 smooth numbers
    pool.update(range(1, 100))    # Small numbers
    # Add some structured numbers
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
        for k in range(1, 50):
            v = base * k
            if v <= 5000:
                pool.add(v)
    pool = sorted(pool)

    # Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    num_steps = 150000

    step = 0

    while step < num_steps:
        if time.time() > deadline:
            break

        step += 1

        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / num_steps)

        # Choose a random index to replace
        idx = random.randint(0, 19)
        old_val = current_set[idx]

        # Choose a new value - bias towards smooth numbers
        if random.random() < 0.7:
            new_val = random.choice(pool[:100])  # Smaller smooth numbers
        else:
            new_val = random.choice(pool)

        # Ensure distinctness
        if new_val in current_set:
            continue

        # Compute new score
        current_set[idx] = new_val
        new_score = compute_metrics(current_set)

        delta = new_score - current_score

        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val

        # Occasionally do a swap
        if step % 10 == 0:
            idx1 = random.randint(0, 19)
            idx2 = random.randint(0, 19)
            if idx1 != idx2:
                v1 = current_set[idx1]
                v2 = current_set[idx2]
                new_v1 = random.choice(pool)
                new_v2 = random.choice(pool)

                if new_v1 not in current_set and new_v2 not in current_set and new_v1 != new_v2:
                    current_set[idx1] = new_v1
                    current_set[idx2] = new_v2
                    new_score = compute_metrics(current_set)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_set = current_set[:]
                    else:
                        current_set[idx1] = v1
                        current_set[idx2] = v2

    # Final local search polish: try replacements from smooth pool for each position
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                # Try smooth pool first (more likely to help)
                for new_val in pool[:150]:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                # Then try broader pool
                for new_val in pool[150:]:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val

    return sorted(best_set)
