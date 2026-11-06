import time
import random
import math

def compute_metrics(A):
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def generate_smooth_numbers(primes, limit):
    nums = {1}
    for p in primes:
        new_nums = set()
        for x in nums:
            v = x
            while v * p <= limit:
                v *= p
                new_nums.add(v)
        nums.update(new_nums)
    return sorted([x for x in nums if x <= limit])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Generate candidate pool
    pool = set(range(1, 101))
    pool.update(generate_smooth_numbers([2, 3, 5], 10000))
    pool.update(generate_smooth_numbers([2, 3, 5, 7], 10000))
    for p in [2, 3, 5, 7, 11, 13]:
        v = 1
        while v <= 10000:
            pool.add(v)
            v *= p

    pool = sorted([x for x in pool if 1 <= x <= 10000])
    pool_size = len(pool)

    # Initial sets
    initial_sets = [
        list(range(1, 21)),
        generate_smooth_numbers([2, 3, 5], 10000)[:20],
        generate_smooth_numbers([2, 3, 5, 7], 10000)[:20],
        [2**i for i in range(20)],
        sorted(random.sample(pool, 20))
    ]

    best_score = float('inf')
    best_A = initial_sets[0]

    # Evaluate initial sets
    for s in initial_sets:
        s = sorted(set(s))
        if len(s) == 20:
            score = compute_metrics(s)
            if score < best_score:
                best_score = score
                best_A = s[:]

    # Simulated Annealing with multiple move types
    current_A = best_A[:]
    current_score = best_score

    random.seed(42)
    T = 50.0
    T_min = 0.1
    cooling = 0.99995

    steps_since_improvement = 0
    max_steps_no_improve = 8000

    while time.time() < deadline:
        if time.time() >= deadline:
            break

        # Check for restart
        if steps_since_improvement > max_steps_no_improve:
            if random.random() < 0.5:
                current_A = best_A[:]
                current_score = best_score
            else:
                current_A = sorted(random.sample(pool, 20))
                current_score = compute_metrics(current_A)
            steps_since_improvement = 0
            T = max(T, 10.0)

        # Choose move type: 60% single swap, 30% double swap, 10% insert+remove
        move_type = random.random()
        trial_A = current_A[:]

        if move_type < 0.6:
            # Single element replacement
            idx = random.randint(0, 19)
            new_val = pool[random.randint(0, pool_size - 1)]
            if new_val not in current_A:
                trial_A[idx] = new_val
                trial_A = sorted(set(trial_A))
            else:
                T = max(T * cooling, T_min)
                continue
        elif move_type < 0.9:
            # Double element swap
            i1, i2 = random.sample(range(20), 2)
            v1 = pool[random.randint(0, pool_size - 1)]
            v2 = pool[random.randint(0, pool_size - 1)]
            if v1 not in current_A and v2 not in current_A and v1 != v2:
                trial_A[i1] = v1
                trial_A[i2] = v2
                trial_A = sorted(set(trial_A))
            else:
                T = max(T * cooling, T_min)
                continue
        else:
            # Insert one, remove one (different positions)
            idx_rm = random.randint(0, 19)
            new_val = pool[random.randint(0, pool_size - 1)]
            if new_val not in current_A:
                trial_A.pop(idx_rm)
                trial_A.append(new_val)
                trial_A = sorted(set(trial_A))
            else:
                T = max(T * cooling, T_min)
                continue

        if len(trial_A) != 20:
            T = max(T * cooling, T_min)
            continue

        trial_score = compute_metrics(trial_A)
        delta = trial_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
            current_A = trial_A[:]
            current_score = trial_score

            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1

        T = max(T * cooling, T_min)

    # Final local search polish
    if time.time() < deadline - 5:
        polish_deadline = min(deadline, time.time() + 4.0)
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                for new_val in pool:
                    if new_val in best_A:
                        continue
                    trial = best_A[:]
                    trial[i] = new_val
                    trial = sorted(set(trial))
                    if len(trial) != 20:
                        continue
                    s = compute_metrics(trial)
                    if s < best_score:
                        best_score = s
                        best_A = trial[:]
                        improved = True
                        break
                if time.time() >= polish_deadline:
                    break

    return best_A
