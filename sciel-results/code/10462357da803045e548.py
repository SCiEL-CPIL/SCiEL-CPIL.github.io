"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _score_set(arr):
    """Count distinct d>0 such that a, a+d, a+2d all in set for some a."""
    n = len(arr)
    if n < 3:
        return 0
    s = set(arr)
    diffs = set()
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            b = arr[j]
            d = b - a
            if d <= 0:
                continue
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs)

def _score_set_fast(arr, membership):
    """Fast scoring using precomputed membership array (1-indexed up to 512)."""
    n = len(arr)
    if n < 3:
        return 0
    diffs = set()
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            b = arr[j]
            d = b - a
            if d <= 0:
                continue
            c = b + d
            if c <= 512 and membership[c]:
                diffs.add(d)
    return len(diffs)

def _make_membership(arr):
    mem = [False] * 513
    for x in arr:
        mem[x] = True
    return mem

def _random_initial():
    return sorted(random.sample(range(1, 513), 32))

def _dense_initial():
    # Consecutive block shifted randomly
    start = random.randint(1, 481)
    return list(range(start, start + 32))

def _structured_initial():
    # Mix: some arithmetic progression + random fill
    d = random.randint(1, 15)
    start = random.randint(1, 512 - 31 * d)
    base = list(range(start, start + 32 * d, d))[:32]
    remaining = [x for x in range(1, 513) if x not in base]
    if len(base) < 32:
        extra = random.sample(remaining, 32 - len(base))
        base.extend(extra)
    return sorted(base)

def _sa(arr, start_time, deadline, rng):
    """Simulated annealing on a 32-element sorted list."""
    n = len(arr)
    arr = list(arr)
    mem = _make_membership(arr)
    best_score = _score_set_fast(arr, mem)
    best_arr = arr[:]
    cur_score = best_score
    cur_arr = arr[:]
    cur_mem = mem

    T = 50.0
    T_min = 0.5
    alpha = 0.997
    steps_since_best = 0
    max_steps_since_best = 2000

    step = 0
    while True:
        # Deadline check
        if time.time() > deadline:
            break
        if T < T_min:
            break

        # Generate move
        move_type = rng.random()
        if move_type < 0.6:
            # Single swap: remove one element, add a new random one
            idx = rng.randrange(n)
            old = cur_arr[idx]
            new = rng.randint(1, 512)
            if new == old:
                continue
            if new in cur_mem:
                continue
            # Try the move
            trial = cur_arr[:]
            trial[idx] = new
            trial.sort()
            trial_mem = cur_mem[:]
            trial_mem[old] = False
            trial_mem[new] = True
            trial_score = _score_set_fast(trial, trial_mem)
            delta = trial_score - cur_score
        elif move_type < 0.85:
            # Double swap: remove two, add two
            i1, i2 = rng.sample(range(n), 2)
            old1, old2 = cur_arr[i1], cur_arr[i2]
            new1, new2 = rng.sample(range(1, 513), 2)
            if new1 == new2 or new1 in cur_mem or new2 in cur_mem:
                continue
            trial = cur_arr[:]
            trial[i1] = new1
            trial[i2] = new2
            trial.sort()
            trial_mem = cur_mem[:]
            trial_mem[old1] = False
            trial_mem[old2] = False
            trial_mem[new1] = True
            trial_mem[new2] = True
            trial_score = _score_set_fast(trial, trial_mem)
            delta = trial_score - cur_score
        else:
            # Local perturbation: shift one element by small amount
            idx = rng.randrange(n)
            old = cur_arr[idx]
            shift = rng.choice([-3, -2, -1, 1, 2, 3, 5, 7])
            new = old + shift
            if new < 1 or new > 512 or new in cur_mem:
                continue
            trial = cur_arr[:]
            trial[idx] = new
            trial.sort()
            trial_mem = cur_mem[:]
            trial_mem[old] = False
            trial_mem[new] = True
            trial_score = _score_set_fast(trial, trial_mem)
            delta = trial_score - cur_score

        # Accept or reject
        if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
            cur_arr = trial
            cur_mem = trial_mem
            cur_score = trial_score
            if cur_score > best_score:
                best_score = cur_score
                best_arr = cur_arr[:]
                steps_since_best = 0
            else:
                steps_since_best += 1
        else:
            steps_since_best += 1

        T *= alpha
        step += 1

        # Restart if stuck too long
        if steps_since_best > max_steps_since_best:
            break

    return best_arr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    # Generate diverse initial solutions
    inits = []
    inits.append(_dense_initial())
    inits.append(_structured_initial())
    inits.append(_random_initial())
    inits.append(_random_initial())
    inits.append(_structured_initial())
    inits.append(_random_initial())

    best_overall = None
    best_overall_score = 0

    for init in inits:
        if time.time() > deadline - 10.0:
            break
        arr, score = _sa(init, start_time, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr

    # Final local improvement on best
    if best_overall is None:
        best_overall = _dense_initial()
        best_overall_score = _score_set(best_overall)

    # Quick hill-climbing polish
    if time.time() < deadline - 5.0:
        arr = best_overall[:]
        mem = _make_membership(arr)
        cur_score = best_overall_score
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(32):
                old = arr[idx]
                for new in range(1, 513):
                    if new == old or mem[new]:
                        continue
                    trial = arr[:]
                    trial[idx] = new
                    trial.sort()
                    trial_mem = mem[:]
                    trial_mem[old] = False
                    trial_mem[new] = True
                    trial_score = _score_set_fast(trial, trial_mem)
                    if trial_score > cur_score:
                        arr = trial
                        mem = trial_mem
                        cur_score = trial_score
                        improved = True
                        break
                if improved:
                    break

        if cur_score > best_overall_score:
            best_overall = arr
            best_overall_score = cur_score

    # Ensure validity: exactly 32 distinct integers in 1..512
    result = sorted(set(best_overall))
    if len(result) != 32:
        # Fallback
        result = list(range(1, 33))
    # Clamp to 1..512
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])

    return result
# EVOLVE-BLOCK-END
