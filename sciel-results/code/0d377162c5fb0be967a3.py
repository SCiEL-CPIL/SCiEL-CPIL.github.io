"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _arr_to_mask(arr):
    m = 0
    for v in arr:
        m |= (1 << v)
    return m

def _mask_to_arr(m, n=32):
    return [i + 1 for i in range(512) if (m >> i) & 1][:n]

def _score_bitset(m):
    """
    Score = number of distinct d > 0 such that there exist a with
    a, a+d, a+2d all in the set (positions 1..512).

    Bitset trick: for each shift s, ((m >> s) & (m >> (2*s))) != 0
    means some element x has both x+s and x+2s present.
    """
    cnt = 0
    m0 = m
    for s in range(1, 257):
        if ((m0 >> s) & (m0 >> (2 * s))) != 0:
            cnt += 1
    return cnt

def _score_bitset_range(m, lo, hi):
    """Score only differences in [lo, hi] (inclusive)."""
    cnt = 0
    m0 = m
    for s in range(max(1, lo), min(hi, 256) + 1):
        if ((m0 >> s) & (m0 >> (2 * s))) != 0:
            cnt += 1
    return cnt

def _diffs_in_mask(m, lo, hi):
    """Return the list of d in [lo, hi] that are 'good'."""
    out = []
    m0 = m
    for s in range(max(1, lo), min(hi, 256) + 1):
        if ((m0 >> s) & (m0 >> (2 * s))) != 0:
            out.append(s)
    return out

def _random_initial(rng):
    return sorted(rng.sample(range(1, 513), 32))

def _dense_initial(rng):
    start = rng.randint(1, 481)
    return list(range(start, start + 32))

def _ap_initial(rng):
    # A random arithmetic progression of length 32 in 1..512
    d = rng.randint(1, 15)
    start = rng.randint(1, 512 - 31 * d)
    return [start + i * d for i in range(32)]

def _mix_initial(rng):
    # Half AP + half random fill
    d = rng.randint(2, 10)
    start = rng.randint(1, 512 - 15 * d)
    base = set(start + i * d for i in range(16))
    rest = [x for x in range(1, 513) if x not in base]
    base.update(rng.sample(rest, 16))
    return sorted(base)

def _sa_bitset(init_arr, deadline, rng, budget_frac=1.0):
    """
    Simulated annealing on a 32-element set represented by a bit mask.
    Uses bitset scoring for speed.
    """
    mask = _arr_to_mask(init_arr)
    cur_score = _score_bitset(mask)
    best_score = cur_score
    best_mask = mask

    T = 8.0
    T_min = 0.4
    alpha = 0.9985
    stuck = 0
    max_stuck = 6000
    total_budget = deadline - time.time()
    t0 = time.time()
    step = 0
    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break
        # Move
        r = rng.random()
        if r < 0.55:
            # Single remove+add
            bits = [i + 1 for i in range(512) if (mask >> i) & 1]
            old = bits[rng.randrange(32)]
            new = rng.randint(1, 512)
            if new == old or ((mask >> (new - 1)) & 1):
                continue
            new_mask = mask & ~(1 << (old - 1)) | (1 << (new - 1))
            ns = _score_bitset(new_mask)
            delta = ns - cur_score
        elif r < 0.80:
            # Double remove+add
            bits = [i + 1 for i in range(512) if (mask >> i) & 1]
            i1, i2 = rng.sample(range(32), 2)
            o1, o2 = bits[i1], bits[i2]
            n1, n2 = rng.sample(range(1, 513), 2)
            if n1 == n2 or ((mask >> (n1 - 1)) & 1) or ((mask >> (n2 - 1)) & 1):
                continue
            if n1 == o1 or n1 == o2 or n2 == o1 or n2 == o2:
                continue
            new_mask = mask
            new_mask &= ~(1 << (o1 - 1))
            new_mask &= ~(1 << (o2 - 1))
            new_mask |= (1 << (n1 - 1))
            new_mask |= (1 << (n2 - 1))
            ns = _score_bitset(new_mask)
            delta = ns - cur_score
        else:
            # Local jitter: shift one element by small amount
            bits = [i + 1 for i in range(512) if (mask >> i) & 1]
            idx = rng.randrange(32)
            old = bits[idx]
            shift = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 6, 8])
            new = old + shift
            if new < 1 or new > 512 or ((mask >> (new - 1)) & 1):
                continue
            new_mask = mask & ~(1 << (old - 1)) | (1 << (new - 1))
            ns = _score_bitset(new_mask)
            delta = ns - cur_score

        # Accept / reject
        if delta >= 0:
            accept = True
        else:
            # Metropolis
            if delta < -20:
                accept = False
            else:
                accept = rng.random() < (0.99 ** (-delta / max(T, 0.01)))
        if accept:
            mask = new_mask
            cur_score = ns
            if cur_score > best_score:
                best_score = cur_score
                best_mask = mask
                stuck = 0
            else:
                stuck += 1
        else:
            stuck += 1

        T *= alpha
        step += 1
        if stuck > max_stuck:
            break

    return _mask_to_arr(best_mask), best_score

def _hill_climb_bitset(init_arr, deadline, rng):
    """
    Local search: try moving one element to each empty position,
    keep the best-improving move; repeat until no improvement.
    """
    mask = _arr_to_mask(init_arr)
    cur_score = _score_bitset(mask)

    # Precompute empty positions once
    empty = [i + 1 for i in range(512) if not ((mask >> i) & 1)]

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle order for diversity
        order = list(range(32))
        rng.shuffle(order)
        bits = [i + 1 for i in range(512) if (mask >> i) & 1]
        for idx in order:
            old = bits[idx]
            # Only try a sample of candidate new positions (all 480 is fine
            # given bitset scoring speed, but we cap for safety)
            best_new = None
            best_ns = cur_score
            # Try each empty position
            for new in empty:
                if new == old:
                    continue
                new_mask = mask & ~(1 << (old - 1)) | (1 << (new - 1))
                ns = _score_bitset(new_mask)
                if ns > best_ns:
                    best_ns = ns
                    best_new = new
            if best_new is not None:
                # Apply move
                mask = mask & ~(1 << (old - 1)) | (1 << (best_new - 1))
                # Update empty list
                empty.remove(old)
                empty.append(best_new)
                cur_score = best_ns
                improved = True
                # Re-read bits since order changed
                bits = [i + 1 for i in range(512) if (mask >> i) & 1]
                break
    return _mask_to_arr(mask), cur_score

def _pair_hill_climb(init_arr, deadline, rng):
    """
    Pair-move local search: swap one element with a random empty position,
    keep if it improves. Iterate until no improvement found in a sweep.
    """
    mask = _arr_to_mask(init_arr)
    cur_score = _score_bitset(mask)
    empty = [i + 1 for i in range(512) if not ((mask >> i) & 1)]
    improved = True
    sweeps = 0
    max_sweeps = 200
    while improved and sweeps < max_sweeps and time.time() < deadline - 2.0:
        improved = False
        sweeps += 1
        bits = [i + 1 for i in range(512) if (mask >> i) & 1]
        for idx in range(32):
            old = bits[idx]
            # Try each empty position
            for new in empty:
                if new == old:
                    continue
                new_mask = mask & ~(1 << (old - 1)) | (1 << (new - 1))
                ns = _score_bitset(new_mask)
                if ns > cur_score:
                    mask = new_mask
                    empty.remove(old)
                    empty.append(new)
                    cur_score = ns
                    improved = True
                    bits = [i + 1 for i in range(512) if (mask >> i) & 1]
                    break
            # Continue with next idx using updated bits if improved
            if improved:
                continue
    return _mask_to_arr(mask), cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill
    rng = random.Random(12345)

    # Generate diverse initializations
    inits = [
        _dense_initial(rng),
        _ap_initial(rng),
        _mix_initial(rng),
        _random_initial(rng),
        _random_initial(rng),
        _ap_initial(rng),
        _mix_initial(rng),
        _dense_initial(rng),
        _random_initial(rng),
    ]

    best_overall = None
    best_overall_score = 0

    # Phase 1: SA on each initialization with time budget
    n_inits = len(inits)
    sa_budget_each = (deadline - start_time - 20.0) / n_inits
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        arr, score = _sa_bitset(init, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr

    if best_overall is None:
        best_overall = _dense_initial(rng)
        best_overall_score = _score_bitset(_arr_to_mask(best_overall))

    # Phase 2: Hill climb polish on best
    if time.time() < deadline - 8.0:
        arr, score = _hill_climb_bitset(best_overall, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr

    # Phase 3: Pair hill climb for additional polish
    if time.time() < deadline - 4.0:
        arr, score = _pair_hill_climb(best_overall, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr

    # Final validity check
    result = sorted(set(best_overall))
    if len(result) != 32:
        result = list(range(1, 33))
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])

    return result
# EVOLVE-BLOCK-END
