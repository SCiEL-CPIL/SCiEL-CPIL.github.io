import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    # Precompute coverage masks for each candidate x in [0, 128]
    # cov_mask[x] is a 129-bit integer: bit i is 1 if x can participate in sum N_LOW+i
    # i.e., there exists y in [0,128] with y != x (or y == x) such that x + y = N_LOW + i
    # and both x, y are in [0, 128]
    cov_mask = np.zeros(MAX_VAL + 1, dtype=np.uint64)
    for x in range(MAX_VAL + 1):
        mask = 0
        # x pairs with y where y = s - x, s in [128, 256], y in [0, 128]
        # So s-x in [0, 128] => s in [x, x+128]
        # s also in [128, 256]
        # So s in [max(128, x), min(256, x+128)]
        s_min = max(N_LOW, x)
        s_max = min(N_HIGH, x + MAX_VAL)
        for s in range(s_min, s_max + 1):
            y = s - x
            if 0 <= y <= MAX_VAL:
                idx = s - N_LOW
                mask |= (1 << idx)
        cov_mask[x] = mask

    # Precompute: for each x, the set of sums where x is the "smaller" or "equal" element
    # In unordered pairs (a,b) with a<=b, x participates as 'a' if x <= s-x, i.e., s >= 2x
    # and as 'b' if x > s-x, i.e., s < 2x
    # For counting, each unordered pair (a,b) with a<b contributes to one sum.
    # (a,a) contributes to one sum.
    # When we add element x to A, the new pairs are (x, y) for all y in A where x <= y,
    # plus (x, x) if we're adding x for the first time.
    # Actually, it's simpler: the new pairs involving x are (min(x,y), max(x,y)) for all y in A.
    # Each such pair contributes to sum x+y.

    # For the load calculation, when we add x to current set A:
    # New pairs: (x, y) for each y in A where y >= x (unordered, so we count each pair once)
    # Plus (x, x) if x not already in A (which it isn't since we're adding)
    # Wait, we need to be careful. Let's think of it differently.
    # 
    # The representation count for sum s is the number of unordered pairs {a,b} with a<=b, a+b=s, a,b in A.
    # When we add x to A, the new pairs are {x, y} for all y in A where y >= x, plus {x, x}.
    # Actually {x,x} is just one pair. And {x,y} for y in A, y > x is another set.
    # But we also need y in A. So the new pairs are:
    #   {x, y} for each y in A where y >= x  (this includes y=x if x in A, but x is not in A yet)
    #   So it's {x, y} for each y in A where y > x, plus {x, x}.
    #   Wait, if y < x, then the pair is {y, x} which was already counted when y was added.
    #   So yes, new pairs from adding x: {x, y} for y in A, y >= x. Since x not in A, y > x.
    #   Plus {x, x}.

    # Let me recompute: for each x, precompute the "load contribution" mask
    # load_mask[x] = set of sums s where x can form a pair with some y >= x
    # But this depends on which y are in A. So it's not precomputable in a simple way.
    # 
    # Alternative: just track the current load array and update it incrementally.
    # When adding x to A:
    #   For each y in A where y > x: load[x+y] += 1
    #   load[2*x] += 1  (for the pair {x,x})
    # When removing x from A:
    #   For each y in A where y > x: load[x+y] -= 1
    #   load[2*x] -= 1

    def calc_reps_fast(A_set):
        """Compute representation counts using the incremental method."""
        A_sorted = sorted(A_set)
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        n = len(A_sorted)
        for i in range(n):
            ai = A_sorted[i]
            # Pair (ai, ai)
            s = 2 * ai
            if N_LOW <= s <= N_HIGH:
                reps[s - N_LOW] += 1
            # Pairs (ai, aj) for j > i
            for j in range(i + 1, n):
                s = ai + A_sorted[j]
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    # Phase 1: Bitset-based greedy set cover
    # Start with empty set, greedily add elements that cover the most uncovered sums
    # while trying to keep loads balanced

    best_A = None
    best_peak = float('inf')

    # Try multiple random initializations and greedy strategies
    for trial in range(20):
        if time.time() > deadline - 10:
            break

        random.seed(trial * 7919 + 13)

        # Start with a few random elements to break symmetry
        A_set = set()
        # Always include some structure: try to include elements near 0 and 128
        if trial % 2 == 0:
            A_set.add(0)
            A_set.add(128)
        else:
            A_set.add(random.randint(0, 32))
            A_set.add(random.randint(96, 128))

        # Compute current coverage and loads
        # Use a dict for loads: sum -> count
        loads = {}
        for x in A_set:
            for y in A_set:
                if x <= y:
                    s = x + y
                    if N_LOW <= s <= N_HIGH:
                        loads[s] = loads.get(s, 0) + 1

        # Convert to array for efficiency
        current_loads = np.zeros(RANGE_SIZE, dtype=np.int32)
        for s, c in loads.items():
            current_loads[s - N_LOW] = c

        uncovered_mask = 0
        for s in range(N_LOW, N_HIGH + 1):
            if current_loads[s - N_LOW] == 0:
                uncovered_mask |= (1 << (s - N_LOW))

        # Greedy: repeatedly add the element that covers the most uncovered sums
        # with the least additional load on already-loaded sums
        max_iters = 100
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break

            if uncovered_mask == 0:
                break

            # Find the best element to add
            best_x = None
            best_score = -1
            best_new_loads = None
            best_new_uncovered = uncovered_mask

            candidates = list(range(0, MAX_VAL + 1))
            random.shuffle(candidates)

            for x in candidates:
                if x in A_set:
                    continue

                # Compute new loads if we add x
                # New pairs: (x, y) for y in A_set where y >= x, plus (x, x)
                new_loads = current_loads.copy()
                for y in A_set:
                    if y >= x:
                        s = x + y
                        if N_LOW <= s <= N_HIGH:
                            new_loads[s - N_LOW] += 1
                s = 2 * x
                if N_LOW <= s <= N_HIGH:
                    new_loads[s - N_LOW] += 1

                # Compute new uncovered
                new_uncovered = 0
                for s in range(N_LOW, N_HIGH + 1):
                    if new_loads[s - N_LOW] == 0:
                        new_uncovered |= (1 << (s - N_LOW))

                # Score: number of newly covered sums - penalty for increasing peak
                newly_covered = bin(uncovered_mask & ~new_uncovered).count('1')
                peak = int(np.max(new_loads))
                # Penalize high peaks
                score = newly_covered * 10 - peak

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_loads = new_loads
                    best_new_uncovered = new_uncovered

            if best_x is None:
                break

            A_set.add(best_x)
            current_loads = best_new_loads
            uncovered_mask = best_new_uncovered

        # Check if we have a valid solution
        if uncovered_mask == 0:
            peak = int(np.max(current_loads))
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_set)

    # Fallback: use a known valid construction
    if best_A is None:
        # A = {64, 65, ..., 128} covers [128, 256]
        best_A = list(range(64, 129))
        best_peak = get_peak(calc_reps_fast(set(best_A)))

    # Phase 2: Local search to reduce peak
    if time.time() < deadline - 5:
        current_A = set(best_A)
        current_loads = calc_reps_fast(current_A)
        current_peak = int(np.max(current_loads))

        # Simulated annealing with targeted moves
        T = 2.0
        cooling_rate = 0.9998
        min_T = 0.1

        while time.time() < deadline - 3 and T > min_T:
            if len(current_A) == 0:
                break

            # Choose a move type: remove, add, or swap
            move_type = random.random()
            if move_type < 0.3 and len(current_A) > 10:
                # Remove an element
                x_to_remove = random.choice(list(current_A))
                new_A = current_A - {x_to_remove}
                if len(new_A) == 0:
                    T *= cooling_rate
                    continue
                new_loads = calc_reps_fast(new_A)
                if not is_covered(new_loads):
                    T *= cooling_rate
                    continue
                new_peak = int(np.max(new_loads))
                delta = new_peak - current_peak
                if delta < 0:
                    current_A = new_A
                    current_loads = new_loads
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_loads = new_loads
                        current_peak = new_peak
            elif move_type < 0.6:
                # Add an element
                x_to_add = random.randint(0, MAX_VAL)
                if x_to_add in current_A:
                    T *= cooling_rate
                    continue
                new_A = current_A | {x_to_add}
                new_loads = calc_reps_fast(new_A)
                new_peak = int(np.max(new_loads))
                delta = new_peak - current_peak
                if delta < 0:
                    current_A = new_A
                    current_loads = new_loads
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_loads = new_loads
                        current_peak = new_peak
            else:
                # Swap: remove one, add another
                x_to_remove = random.choice(list(current_A))
                x_to_add = random.randint(0, MAX_VAL)
                if x_to_add == x_to_remove:
                    T *= cooling_rate
                    continue
                new_A = set(current_A)
                new_A.remove(x_to_remove)
                new_A.add(x_to_add)
                if len(new_A) == 0:
                    T *= cooling_rate
                    continue
                new_loads = calc_reps_fast(new_A)
                if not is_covered(new_loads):
                    T *= cooling_rate
                    continue
                new_peak = int(np.max(new_loads))
                delta = new_peak - current_peak
                if delta < 0:
                    current_A = new_A
                    current_loads = new_loads
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_loads = new_loads
                        current_peak = new_peak

            T *= cooling_rate

    # Phase 3: Final removal pass
    if time.time() < deadline - 2:
        current_A = set(best_A)
        current_loads = calc_reps_fast(current_A)
        current_peak = int(np.max(current_loads))

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in list(current_A):
                test_A = current_A - {x}
                if len(test_A) == 0:
                    continue
                test_loads = calc_reps_fast(test_A)
                if is_covered(test_loads):
                    test_peak = int(np.max(test_loads))
                    if test_peak <= current_peak:
                        current_A = test_A
                        current_loads = test_loads
                        current_peak = test_peak
                        improved = True
                        break

        if is_covered(current_loads):
            peak = int(np.max(current_loads))
            if peak < best_peak:
                best_peak = peak
                best_A = list(current_A)

    return best_A

