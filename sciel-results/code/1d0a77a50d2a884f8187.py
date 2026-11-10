import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]

    # Precompute all 3-APs: list of (a, b, c) where a < b < c and a+c=2b
    three_aps = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            three_aps.append((a, b, c))

    # Precompute conflict pairs for each element
    # conflict_pairs[x] = list of (y, z) such that {x,y,z} forms a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    for a, b, c in three_aps:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # Precompute bitmask pairs for fast conflict checking
    conflict_pair_masks = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_pair_masks[x].append((1 << y, 1 << z))

    def can_add(x, S_mask):
        """Check if x can be added to set S_mask without creating a 3-AP."""
        for my, mz in conflict_pair_masks[x]:
            if (S_mask & my) and (S_mask & mz):
                return False
        return True

    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def get_elements(S_mask):
        """Get list of elements in bitmask."""
        return [x for x in range(1, N + 1) if S_mask & (1 << x)]

    def greedy_fill(forbidden_mask, S_mask, order=None):
        """Fill S by adding elements in a given order."""
        S = S_mask
        if order is None:
            order = range(1, N + 1)
        for x in order:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    def simulated_annealing(S_init, max_time):
        """
        Simulated annealing on the QUBO-inspired energy landscape.
        State: bitmask S
        Energy: -sum(recip[x] for x in S) + penalty * (number of 3-APs in S)
        We want to minimize energy = maximize reciprocal sum subject to no 3-APs.
        Moves: flip one bit (add or remove an element)
        """
        S = S_init
        current_sum = get_float_sum(S)
        current_aps = 0  # Number of 3-APs in current set (should be 0 for valid states)

        # Count 3-APs in initial state
        S_set = set(get_elements(S))
        for a, b, c in three_aps:
            if a in S_set and b in S_set and c in S_set:
                current_aps += 1

        best_S = S
        best_sum = current_sum
        best_energy = -current_sum + 1000.0 * current_aps

        # Annealing parameters
        T_start = 0.5
        T_end = 0.001
        alpha = 0.9995  # Cooling rate
        T = T_start

        rng = random.Random(42)

        while (time.time() - start_time) < max_time and T > T_end:
            # Choose a random element to flip
            x = rng.randint(1, N)
            bit = 1 << x

            if S & bit:
                # Try removing x
                S_new = S & ~bit
                sum_new = current_sum - recip[x]

                # Count change in 3-APs
                # Removing x can only reduce AP count
                ap_change = 0
                S_set = set(get_elements(S))
                S_set.discard(x)
                for a, b, c in three_aps:
                    if x in (a, b, c) and a in S_set and b in S_set and c in S_set:
                        ap_change -= 1

                aps_new = current_aps + ap_change
                energy_new = -sum_new + 1000.0 * aps_new
                energy_cur = -current_sum + 1000.0 * current_aps
                delta_e = energy_new - energy_cur

                if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                    S = S_new
                    current_sum = sum_new
                    current_aps = aps_new
                    if current_aps == 0 and current_sum > best_sum:
                        best_S = S
                        best_sum = current_sum
                        best_energy = -current_sum
            else:
                # Try adding x
                if not can_add(x, S):
                    T *= alpha
                    continue

                S_new = S | bit
                sum_new = current_sum + recip[x]

                # Adding x can only increase AP count
                ap_change = 0
                S_set = set(get_elements(S))
                S_set.add(x)
                for a, b, c in three_aps:
                    if x in (a, b, c) and a in S_set and b in S_set and c in S_set:
                        ap_change += 1

                aps_new = current_aps + ap_change
                energy_new = -sum_new + 1000.0 * aps_new
                energy_cur = -current_sum + 1000.0 * current_aps
                delta_e = energy_new - energy_cur

                if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                    S = S_new
                    current_sum = sum_new
                    current_aps = aps_new
                    if current_aps == 0 and current_sum > best_sum:
                        best_S = S
                        best_sum = current_sum
                        best_energy = -current_sum

            T *= alpha

        return best_S, best_sum

    # Step 1: Generate diverse initializations
    initializations = []

    # Greedy ascending
    S_asc = greedy_fill(0, 0, order=range(1, N + 1))
    initializations.append(S_asc)

    # Greedy descending
    S_desc = greedy_fill(0, 0, order=range(N, 0, -1))
    initializations.append(S_desc)

    # Greedy with random shuffles
    for trial in range(20):
        if (time.time() - start_time) > deadline - 25:
            break
        random.seed(trial * 12345 + 67890)
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy_fill(0, 0, order=order)
        initializations.append(S_rand)

    # Weighted random greedy (favor smaller numbers)
    for trial in range(10):
        if (time.time() - start_time) > deadline - 25:
            break
        random.seed(100000 + trial * 7777)
        weights = [recip[i] for i in range(1, N + 1)]
        order = []
        remaining = list(range(1, N + 1))
        rem_weights = weights[:]
        while remaining:
            total_w = sum(rem_weights)
            if total_w <= 0:
                order.extend(remaining)
                break
            r = random.random() * total_w
            cum = 0
            for idx, w in enumerate(rem_weights):
                cum += w
                if r <= cum:
                    order.append(remaining[idx])
                    remaining.pop(idx)
                    rem_weights.pop(idx)
                    break
        S_wrand = greedy_fill(0, 0, order=order)
        initializations.append(S_wrand)

    # Step 2: Run simulated annealing from each initialization
    best_S = 0
    best_sum = 0.0

    # Sort initializations by quality
    init_with_sums = []
    for S_init in initializations:
        s = get_float_sum(S_init)
        init_with_sums.append((s, S_init))
    init_with_sums.sort(key=lambda x: -x[0])

    time_per_sa = max(3.0, (deadline - 12 - (time.time() - start_time)) / max(1, len(init_with_sums)))

    for i, (init_sum, S_init) in enumerate(init_with_sums):
        if (time.time() - start_time) > deadline - 10:
            break

        remaining_time = deadline - 10 - (time.time() - start_time)
        if remaining_time < 2:
            break

        # Allocate more time to better initial solutions
        alloc = min(time_per_sa, remaining_time)
        if i < 3:
            alloc = min(alloc * 1.5, remaining_time)

        S_sa, sum_sa = simulated_annealing(S_init, alloc)

        if sum_sa > best_sum:
            best_sum = sum_sa
            best_S = S_sa

    # Step 3: Final polish with extended SA
    if best_S != 0:
        remaining = deadline - 4 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = simulated_annealing(best_S, remaining)
            if sum_final > best_sum:
                best_S = S_final
                best_sum = sum_final

    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]

    # Validate
    S_set = set(result)
    valid = True
    for a, b, c in three_aps:
        if a in S_set and b in S_set and c in S_set:
            valid = False
            break

    if not valid:
        # Fallback to greedy result
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]

    return result
