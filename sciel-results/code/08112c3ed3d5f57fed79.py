import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0  # safety margin under 90s kill

    # Precompute all 3-APs
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))
    m3 = len(aps_3)

    # For each element x, list of 3-AP indices containing x
    x_aps = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(aps_3):
        x_aps[a].append(i)
        x_aps[b].append(i)
        x_aps[c].append(i)

    # Precompute 4-AP conflict masks
    # For each x, the set of bitmasks of the other 3 terms in each 4-AP containing x
    x_4ap_masks = [[] for _ in range(n + 1)]
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            terms = [a, a + d, a + 2 * d, a + 3 * d]
            for pos in range(4):
                x = terms[pos]
                other_mask = 0
                for j in range(4):
                    if j != pos:
                        other_mask |= (1 << terms[j])
                x_4ap_masks[x].append(other_mask)

    def is_valid_add(x, S_bits):
        """Check if adding x to S_bits creates no 4-AP."""
        for mask in x_4ap_masks[x]:
            if (S_bits & mask) == mask:
                return False
        return True

    def count_3aps(S_bits):
        """Count 3-APs in S_bits."""
        cnt = 0
        for a, b, c in aps_3:
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                cnt += 1
        return cnt

    def gain_of_add(x, S_bits):
        """Count new 3-APs formed by adding x to S_bits."""
        g = 0
        for i in x_aps[x]:
            a, b, c = aps_3[i]
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    g += 1
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1):
                    g += 1
            else:
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1):
                    g += 1
        return g

    def loss_of_remove(x, S_bits):
        """Count 3-APs that would be lost by removing x from S_bits."""
        loss = 0
        for i in x_aps[x]:
            a, b, c = aps_3[i]
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                loss += 1
        return loss

    def greedy_build(order):
        """Build a set greedily in the given order."""
        S_bits = 0
        for x in order:
            if is_valid_add(x, S_bits):
                S_bits |= (1 << x)
        return S_bits

    def local_search(S_bits, max_iters=5000):
        """Local search with add, remove, and swap moves."""
        current_score = count_3aps(S_bits)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 8:
            improved = False
            iters += 1

            # Phase 1: Add any valid element with positive gain
            best_add_gain = 0
            best_add_x = None
            for x in range(1, n + 1):
                if (S_bits >> x) & 1:
                    continue
                if is_valid_add(x, S_bits):
                    g = gain_of_add(x, S_bits)
                    if g > best_add_gain:
                        best_add_gain = g
                        best_add_x = x

            if best_add_x is not None and best_add_gain > 0:
                S_bits |= (1 << best_add_x)
                current_score += best_add_gain
                improved = True
                continue

            # Phase 2: Try swaps
            S_list = [i for i in range(1, n + 1) if (S_bits >> i) & 1]
            if not S_list:
                break

            # Compute gains for additions and losses for removals
            add_candidates = []
            for x in range(1, n + 1):
                if (S_bits >> x) & 1:
                    continue
                if is_valid_add(x, S_bits):
                    g = gain_of_add(x, S_bits)
                    if g > 0:
                        add_candidates.append((x, g))

            if not add_candidates:
                break

            add_candidates.sort(key=lambda kv: -kv[1])

            # Try swaps with top addition candidates
            best_swap_delta = 0
            best_swap = None
            for x, gx in add_candidates[:15]:
                for y in S_list:
                    ly = loss_of_remove(y, S_bits)
                    if gx <= ly:
                        continue
                    # Check validity after swap
                    S_test = (S_bits & ~(1 << y)) | (1 << x)
                    # x is valid in S\{y} since S\{y} ⊆ S
                    # But we need to verify no new 4-APs
                    # Actually, removing y can only help, so if x was valid in S, it's valid in S\{y}
                    # The new 4-APs would need to involve x and 3 elements in S\{y}
                    # Since x was valid in S (which contains S\{y}), it's valid in S\{y}
                    s_new = count_3aps(S_test)
                    delta = s_new - current_score
                    if delta > best_swap_delta:
                        best_swap_delta = delta
                        best_swap = (x, y)

            if best_swap and best_swap_delta > 0:
                x, y = best_swap
                S_bits = (S_bits & ~(1 << y)) | (1 << x)
                current_score += best_swap_delta
                improved = True

        return S_bits, current_score

    def simulated_annealing(S_bits, steps=20000):
        """Simulated annealing to escape local optima."""
        current_score = count_3aps(S_bits)
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        best_S = S_bits
        best_score = current_score

        for step in range(steps):
            if time.time() > deadline - 10:
                break
            if T < T_min:
                break

            # Random move: add, remove, or swap
            move_type = random.random()
            S_list = [i for i in range(1, n + 1) if (S_bits >> i) & 1]

            if move_type < 0.3 and len(S_list) > 5:
                # Remove a random element
                y = random.choice(S_list)
                S_new = S_bits & ~(1 << y)
                s_new = count_3aps(S_new)
            elif move_type < 0.6:
                # Add a random valid element
                valid_adds = []
                for x in range(1, n + 1):
                    if (S_bits >> x) & 1:
                        continue
                    if is_valid_add(x, S_bits):
                        valid_adds.append(x)
                if not valid_adds:
                    continue
                x = random.choice(valid_adds)
                S_new = S_bits | (1 << x)
                s_new = count_3aps(S_new)
            else:
                # Swap
                if not S_list:
                    continue
                y = random.choice(S_list)
                valid_adds = []
                for x in range(1, n + 1):
                    if (S_bits >> x) & 1:
                        continue
                    if is_valid_add(x, S_bits & ~(1 << y)):
                        valid_adds.append(x)
                if not valid_adds:
                    continue
                x = random.choice(valid_adds)
                S_new = (S_bits & ~(1 << y)) | (1 << x)
                s_new = count_3aps(S_new)

            delta = s_new - current_score
            if delta >= 0 or random.random() < np.exp(delta / T):
                S_bits = S_new
                current_score = s_new
                if current_score > best_score:
                    best_score = current_score
                    best_S = S_bits

            T *= cooling

        return best_S, best_score

    # Main: multiple random restarts with SA polish
    best_S_bits = 0
    best_score = 0

    # Natural order greedy
    S1 = greedy_build(range(1, n + 1))
    s1 = count_3aps(S1)
    if s1 > best_score:
        best_score = s1
        best_S_bits = S1

    # Random restarts
    rng = random.Random(42)
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S_cand = greedy_build(order)
        S_cand, s_cand = local_search(S_cand, max_iters=3000)
        if s_cand > best_score:
            best_score = s_cand
            best_S_bits = S_cand

    # SA polish on best
    if time.time() < deadline - 15:
        best_S_bits, best_score = simulated_annealing(best_S_bits, steps=30000)

    # Final local search
    if time.time() < deadline - 5:
        best_S_bits, best_score = local_search(best_S_bits, max_iters=5000)

    result = [i for i in range(1, n + 1) if (best_S_bits >> i) & 1]
    return sorted(result)
