"""Optimized search for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set for some a."""
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count

    def greedy_initialize(rng):
        """Greedy initialization: pick elements that maximize 3-AP coverage."""
        current_set = set()
        current_bit = 0
        current_score = 0
        candidates = list(range(1, 513))
        rng.shuffle(candidates)

        for _ in range(32):
            if not candidates:
                break
            best_elem = None
            best_gain = -1
            sample_size = min(80, len(candidates))
            sample = rng.sample(candidates, sample_size)
            for elem in sample:
                if elem in current_set:
                    continue
                new_bit = current_bit | (1 << (elem - 1))
                new_score = score_bitset(new_bit)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_elem = elem
            if best_elem is not None:
                current_set.add(best_elem)
                current_bit |= (1 << (best_elem - 1))
                current_score += best_gain
                candidates.remove(best_elem)
            else:
                elem = candidates.pop()
                current_set.add(elem)
                current_bit |= (1 << (elem - 1))
                current_score = score_bitset(current_bit)

        return list(current_set), current_bit, current_score

    def simulated_annealing(initial_set, initial_bit, initial_score, rng, time_budget):
        """Run simulated annealing starting from given state."""
        current_set = list(initial_set)
        current_bit = initial_bit
        current_score = initial_score

        best_set = list(current_set)
        best_bit = current_bit
        best_score = current_score

        temperature = 40.0
        cooling = 0.99997
        min_temp = 0.01
        e = 2.718281828459045

        while temperature > min_temp:
            if time.time() > time_budget:
                break

            move_type = rng.random()

            if move_type < 0.5:
                # Single swap
                i = rng.randrange(32)
                old_elem = current_set[i]
                new_elem = rng.randint(1, 512)
                if new_elem == old_elem or new_elem in current_set:
                    continue

                new_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
                new_score = score_bitset(new_bit)

                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new_elem
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            elif move_type < 0.75:
                # Double swap
                i = rng.randrange(32)
                j = rng.randrange(32)
                if i == j:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                new1 = rng.randint(1, 512)
                new2 = rng.randint(1, 512)
                if new1 in current_set or new2 in current_set or new1 == new2:
                    continue

                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_score = score_bitset(new_bit)

                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            elif move_type < 0.9:
                # Triple swap
                i = rng.randrange(32)
                j = rng.randrange(32)
                k = rng.randrange(32)
                if i == j or i == k or j == k:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                old3 = current_set[k]
                new1 = rng.randint(1, 512)
                new2 = rng.randint(1, 512)
                new3 = rng.randint(1, 512)
                if new1 in current_set or new2 in current_set or new3 in current_set:
                    continue
                if len({new1, new2, new3}) < 3:
                    continue

                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit &= ~(1 << (old3 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_bit |= (1 << (new3 - 1))
                new_score = score_bitset(new_bit)

                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_set[k] = new3
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            else:
                # Quadruple swap
                i = rng.randrange(32)
                j = rng.randrange(32)
                k = rng.randrange(32)
                m = rng.randrange(32)
                if len({i, j, k, m}) < 4:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                old3 = current_set[k]
                old4 = current_set[m]
                new1 = rng.randint(1, 512)
                new2 = rng.randint(1, 512)
                new3 = rng.randint(1, 512)
                new4 = rng.randint(1, 512)
                if len({new1, new2, new3, new4}) < 4:
                    continue
                if new1 in current_set or new2 in current_set or new3 in current_set or new4 in current_set:
                    continue

                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit &= ~(1 << (old3 - 1))
                new_bit &= ~(1 << (old4 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_bit |= (1 << (new3 - 1))
                new_bit |= (1 << (new4 - 1))
                new_score = score_bitset(new_bit)

                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_set[k] = new3
                    current_set[m] = new4
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit

            temperature *= cooling

        return best_set, best_bit, best_score

    def local_search_polish(s, b, sc, rng, max_iters):
        """Intensive local search: try all single swaps, then double swaps."""
        current_set = list(s)
        current_bit = b
        current_score = sc
        best_set = list(current_set)
        best_bit = current_bit
        best_score = current_score

        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            improved = False

            # Try all single swaps
            for i in range(32):
                old_elem = current_set[i]
                for new_elem in range(1, 513):
                    if new_elem == old_elem or new_elem in current_set:
                        continue
                    new_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
                    new_score = score_bitset(new_bit)
                    if new_score > current_score:
                        current_set[i] = new_elem
                        current_bit = new_bit
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                            best_bit = current_bit
                        break

            if not improved:
                # Try random double swaps
                for _ in range(200):
                    if time.time() > deadline - 2.0:
                        break
                    i = rng.randrange(32)
                    j = rng.randrange(32)
                    if i == j:
                        continue
                    old1 = current_set[i]
                    old2 = current_set[j]
                    new1 = rng.randint(1, 512)
                    new2 = rng.randint(1, 512)
                    if new1 in current_set or new2 in current_set or new1 == new2:
                        continue

                    new_bit = current_bit
                    new_bit &= ~(1 << (old1 - 1))
                    new_bit &= ~(1 << (old2 - 1))
                    new_bit |= (1 << (new1 - 1))
                    new_bit |= (1 << (new2 - 1))
                    new_score = score_bitset(new_bit)

                    if new_score > current_score:
                        current_set[i] = new1
                        current_set[j] = new2
                        current_bit = new_bit
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                            best_bit = current_bit

            if not improved:
                break

        return best_set, best_bit, best_score

    # Multi-restart strategy
    global_best_set = None
    global_best_bit = 0
    global_best_score = -1

    restarts = 8
    time_per_restart = (deadline - time.time()) / restarts

    for restart in range(restarts):
        if time.time() > deadline - 10.0:
            break

        rng = random.Random(1000 + restart * 7919)

        # Greedy initialization
        init_set, init_bit, init_score = greedy_initialize(rng)

        # Simulated annealing
        sa_time_budget = min(deadline - 15.0, time.time() + time_per_restart)
        best_set, best_bit, best_score = simulated_annealing(
            init_set, init_bit, init_score, rng, sa_time_budget
        )

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_set = best_set
            global_best_bit = best_bit

    # Final intensive polish on the best solution
    if global_best_set is not None and time.time() < deadline - 10.0:
        polish_rng = random.Random(99999)
        max_iters = 50
        global_best_set, global_best_bit, global_best_score = local_search_polish(
            global_best_set, global_best_bit, global_best_score, polish_rng, max_iters
        )

    # Second round of SA from the polished solution if time permits
    if global_best_set is not None and time.time() < deadline - 15.0:
        rng2 = random.Random(77777)
        sa2_budget = deadline - 5.0
        s2, b2, sc2 = simulated_annealing(
            global_best_set, global_best_bit, global_best_score, rng2, sa2_budget
        )
        if sc2 > global_best_score:
            global_best_score = sc2
            global_best_set = s2
            global_best_bit = b2

    # Final polish
    if global_best_set is not None and time.time() < deadline - 5.0:
        polish_rng2 = random.Random(88888)
        global_best_set, global_best_bit, global_best_score = local_search_polish(
            global_best_set, global_best_bit, global_best_score, polish_rng2, 30
        )

    return global_best_set if global_best_set is not None else list(range(1, 33))
