"""Combined program: genetic algorithm (diverse exploration) + fast delta local search (refinement)."""
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns list of lists: ap_list[i] = list of (a, d, k) where position i is in AP at index k."""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((a, d, k))
    return ap_list

def _delta_flip(c, i, ap_list):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    delta = 0
    for (a, d, k) in ap_list[i]:
        idxs = [a, a + d, a + 2 * d, a + 3 * d]
        vals = [c[j] for j in idxs]
        was_mono = (vals[0] == vals[1] == vals[2] == vals[3])
        # After flip: position i has flipped
        if k == 0:
            new_vals = [1 - c[a], c[a+d], c[a+2*d], c[a+3*d]]
        elif k == 1:
            new_vals = [c[a], 1 - c[a+d], c[a+2*d], c[a+3*d]]
        elif k == 2:
            new_vals = [c[a], c[a+d], 1 - c[a+2*d], c[a+3*d]]
        else:
            new_vals = [c[a], c[a+d], c[a+2*d], 1 - c[a+3*d]]
        is_mono = (new_vals[0] == new_vals[1] == new_vals[2] == new_vals[3])
        if was_mono and not is_mono:
            delta -= 1
        elif not was_mono and is_mono:
            delta += 1
    return delta

def _local_search(c, time_limit, ap_list, max_passes=50):
    """Improve coloring via single-flip local search using delta computation.
    Returns (c, count)."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(N):
            if time.time() - start > time_limit:
                return c, current
            delta = _delta_flip(c, i, ap_list)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            return c, current
    return c, current

def _random_coloring(rng):
    """Generate a random 0/1 coloring."""
    return rng.integers(0, 2, size=N).astype(np.int8)

def _mutate(c, rng, rate=0.03):
    """Bit-flip mutation with given rate."""
    flips = rng.random(N) < rate
    c[flips] = 1 - c[flips]
    return c

def _crossover(p1, p2, rng):
    """Uniform crossover: for each position, pick from p1 or p2 with 50% probability."""
    mask = rng.integers(0, 2, size=N).astype(np.int8)
    child = np.where(mask == 1, p1, p2)
    return child.copy()

def construct():
    start_time = time.time()

    # Precompute AP membership for fast delta computation
    ap_list = _precompute_ap_membership()

    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)

    # Local search from Thue-Morse
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        c, cnt = _local_search(c, remaining - 10, ap_list)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()

    # Genetic Algorithm with diverse population + fast local search refinement
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed

    if remaining > 20:
        pop_size = 16
        elite_size = 4
        mutation_rate = 0.04

        # Initialize population: best solution + diverse starts
        population = [best_c.copy()]
        scores = [best_count]

        # Add Thue-Morse mutated variants and random starts for diversity
        tm = _thue_morse()
        population.append(tm.copy())
        scores.append(_count_mono4(tm))

        for _ in range(pop_size - 2):
            if len(population) < pop_size:
                if rng.random() < 0.5:
                    c_rand = _random_coloring(rng)
                else:
                    c_rand = best_c.copy()
                    c_rand = _mutate(c_rand, rng, rate=0.15)
                # Quick local search to improve
                time_left = DEADLINE - (time.time() - start_time)
                if time_left > 15:
                    c_rand, cnt_rand = _local_search(c_rand, min(2.0, time_left / (pop_size + 5)), ap_list, max_passes=20)
                else:
                    cnt_rand = _count_mono4(c_rand)
                population.append(c_rand)
                scores.append(cnt_rand)

        # Update best from initial population
        for i in range(len(population)):
            if scores[i] < best_count:
                best_count = scores[i]
                best_c = population[i].copy()

        # GA loop
        gen = 0
        while time.time() - start_time < DEADLINE - 8:
            gen += 1

            # Sort by score
            sorted_idx = np.argsort(scores)
            population = [population[i].copy() for i in sorted_idx]
            scores = [scores[i] for i in sorted_idx]

            # Elitism: keep best elite_size
            new_population = [population[i].copy() for i in range(min(elite_size, len(population)))]
            new_scores = [scores[i] for i in range(min(elite_size, len(population)))]

            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() - start_time > DEADLINE - 8:
                    break

                # Tournament selection
                def tournament_select(pop, sc):
                    idxs = rng.integers(0, len(pop), size=3)
                    best_idx = idxs[np.argmin([sc[i] for i in idxs])]
                    return pop[best_idx].copy()

                p1 = tournament_select(population, scores)
                p2 = tournament_select(population, scores)

                # Crossover
                child = _crossover(p1, p2, rng)

                # Mutation
                child = _mutate(child, rng, mutation_rate)

                # Fast local search on child (the refinement machinery)
                time_left_for_ls = min(1.5, (DEADLINE - 8 - (time.time() - start_time)) / 8)
                if time_left_for_ls > 0.1:
                    child, cnt_child = _local_search(child, time_left_for_ls, ap_list, max_passes=15)
                else:
                    cnt_child = _count_mono4(child)

                new_population.append(child)
                new_scores.append(cnt_child)

            population = new_population
            scores = new_scores

            # Update best
            min_score_idx = int(np.argmin(scores))
            if scores[min_score_idx] < best_count:
                best_count = scores[min_score_idx]
                best_c = population[min_score_idx].copy()

    # Final polish: more thorough local search on best solution
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        c = best_c.copy()
        c, cnt = _local_search(c, remaining - 3, ap_list, max_passes=100)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()

    return [int(x) for x in best_c]
