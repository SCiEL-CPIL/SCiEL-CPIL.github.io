import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def to_int(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val

    def to_digits(val):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = val % q
            val //= q
        return digits

    # Precompute balls
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = to_digits(w)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv != orig:
                    digits[pos] = nv
                    ball.append(to_int(digits))
            digits[pos] = orig
        balls[w] = ball

    # Precompute inverse index: for each word, which centers cover it
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = set()
            centers_covering[w].add(i)

    def structured_init():
        """Tetracode product initialization for q=3, n=8."""
        if q != 3 or n != 8:
            return greedy_init()

        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list(set(to_int(c) for c in product_code))

        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1

        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            gains = uncovered[balls].sum(axis=1)
            code_set = set(code_ints)
            for c in code_set:
                gains[c] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best = int(np.argmax(gains))
            code_ints.append(best)
            cover_count[balls[best]] += 1
            uncovered = (cover_count == 0)

        return code_ints, cover_count

    def greedy_init():
        """Standard greedy set cover."""
        uncovered = np.ones(total, dtype=bool)
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            best = int(np.argmax(gains))
            code.append(best)
            uncovered[balls[best]] = False
            cover_count[balls[best]] += 1
        return code, cover_count

    def trim_code(code, cover_count):
        """Remove all redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.all(cover_count[balls[c]] > 1):
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iters=50000):
        """Targeted local search using inverse index for fast candidate finding."""
        code_set = set(code)
        improved = True
        iter_count = 0

        while improved and time.time() < deadline - 5 and iter_count < max_iters:
            improved = False
            iter_count += 1
            rng = random.Random(iter_count)
            code_indices = list(range(len(code)))
            rng.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline - 5:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates covering all only_by_i using inverse index
                candidate_set = None
                for w in only_by_i:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                # Exclude current code
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Score candidates: how many uniquely-covered points they help
                best_cand = None
                best_score = -1
                for cand in candidate_set:
                    # After removing i, points in i_ball lose one cover
                    # cand covers a point p if p in balls[cand]
                    # Effective: count points in balls[cand] that would be uncovered
                    # i.e., cover_count[p] == 1 and p in i_ball (so removing i makes it 0)
                    # OR cover_count[p] == 0 (shouldn't happen if valid)
                    bp_cand = balls[cand]
                    # Points that become uncovered after removing i:
                    # those in i_ball with cover_count == 1
                    # cand helps if it covers any of those
                    help_count = 0
                    for p in only_by_i:
                        if p in set(bp_cand.tolist()):
                            help_count += 1
                    if help_count > best_score:
                        best_score = help_count
                        best_cand = cand

                if best_cand is not None and best_score > 0:
                    # Perform swap
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    code[ci] = best_cand
                    code_set.add(best_cand)
                    cover_count[balls[best_cand]] += 1
                    improved = True
                    break

        return code, cover_count

    def genetic_algorithm(code, cover_count, pop_size=20, generations=50):
        """
        Genetic algorithm on binary genomes representing codeword presence.
        Fitness: minimize codeword count, penalize coverage gaps.
        """
        code_set = set(code)
        best_code = list(code)
        best_cover = cover_count.copy()
        best_fitness = len(code)

        # Create initial population
        population = []
        for c in best_code:
            cover_count[balls[c]] += 1

        # Population: list of (code_list, cover_count, fitness)
        pop = [(list(best_code), best_cover.copy(), len(best_code))]

        # Add some perturbed versions
        rng = random.Random(42)
        for _ in range(pop_size - 1):
            perturbed = list(best_code)
            # Randomly remove a few
            for _ in range(min(3, len(perturbed))):
                idx = rng.randint(0, len(perturbed) - 1)
                c = perturbed.pop(idx)
            # Randomly add a few
            for _ in range(min(3, total)):
                c = rng.randint(0, total - 1)
                if c not in perturbed:
                    perturbed.append(c)
            # Compute cover count
            pc = np.zeros(total, dtype=np.int32)
            for c in perturbed:
                pc[balls[c]] += 1
            uncovered = (pc == 0).sum()
            fitness = len(perturbed) + uncovered * 10
            pop.append((perturbed, pc, fitness))

        for gen in range(generations):
            if time.time() > deadline - 10:
                break

            # Sort by fitness
            pop.sort(key=lambda x: x[2])

            # Keep best
            if pop[0][2] < best_fitness:
                best_fitness = pop[0][2]
                best_code = list(pop[0][0])
                best_cover = pop[0][1].copy()

            # Create next generation
            new_pop = []

            # Elitism: keep top half
            elite_count = max(1, len(pop) // 2)
            new_pop.extend(pop[:elite_count])

            # Crossover and mutation
            while len(new_pop) < pop_size:
                if time.time() > deadline - 10:
                    break

                # Select parents (tournament)
                p1_idx = rng.randint(0, len(pop) - 1)
                p2_idx = rng.randint(0, len(pop) - 1)
                # Tournament selection
                for _ in range(3):
                    c1 = rng.randint(0, len(pop) - 1)
                    if pop[c1][2] < pop[p1_idx][2]:
                        p1_idx = c1
                for _ in range(3):
                    c2 = rng.randint(0, len(pop) - 1)
                    if pop[c2][2] < pop[p2_idx][2]:
                        p2_idx = c2

                parent1 = pop[p1_idx][0]
                parent2 = pop[p2_idx][0]

                # Crossover: uniform crossover on codeword sets
                child_set = set(parent1) | set(parent2)
                # Randomly drop some
                child_list = list(child_set)
                rng.shuffle(child_list)
                # Keep a random subset
                keep_count = max(1, len(child_list) - rng.randint(0, 5))
                child_list = child_list[:keep_count]

                # Mutation: add/remove random codewords
                for _ in range(rng.randint(0, 3)):
                    if child_list and rng.random() < 0.5:
                        idx = rng.randint(0, len(child_list) - 1)
                        child_list.pop(idx)
                    else:
                        c = rng.randint(0, total - 1)
                        if c not in child_list:
                            child_list.append(c)

                # Compute fitness
                pc = np.zeros(total, dtype=np.int32)
                for c in child_list:
                    pc[balls[c]] += 1
                uncovered = (pc == 0).sum()
                fitness = len(child_list) + uncovered * 10
                new_pop.append((child_list, pc, fitness))

            pop = new_pop

        # Final trim
        best_code, best_cover = trim_code(best_code, best_cover)
        return best_code, best_cover

    # Main execution
    # Try structured init first
    code, cover_count = structured_init()
    code, cover_count = trim_code(code, cover_count)
    code, cover_count = targeted_local_search(code, cover_count, max_iters=30000)

    if time.time() < deadline - 20:
        # Try genetic algorithm
        code, cover_count = genetic_algorithm(code, cover_count, pop_size=15, generations=30)
        code, cover_count = trim_code(code, cover_count)
        code, cover_count = targeted_local_search(code, cover_count, max_iters=20000)

    # Final cleanup
    code, cover_count = trim_code(code, cover_count)

    # Convert to word lists
    result = [to_digits(w) for w in code]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
