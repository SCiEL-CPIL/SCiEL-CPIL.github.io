import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Compute 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    Uses efficient numpy computation.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0

    U_arr = np.array(U, dtype=np.int64)

    # All pairwise sums i<=j
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))

    # All pairwise differences i<j (positive), then |U-U| = 2*num_pos + 1
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1

    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = int(U_arr[-1])
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1.0 + log(ratio) / log(D)


def score_fast_small(U):
    """Python-only scorer for very small sets (n<=20), avoids numpy overhead."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            diffs.add(U[j] - U[i])
    num_sums = len(sums)
    num_diffs = 2 * len(diffs) + 1
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1.0 + log(ratio) / log(D)


def greedy_golomb(n, start=1):
    """
    Greedy construction of a Golomb ruler with n marks including 0.
    Returns sorted list [0, a1, a2, ..., a_{n-1}].
    Uses the classic greedy algorithm: place each new mark as far right as
    possible while keeping all differences distinct.
    """
    if n < 1:
        return [0]
    marks = [0]
    diffs = set()
    for _ in range(1, n):
        # Try to place the new mark as far right as possible
        # Start from a large candidate and work down
        # The new mark x must satisfy: for all existing m, (x - m) not in diffs
        # and (x - m) > 0
        # We want x as large as possible but bounded by a reasonable span.
        # Use a heuristic upper bound: 2 * current_max + 10
        cur_max = max(marks)
        # Try candidates from high to low
        # Upper bound: we want a compact ruler, so try up to cur_max + cur_max
        upper = cur_max * 2 + 20
        lower = cur_max + 1
        # Try from upper down to lower
        for x in range(upper, lower - 1, -1):
            ok = True
            for m in marks:
                d = x - m
                if d in diffs:
                    ok = False
                    break
            if ok:
                marks.append(x)
                for m in marks[:-1]:
                    diffs.add(x - m)
                break
        else:
            # Fallback: place just after max
            x = cur_max + 1
            while True:
                ok = True
                for m in marks:
                    d = x - m
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    break
                x += 1
            marks.append(x)
            for m in marks[:-1]:
                diffs.add(x - m)
    return sorted(marks)


def local_search_golomb(U, deadline, max_iter=5000, seed=None):
    """
    Local search to improve a Golomb-ruler-like set.
    Moves:
    - Replace one mark (keep 0)
    - Add a mark
    - Remove a mark
    - Swap two marks
    Accept if score improves; occasional random restarts of temperature.
    """
    if seed is not None:
        random.seed(seed)
    current = sorted(set(U))
    if 0 not in current:
        current = [0] + [x for x in current if x > 0]
    if len(current) < 2:
        current = [0, 1]
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score

    best = current[:]
    best_score = current_score

    T = 0.05
    cooling = 0.999
    start_time = time.time()
    no_improve = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break

        n = len(current)
        max_val = current[-1]
        move_type = random.choices(
            ['replace', 'add', 'remove', 'swap', 'shift'],
            weights=[30, 15, 10, 15, 30]
        )[0]

        new_set = current[:]

        if move_type == 'replace' and n > 1:
            i = random.randint(1, n - 1)
            # Try to move mark i to a new value
            # Bias toward values that create new differences
            scale = max(1, int(max_val * 0.3))
            new_val = current[i] + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = random.randint(1, max_val)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue

        elif move_type == 'add' and n < 40:
            # Add a new mark that creates many new differences
            # Try values in gaps
            candidates = []
            # Near existing marks
            for x in current:
                for delta in [1, 2, 3, max(1, x // 2), max(1, x // 3)]:
                    candidates.append(x + delta)
                    candidates.append(max(1, x - delta))
            # Random
            for _ in range(10):
                candidates.append(random.randint(1, max_val * 2))
            # Far out
            candidates.append(max_val + random.randint(1, max(2, int(max_val * 0.5))))
            candidates = [c for c in candidates if c > 0 and c not in current]
            if not candidates:
                continue
            new_val = random.choice(candidates)
            new_set.append(new_val)
            new_set = sorted(set(new_set))

        elif move_type == 'remove' and n > 3:
            i = random.randint(1, n - 1)
            new_set.pop(i)

        elif move_type == 'swap' and n > 2:
            i, j = random.sample(range(1, n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue

        elif move_type == 'shift' and n > 1:
            # Shift all marks by a small amount (except 0)
            delta = random.randint(-max(1, int(max_val * 0.1)), max(1, int(max_val * 0.1)))
            if delta == 0:
                continue
            new_set = [0] + [x + delta for x in current[1:]]
            new_set = sorted(set(x for x in new_set if x > 0))
            if len(new_set) != n - 1:
                # Some became <=0 or collided
                continue
            if 0 not in new_set:
                new_set = [0] + new_set

        else:
            continue

        new_score = fast_score(new_set)

        if new_score > current_score:
            current = new_set
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            no_improve += 1
            delta_s = current_score - new_score
            if random.random() < np.exp(-delta_s / T):
                current = new_set
                current_score = new_score

        T *= cooling
        if T < 0.005:
            T = 0.005

        # Occasionally restart with a perturbation if stuck
        if no_improve > 200:
            no_improve = 0
            # Perturb best
            current = best[:]
            n = len(current)
            if n > 1:
                i = random.randint(1, n - 1)
                scale = max(1, int(current[-1] * 0.2))
                current[i] = max(1, current[i] + random.randint(-scale, scale))
                current = sorted(set(current))
                current_score = fast_score(current)
                if current_score < 0:
                    current = best[:]
                    current_score = best_score

    return best, best_score


def polish_search(U, deadline, max_iter=10000, seed=None):
    """
    High-precision polish: fine-grained local search with small perturbations.
    Uses a tighter temperature and more iterations for final optimization.
    """
    if seed is not None:
        random.seed(seed)
    current = sorted(set(U))
    if 0 not in current:
        current = [0] + [x for x in current if x > 0]
    if len(current) < 2:
        return current, fast_score(current)
    
    current_score = fast_score(current)
    best = current[:]
    best_score = current_score
    
    T = 0.01
    cooling = 0.9995
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        max_val = current[-1]
        
        # Fine-grained moves: small perturbations
        move_type = random.choices(
            ['fine_replace', 'fine_add', 'fine_remove', 'fine_shift'],
            weights=[40, 20, 10, 30]
        )[0]
        
        new_set = current[:]
        
        if move_type == 'fine_replace' and n > 1:
            i = random.randint(1, n - 1)
            # Small perturbation: ±1 to ±3
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = current[i] + delta
            if new_val <= 0:
                new_val = random.randint(1, max(1, current[i] - 1))
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
            
        elif move_type == 'fine_add' and n < 35:
            # Add near an existing mark
            i = random.randint(0, n - 1)
            delta = random.choice([-2, -1, 1, 2, 3, 4])
            new_val = current[i] + delta
            if new_val > 0 and new_val not in current:
                new_set.append(new_val)
                new_set = sorted(new_set)
            else:
                continue
                
        elif move_type == 'fine_remove' and n > 4:
            i = random.randint(1, n - 1)
            new_set.pop(i)
            
        elif move_type == 'fine_shift' and n > 1:
            # Shift a subset of marks
            subset_size = min(n - 1, random.randint(1, 3))
            indices = random.sample(range(1, n), subset_size)
            delta = random.choice([-2, -1, 1, 2])
            new_set = current[:]
            for idx in indices:
                new_set[idx] += delta
            new_set = sorted(set(x for x in new_set if x > 0))
            if 0 not in new_set:
                new_set = [0] + new_set
            if len(new_set) != n:
                continue
        else:
            continue
        
        new_score = fast_score(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            no_improve += 1
            delta_s = current_score - new_score
            if random.random() < np.exp(-delta_s / T):
                current = new_set
                current_score = new_score
        
        T *= cooling
        if T < 0.002:
            T = 0.002
        
        if no_improve > 500:
            no_improve = 0
            current = best[:]
            current_score = best_score

    return best, best_score


def genetic_algorithm(U, deadline, pop_size=30, max_gen=200, seed=None):
    """
    Genetic algorithm treating the set as a genome.
    Genome: sorted list of marks (excluding 0).
    Crossover: take prefix from one, suffix from another, then repair.
    Mutation: add/remove/replace marks.
    """
    if seed is not None:
        random.seed(seed)

    def to_genome(U):
        return [x for x in sorted(set(U)) if x > 0]

    def from_genome(g):
        return [0] + sorted(set(g))

    def repair(g):
        # Ensure sorted, unique, positive
        g = sorted(set(x for x in g if x > 0))
        return g

    # Initialize population
    pop = [to_genome(U)]
    while len(pop) < pop_size:
        if random.random() < 0.5:
            # Greedy Golomb with random n
            n = random.randint(3, 15)
            pop.append(to_genome(greedy_golomb(n)))
        else:
            # Perturb existing
            base = random.choice(pop)
            g = base[:]
            # Mutate
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5 and g:
                    i = random.randint(0, len(g) - 1)
                    scale = max(1, int(max(g) * 0.3)) if g else 10
                    g[i] = max(1, g[i] + random.randint(-scale, scale))
                elif random.random() < 0.5 and len(g) < 20:
                    max_v = max(g) if g else 10
                    g.append(random.randint(1, max_v * 2))
                elif g and len(g) > 2:
                    g.pop(random.randint(0, len(g) - 1))
            g = repair(g)
            pop.append(g)

    best_g = to_genome(U)
    best_score = fast_score(from_genome(best_g))

    start_time = time.time()
    for gen in range(max_gen):
        if time.time() > deadline:
            break

        # Score population
        scored = []
        for g in pop:
            U = from_genome(g)
            s = fast_score(U)
            scored.append((s, g))
        scored.sort(key=lambda x: -x[0])

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_g = scored[0][1]

        # Elitism: keep top 2
        new_pop = [scored[0][1][:], scored[1][1][:]] if len(scored) > 1 else [scored[0][1][:]]

        # Crossover + mutation to fill rest
        while len(new_pop) < pop_size:
            if random.random() < 0.3:
                # Mutation of a random parent
                parent = random.choice(scored[:max(1, len(scored)//2)])[1]
                g = parent[:]
                for _ in range(random.randint(1, 4)):
                    if random.random() < 0.4 and g:
                        i = random.randint(0, len(g) - 1)
                        scale = max(1, int(max(g) * 0.3)) if g else 10
                        g[i] = max(1, g[i] + random.randint(-scale, scale))
                    elif random.random() < 0.4 and len(g) < 20:
                        max_v = max(g) if g else 10
                        g.append(random.randint(1, max_v * 2))
                    elif g and len(g) > 2:
                        g.pop(random.randint(0, len(g) - 1))
                g = repair(g)
                new_pop.append(g)
            else:
                # Crossover
                p1, p2 = random.sample(scored[:max(2, len(scored)//2)], 2)
                g1, g2 = p1[1][:], p2[1][:]
                # Sort both
                g1 = sorted(g1)
                g2 = sorted(g2)
                # Take prefix from g1, suffix from g2
                if g1 and g2:
                    cut = random.randint(0, len(g1))
                    g = g1[:cut] + g2[cut:]
                    g = repair(g)
                elif g1:
                    g = g1[:]
                else:
                    g = g2[:]
                # Mutate
                for _ in range(random.randint(0, 2)):
                    if random.random() < 0.5 and g:
                        i = random.randint(0, len(g) - 1)
                        scale = max(1, int(max(g) * 0.3)) if g else 10
                        g[i] = max(1, g[i] + random.randint(-scale, scale))
                    elif random.random() < 0.5 and len(g) < 20:
                        max_v = max(g) if g else 10
                        g.append(random.randint(1, max_v * 2))
                    elif g and len(g) > 2:
                        g.pop(random.randint(0, len(g) - 1))
                g = repair(g)
                new_pop.append(g)

        pop = new_pop

    return from_genome(best_g), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    random.seed(42)

    best_set = [0, 1]
    best_score = fast_score(best_set)

    # 1. Generate candidate Golomb rulers of various sizes
    candidates = []
    for n in range(3, 25):
        if time.time() > deadline - 30:
            break
        g = greedy_golomb(n)
        candidates.append(g)

    # 2. Some hand-crafted known good Golomb rulers
    known_golomb = [
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72],
    ]
    candidates.extend(known_golomb)

    # 3. Geometric sets
    for ratio in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        U = [0]
        val = 1
        for _ in range(15):
            U.append(val)
            val = int(val * ratio)
            if val > 5000:
                break
        candidates.append(sorted(set(U)))

    # 4. Random sparse sets
    for _ in range(15):
        n_elem = random.randint(4, 18)
        max_val = random.randint(50, 500)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)

    # Evaluate all candidates
    scored = []
    for U in candidates:
        s = fast_score(U)
        if s > 0:
            scored.append((s, U))
    scored.sort(key=lambda x: -x[0])

    if scored:
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_set = scored[0][1]

    # 5. Local search on top candidates
    top_k = min(8, len(scored))
    for i in range(top_k):
        if time.time() > deadline - 15:
            break
        s, U = scored[i]
        # Use shorter deadline for early candidates, longer for top ones
        search_deadline = deadline - 5 - i * 0.5
        refined, refined_score = local_search_golomb(U, search_deadline, max_iter=3000, seed=42 + i)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined

    # 6. Genetic algorithm on the best so far
    if time.time() < deadline - 10:
        ga_set, ga_score = genetic_algorithm(best_set, deadline - 5, pop_size=25, max_gen=150, seed=123)
        if ga_score > best_score:
            best_score = ga_score
            best_set = ga_set

    # 7. Final polish: high-precision local search on the global best
    if time.time() < deadline - 3:
        refined, refined_score = polish_search(best_set, deadline - 1, max_iter=10000, seed=999)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
        else:
            # Try one more round with different seed
            if time.time() < deadline - 1:
                refined2, refined_score2 = polish_search(best_set, deadline - 1, max_iter=5000, seed=1234)
                if refined_score2 > best_score:
                    best_score = refined_score2
                    best_set = refined2

    return best_set


if __name__ == "__main__":
    import sys
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Score: {fast_score(result):.6f}")
    print(f"Set size: {len(result)}, max: {max(result)}")
    print(f"Set: {result[:20]}{'...' if len(result)>20 else ''}")
