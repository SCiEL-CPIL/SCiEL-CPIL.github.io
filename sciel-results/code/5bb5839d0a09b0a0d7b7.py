"""Size-efficient sorting networks on 17 wires.

Combines the vectorised zero-one verification and local-search loop of the
current program with a depth-aware genetic / mutation layer that optimises
comparator placement per stage.  The goal is to reduce the comparator count
below the 136-comparator odd-even transposition seed within the 90 s budget.
"""
import time
import random
import math
import numpy as np

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 90.0
SAFETY_MARGIN = 3.0

# ---------------------------------------------------------------------------
# Fast vectorised verification
# ---------------------------------------------------------------------------

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        masks = np.arange(1 << N, dtype=np.int64)
        _ALL_INPUTS = np.array(
            [((masks >> j) & 1) for j in range(N)], dtype=np.int8
        ).T
    return _ALL_INPUTS


def _is_sorted(network):
    """Check whether *network* sorts every Boolean input (zero-one principle)."""
    if not network:
        return N <= 1

    all_in = _get_all_inputs()
    arr = all_in.copy()

    for i, j in network:
        swap_mask = arr[:, i] > arr[:, j]
        if np.any(swap_mask):
            tmp = arr[swap_mask, i].copy()
            arr[swap_mask, i] = arr[swap_mask, j]
            arr[swap_mask, j] = tmp

    diffs = arr[:, 1:] - arr[:, :-1]
    return bool(np.all(diffs >= 0))


def _depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth)


# ---------------------------------------------------------------------------
# Seed networks
# ---------------------------------------------------------------------------

def _odd_even_transposition():
    """Odd-even transposition sort network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _bitonic_sort_network():
    """Bitonic sort network for N wires, padded to next power of 2."""
    n = 1
    while n < N:
        n *= 2

    net = []
    k = int(math.log2(n))
    for stage in range(k):
        step = 1 << stage
        for group_start in range(0, n, 2 * step):
            group_idx = group_start // (2 * step)
            asc = ((group_idx >> stage) % 2) == 0
            for offset in range(step):
                a = group_start + offset
                b = group_start + offset + step
                if a < N and b < N:
                    if asc:
                        net.append((a, b))
                    else:
                        net.append((b, a))
    return net


# ---------------------------------------------------------------------------
# Depth-aware genetic / mutation operators
# ---------------------------------------------------------------------------

def _to_depth_stages(net):
    """
    Convert a flat list of comparators into a list of stages (lists of
    comparators that can be executed in parallel).  Two comparators are in
    the same stage iff they do not share a wire.
    """
    if not net:
        return []

    stages = []
    used_wires = set()
    current_stage = []
    for comp in net:
        i, j = comp
        if i not in used_wires and j not in used_wires:
            current_stage.append(comp)
            used_wires.add(i)
            used_wires.add(j)
        else:
            if current_stage:
                stages.append(current_stage)
            current_stage = [comp]
            used_wires = {i, j}
    if current_stage:
        stages.append(current_stage)
    return stages


def _from_depth_stages(stages):
    """Flatten a list of stages back into a list of comparators."""
    net = []
    for stage in stages:
        net.extend(stage)
    return net


def _mutate_depth_aware(net, rng, mutation_rate=0.3):
    """
    Depth-aware mutation:
      1. Convert to depth stages.
      2. With probability mutation_rate, for each stage, either:
         a. Remove a random comparator from the stage.
         b. Add a random comparator that does not conflict with the stage.
         c. Swap two comparators within the stage.
      3. Flatten back.
    """
    stages = _to_depth_stages(net)
    if not stages:
        return net

    for s_idx in range(len(stages)):
        if rng.random() > mutation_rate:
            continue
        stage = stages[s_idx]
        if not stage:
            continue

        op = rng.choice(['remove', 'add', 'swap'])
        if op == 'remove' and len(stage) > 0:
            idx = rng.randrange(len(stage))
            stage.pop(idx)
        elif op == 'add':
            # Find a wire pair that does not conflict with the current stage
            used = set()
            for i, j in stage:
                used.add(i)
                used.add(j)
            free = [w for w in range(N) if w not in used]
            if len(free) >= 2:
                a, b = rng.sample(free, 2)
                if a > b:
                    a, b = b, a
                stage.append((a, b))
        elif op == 'swap' and len(stage) >= 2:
            i, j = rng.sample(range(len(stage)), 2)
            stage[i], stage[j] = stage[j], stage[i]

    return _from_depth_stages(stages)


def _crossover_depth_aware(net1, net2, rng):
    """
    Depth-aware crossover:
      1. Convert both to depth stages.
      2. For each depth level, take the stage from one parent or the other.
      3. Flatten.
    """
    stages1 = _to_depth_stages(net1)
    stages2 = _to_depth_stages(net2)

    max_depth = max(len(stages1), len(stages2))
    child_stages = []
    for d in range(max_depth):
        if d < len(stages1) and d < len(stages2):
            if rng.random() < 0.5:
                child_stages.append(stages1[d][:])
            else:
                child_stages.append(stages2[d][:])
        elif d < len(stages1):
            child_stages.append(stages1[d][:])
        else:
            child_stages.append(stages2[d][:])

    return _from_depth_stages(child_stages)


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _local_search(net, deadline, max_no_improve=100):
    """
    Hill-climbing local search: try to remove, swap, or replace comparators.
    Returns the best network found before *deadline*.
    """
    best = list(net)
    best_len = len(best)
    no_improve = 0
    rng = random.Random(42)

    while time.time() < deadline:
        improved = False

        # --- Move 1: try removing each comparator ---------------------------
        for idx in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx + 1:]
            if _is_sorted(candidate):
                best = candidate
                best_len -= 1
                improved = True
                no_improve = 0
                break

        if improved:
            continue

        # --- Move 2: try swapping two random comparators --------------------
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(best) < 2:
                break
            i, j = rng.sample(range(len(best)), 2)
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _is_sorted(candidate):
                best = candidate
                improved = True
                no_improve = 0
                break

        if improved:
            continue

        # --- Move 3: try replacing one comparator with a random one ---------
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(best) == 0:
                break
            idx = rng.randrange(len(best))
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            candidate = best[:]
            candidate[idx] = (a, b)
            if _is_sorted(candidate):
                best = candidate
                improved = True
                no_improve = 0
                break

        if improved:
            continue

        # --- Move 4: depth-aware mutation -----------------------------------
        if time.time() < deadline:
            mutated = _mutate_depth_aware(best, rng, mutation_rate=0.4)
            if _is_sorted(mutated) and len(mutated) <= best_len:
                best = mutated
                best_len = len(best)
                improved = True
                no_improve = 0

        if improved:
            continue

        # --- Move 5: add a random comparator, then try to remove any --------
        if len(best) < MAX_COMPARATORS and time.time() < deadline:
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randrange(len(best) + 1)
            candidate = best[:]
            candidate.insert(pos, (a, b))

            for ridx in range(len(candidate)):
                if time.time() > deadline:
                    break
                test = candidate[:ridx] + candidate[ridx + 1:]
                if len(test) < best_len and _is_sorted(test):
                    best = test
                    best_len = len(test)
                    improved = True
                    no_improve = 0
                    break

            if improved:
                continue

        # No improvement in this round
        no_improve += 1
        if no_improve >= max_no_improve:
            break

    return best


def _genetic_search(deadline, initial_net, pop_size=8, generations=None):
    """
    Genetic search with depth-aware crossover and mutation.
    """
    rng = random.Random(99)
    population = [list(initial_net)]
    # Add some perturbed individuals
    for _ in range(pop_size - 1):
        perturbed = list(initial_net)
        for _ in range(rng.randint(2, 5)):
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randrange(len(perturbed) + 1)
            perturbed.insert(pos, (a, b))
        population.append(perturbed)

    best = min(population, key=len)
    best_len = len(best)

    gen = 0
    while time.time() < deadline:
        gen += 1
        # Sort population by length
        population.sort(key=len)

        # Select parents (tournament)
        new_population = [list(population[0])]  # keep best
        for _ in range(pop_size - 1):
            if time.time() > deadline:
                break
            # Tournament selection
            p1_idx = rng.randrange(min(3, len(population)))
            p2_idx = rng.randrange(min(3, len(population)))
            if p1_idx == p2_idx:
                p2_idx = (p2_idx + 1) % len(population)

            p1 = population[p1_idx]
            p2 = population[p2_idx]

            # Crossover
            child = _crossover_depth_aware(p1, p2, rng)
            # Mutation
            if rng.random() < 0.5:
                child = _mutate_depth_aware(child, rng, mutation_rate=0.3)

            # Verify and possibly local search
            if _is_sorted(child):
                if len(child) < best_len:
                    best = child
                    best_len = len(child)
                new_population.append(child)
            else:
                # Try to repair with local search (limited)
                repaired = _local_search(child, min(time.time() + 0.5, deadline), max_no_improve=10)
                if _is_sorted(repaired):
                    if len(repaired) < best_len:
                        best = repaired
                        best_len = len(repaired)
                    new_population.append(repaired)

        population = new_population

        # Elitism: keep best
        population.sort(key=len)
        if population[0] is not best:
            population[0] = list(best)

    return best


def _multi_restart(deadline):
    """
    Multi-restart local search + genetic search.
    """
    best = _odd_even_transposition()
    best_len = len(best)

    if not _is_sorted(best):
        best = _bitonic_sort_network()
        best_len = len(best)

    # First full local search
    improved = _local_search(best, deadline)
    best = improved
    best_len = len(best)

    # Genetic search phase
    remaining = deadline - time.time()
    if remaining > 5:
        gen_result = _genetic_search(deadline, best, pop_size=6)
        if _is_sorted(gen_result) and len(gen_result) < best_len:
            best = gen_result
            best_len = len(best)

    # Perturbed restarts
    rng = random.Random(123)
    for restart in range(15):
        if time.time() > deadline:
            break
        start = best[:]
        for _ in range(rng.randint(2, 6)):
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randrange(len(start) + 1)
            start.insert(pos, (a, b))
        improved = _local_search(start, deadline)
        if len(improved) < best_len:
            best = improved
            best_len = len(best)

    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def construct():
    """Construct a sorting network for 17 wires.  Returns list of (i, j)."""
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)

    # Quick sanity: odd-even transposition is always valid
    oet = _odd_even_transposition()
    if not _is_sorted(oet):
        return oet

    # Run the main search
    result = _multi_restart(deadline)

    # Final verification
    if not _is_sorted(result):
        return oet

    return result
