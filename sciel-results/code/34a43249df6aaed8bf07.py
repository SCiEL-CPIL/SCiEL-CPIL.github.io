"""Genetic algorithm for size-efficient sorting networks on 18 wires."""
import time
import numpy as np

N = 18
DEADLINE = 80.0  # safety margin before 90s kill
MAX_COMPARATORS = 4096

# Precompute all 2^18 Boolean inputs as a bit matrix: shape (262144, 18)
# Row k has bits = binary representation of k
def _build_inputs():
    k = np.arange(2**N, dtype=np.uint8)
    bits = np.zeros((2**N, N), dtype=np.uint8)
    for w in range(N):
        bits[:, w] = (k >> w) & 1
    return bits

# Precompute sorted target: for each input, the sorted order is [0]*zeros + [1]*ones
def _build_target(inputs):
    cnt_ones = inputs.sum(axis=1)  # number of 1s
    target = np.zeros_like(inputs)
    for w in range(N):
        # wire w is 1 iff w >= (N - cnt_ones)
        target[:, w] = (w >= (N - cnt_ones)).astype(np.uint8)
    return target

_INPUTS = None
_TARGET = None

def _init_data():
    global _INPUTS, _TARGET
    if _INPUTS is None:
        _INPUTS = _build_inputs()
        _TARGET = _build_target(_INPUTS)

def _apply_comparator(state, i, j):
    """state: (262144, N) uint8. Compare-exchange: min to i, max to j."""
    a = state[:, i]
    b = state[:, j]
    m = np.minimum(a, b)
    M = np.maximum(a, b)
    state[:, i] = m
    state[:, j] = M

def _evaluate_network(comparators):
    """Return number of unsorted Boolean inputs (0 = sorted)."""
    state = _INPUTS.copy()
    for (i, j) in comparators:
        _apply_comparator(state, i, j)
    # count mismatches against target
    mismatches = (state != _TARGET).sum()
    return int(mismatches)

def _is_sorted_network(comparators):
    return _evaluate_network(comparators) == 0

# ---------- Genetic algorithm ----------

def _random_comparator():
    i = np.random.randint(0, N)
    j = np.random.randint(i + 1, N)
    return (int(i), int(j))

def _random_network(length):
    return [_random_comparator() for _ in range(length)]

def _crossover(a, b):
    """Uniform crossover between two networks (as lists of tuples)."""
    la, lb = len(a), len(b)
    L = max(la, lb)
    child = []
    for k in range(L):
        if np.random.rand() < 0.5:
            child.append(a[k % la] if k < la else _random_comparator())
        else:
            child.append(b[k % lb] if k < lb else _random_comparator())
    return child

def _mutate(network, rate=0.1):
    """Mutation: per comparator, small chance to change it."""
    child = list(network)
    for k in range(len(child)):
        r = np.random.rand()
        if r < rate * 0.5:
            # replace with random comparator
            child[k] = _random_comparator()
        elif r < rate:
            # delete
            del child[k]
    # occasional insertion
    if np.random.rand() < rate * 0.5:
        pos = np.random.randint(0, len(child) + 1)
        child.insert(pos, _random_comparator())
    return child

def _fitness(comparators):
    """Lower is better: unsorted count, plus a small penalty for length."""
    unsorted = _evaluate_network(comparators)
    return unsorted + 0.01 * len(comparators)

def _ga(starting_network, time_budget):
    _init_data()
    pop_size = 60
    # initialize population: seed + random networks of varying lengths
    pop = [list(starting_network)]
    base_len = len(starting_network)
    for _ in range(pop_size - 1):
        L = int(np.random.randint(max(5, base_len // 2), base_len + 10))
        pop.append(_random_network(L))

    best = min(pop, key=_fitness)
    best_fit = _fitness(best)

    start = time.time()
    gen = 0
    while time.time() - start < time_budget:
        gen += 1
        # rank population
        ranked = sorted(pop, key=_fitness)
        new_pop = [list(ranked[0])]  # elitism
        # tournament selection + crossover + mutation
        while len(new_pop) < pop_size:
            # tournament
            idx1 = np.random.randint(0, len(ranked))
            idx2 = np.random.randint(0, len(ranked))
            p1 = ranked[idx1] if _fitness(ranked[idx1]) < _fitness(ranked[idx2]) else ranked[idx2]
            idx3 = np.random.randint(0, len(ranked))
            idx4 = np.random.randint(0, len(ranked))
            p2 = ranked[idx3] if _fitness(ranked[idx3]) < _fitness(ranked[idx4]) else ranked[idx4]
            child = _crossover(p1, p2)
            child = _mutate(child)
            if len(child) > MAX_COMPARATORS:
                child = child[:MAX_COMPARATORS]
            new_pop.append(child)
        pop = new_pop
        # track best
        cur_best = min(pop, key=_fitness)
        if _fitness(cur_best) < best_fit:
            best = list(cur_best)
            best_fit = _fitness(cur_best)
            # if we found a fully sorted network, we can stop improving unsorted
            if _evaluate_network(best) == 0:
                # keep going to reduce length? stop to save time
                break
    return best

def construct():
    _init_data()
    t0 = time.time()
    # Start from a decent seed: odd-even transposition (known correct, 153 comps)
    seed = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    # Reduce time budget to leave margin
    remaining = DEADLINE - (time.time() - t0)
    # Reserve time for final verification
    ga_budget = max(5.0, remaining - 15.0)
    result = _ga(seed, ga_budget)

    # Verify the result is a valid sorting network
    if not _is_sorted_network(result):
        # fall back to the known-correct seed
        result = seed

    # Ensure length limit
    if len(result) > MAX_COMPARATORS:
        result = result[:MAX_COMPARATORS]
        if not _is_sorted_network(result):
            result = seed

    # Final safety check on time
    if time.time() - t0 > DEADLINE:
        return seed

    return result
