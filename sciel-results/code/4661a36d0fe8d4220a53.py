"""Find maximum cardinality dissociated reciprocal subset of [1,22].

Key insight: we want the largest subset S of {1,...,22} such that all 2^|S|
subset sums of {1/x : x in S} are distinct. This is equivalent to requiring
that no non-empty subset of the weight vector w_x = LCM/x has sum 0 mod 0
(i.e., the weights are linearly independent over {−1,0,1}).

Strategy:
1. Precompute weights w_x = LCM(1..22) // x for x in 1..22.
2. Use a fast collision-check based on maintaining the set of all subset sums.
3. Exhaustive search with branch-and-bound for small sizes, then randomized
   local search (add/swap) for larger sizes.
4. Since the maximum is likely 12 or 13, focus search there.

We use integer arithmetic throughout for speed.
"""
import time
import random
import itertools

_LCM = 232792560
_WEIGHTS = {x: _LCM // x for x in range(1, 23)}
_UNIVERSE = list(range(1, 23))


def _build_sums(subset):
    """Return the frozenset of all subset sums of weights for the given subset."""
    sums = {0}
    for x in subset:
        v = _WEIGHTS[x]
        sums |= {s + v for s in sums}
    return sums


def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct (i.e., weights are dissociated)."""
    if not subset:
        return True
    sums = {0}
    for x in subset:
        v = _WEIGHTS[x]
        new_sums = {s + v for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
    return True


def _check_add(sums, v):
    """Check if adding weight v to existing sums creates a collision."""
    for s in sums:
        if (s + v) in sums:
            return False
    return True


def _greedy_build(order, deadline):
    """Build a dissociated set greedily following the given order."""
    current = []
    sums = {0}
    for x in order:
        if time.time() > deadline:
            break
        v = _WEIGHTS[x]
        if _check_add(sums, v):
            current.append(x)
            sums |= {s + v for s in sums}
    return current, sums


def _local_search_add(subset, sums, deadline, max_rounds=200):
    """Try to add elements to subset. Returns updated subset, sums."""
    subset = subset[:]
    improved = True
    rounds = 0
    while improved and time.time() < deadline and rounds < max_rounds:
        improved = False
        rounds += 1
        for x in _UNIVERSE:
            if x in subset:
                continue
            v = _WEIGHTS[x]
            if _check_add(sums, v):
                subset.append(x)
                sums |= {s + v for s in sums}
                improved = True
                break
    return subset, sums


def _swap_improve(subset, sums, deadline, max_iterations=5000):
    """Try swap moves: remove one element, add another. Returns best found."""
    best_subset = subset[:]
    best_size = len(subset)
    current_subset = subset[:]
    current_sums = sums
    current_size = len(subset)

    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if not current_subset:
            break

        # Pick a random element to remove
        out_idx = random.randint(0, len(current_subset) - 1)
        out_elem = current_subset[out_idx]

        # Try adding each candidate
        for in_elem in _UNIVERSE:
            if in_elem in current_subset:
                continue

            new_subset = current_subset[:out_idx] + current_subset[out_idx + 1:] + [in_elem]
            if _is_dissociated(new_subset):
                new_size = len(new_subset)
                if new_size >= current_size:
                    # Accept (swap keeps same size, or if we removed and added, same size)
                    # Actually swap keeps same size. Let's also try pure adds.
                    current_subset = new_subset
                    current_sums = _build_sums(current_subset)
                    current_size = new_size
                    break

    if current_size > best_size:
        best_subset = current_subset[:]
        best_size = current_size

    return best_subset, best_size


def _random_restart_greedy(deadline, num_restarts=200):
    """Multiple greedy restarts with different orderings."""
    best = []
    best_size = 0

    for _ in range(num_restarts):
        if time.time() > deadline:
            break

        # Create a random ordering, sometimes biased
        r = random.random()
        if r < 0.3:
            order = sorted(_UNIVERSE, key=lambda x: -x)  # Descending
        elif r < 0.6:
            order = sorted(_UNIVERSE, key=lambda x: x)   # Ascending
        else:
            order = _UNIVERSE[:]
            random.shuffle(order)

        current, sums = _greedy_build(order, deadline)
        current, sums = _local_search_add(current, sums, deadline)

        if len(current) > best_size:
            best = current[:]
            best_size = len(current)

    return best, best_size


def _annealing_search(start_subset, deadline, num_steps=10000):
    """Simulated annealing to escape local optima."""
    current = start_subset[:]
    current_sums = _build_sums(current)
    current_size = len(current)
    best = current[:]
    best_size = current_size

    temperature = 1.0
    cooling = 0.999

    for step in range(num_steps):
        if time.time() > deadline:
            break

        # Random move
        move_type = random.random()

        if move_type < 0.4:
            # Add a random element
            available = [x for x in _UNIVERSE if x not in current]
            if not available:
                continue
            in_elem = random.choice(available)
            new_subset = current + [in_elem]
        elif move_type < 0.6:
            # Remove a random element
            if len(current) <= 1:
                continue
            out_idx = random.randint(0, len(current) - 1)
            new_subset = current[:out_idx] + current[out_idx + 1:]
        else:
            # Swap
            if not current:
                continue
            out_idx = random.randint(0, len(current) - 1)
            out_elem = current[out_idx]
            available = [x for x in _UNIVERSE if x not in current]
            if not available:
                continue
            in_elem = random.choice(available)
            new_subset = current[:out_idx] + current[out_idx + 1:] + [in_elem]

        # Evaluate
        if _is_dissociated(new_subset):
            new_size = len(new_subset)
            delta = new_size - current_size
            # Accept if better, or worse with probability
            if delta >= 0 or random.random() < (1.0 / (temperature + 0.1)):
                current = new_subset
                current_sums = _build_sums(current)
                current_size = new_size
                if current_size > best_size:
                    best = current[:]
                    best_size = current_size

        temperature *= cooling

    return best, best_size


def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    random.seed(42)

    # Start with a known good seed (size 9)
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)

    # Verify seed
    if not _is_dissociated(best_subset):
        # Find a valid seed
        for size in range(9, 0, -1):
            found = False
            for combo in itertools.combinations(range(1, 23), size):
                if _is_dissociated(list(combo)):
                    best_subset = list(combo)
                    best_size = size
                    found = True
                    break
            if found:
                break

    # Strategy 1: Random restarts with greedy
    best_subset, best_size = _random_restart_greedy(deadline, num_restarts=300)

    # Strategy 2: Local search add on best
    best_sums = _build_sums(best_subset)
    best_subset, best_sums = _local_search_add(best_subset, best_sums, deadline)
    best_size = len(best_subset)

    # Strategy 3: Simulated annealing
    if time.time() < deadline:
        anneal_deadline = min(deadline, time.time() + 40.0)
        best_subset, best_size = _annealing_search(best_subset, anneal_deadline, num_steps=15000)

    # Strategy 4: Final local search
    if time.time() < deadline:
        best_sums = _build_sums(best_subset)
        best_subset, best_sums = _local_search_add(best_subset, best_sums, deadline)
        best_size = len(best_subset)

    return best_subset
