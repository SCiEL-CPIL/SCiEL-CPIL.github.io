"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


_primes_list = _primes(N)
_num_primes = len(_primes_list)


def _precompute_indices(primes):
    indices = {}
    for p in primes:
        idx = [np.arange(r, N + 1, p) for r in range(p)]
        indices[p] = idx
    return indices


_indices = _precompute_indices(_primes_list)


def _eval_config(config):
    """Evaluate a config: returns (valid_mask, score)."""
    valid = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid[_indices[p][config[p]]] = False
    return valid, int(np.sum(valid))


def _greedy_init(seed=None):
    """Greedy initialization with optional random tie-breaking."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    active = np.ones(N + 1, dtype=bool)
    config = {}
    for p in _primes_list:
        residue_counts = np.zeros(p, dtype=np.int64)
        for r in range(p):
            idx = _indices[p][r]
            residue_counts[r] = np.sum(active[idx])
        min_count = np.min(residue_counts)
        candidates = np.where(residue_counts == min_count)[0]
        if len(candidates) > 1 and rng.rand() < 0.5:
            chosen_r = int(candidates[rng.randint(len(candidates))])
        else:
            chosen_r = int(candidates[0])
        config[p] = chosen_r
        active[_indices[p][chosen_r]] = False
    return config, int(np.sum(active))


def _tabu_search(config, start_score, deadline, tabu_tenure=30, max_iter=5000):
    """
    Tabu search: try changing one prime's residue at a time.
    Maintains a tabu list of recently changed (prime, residue) pairs.
    Aspiration criterion: accept if new score > best_score.
    """
    rng = np.random.RandomState(int(time.time()) % 100000)

    current = config.copy()
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))

    best_config = current.copy()
    best_score = start_score

    # Tabu list: set of (p_idx, r) tuples
    tabu = set()
    tabu_history = []

    # Prime ordering by size (smaller primes first for more impact)
    primes_sorted = sorted(_primes_list)
    primes_arr = np.array(primes_sorted)

    no_improvement = 0
    iter_count = 0

    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        # Try all primes, pick best non-tabu move (or best overall with aspiration)
        best_move_score = -1
        best_move_p = -1
        best_move_r = -1
        best_aspiration = False

        for pi, p in enumerate(primes_sorted):
            old_r = current[p]

            for new_r in range(p):
                if new_r == old_r:
                    continue

                # Check tabu
                if (pi, new_r) in tabu:
                    # Aspiration: accept if it improves global best
                    old_idx = _indices[p][old_r]
                    new_idx = _indices[p][new_r]
                    valid_mask[old_idx] = True
                    valid_mask[new_idx] = False
                    new_score = int(np.sum(valid_mask))
                    valid_mask[new_idx] = True
                    valid_mask[old_idx] = False

                    if new_score > best_score:
                        # Aspiration override
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_move_p = p
                            best_move_r = new_r
                            best_aspiration = True
                    continue

                # Normal move
                old_idx = _indices[p][old_r]
                new_idx = _indices[p][new_r]
                valid_mask[old_idx] = True
                valid_mask[new_idx] = False
                new_score = int(np.sum(valid_mask))
                valid_mask[new_idx] = True
                valid_mask[old_idx] = False

                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_p = p
                    best_move_r = new_r
                    best_aspiration = False

        if best_move_p == -1:
            # No improving move found, try random perturbation
            num_perturb = rng.randint(1, 4)
            for _ in range(num_perturb):
                pi = rng.randint(len(primes_sorted))
                p = primes_sorted[pi]
                old_r = current[p]
                new_r = rng.randint(p)
                if new_r != old_r:
                    old_idx = _indices[p][old_r]
                    new_idx = _indices[p][new_r]
                    valid_mask[old_idx] = True
                    valid_mask[new_idx] = False
                    current[p] = new_r
                    # Add to tabu
                    tabu.add((pi, new_r))
                    tabu_history.append((pi, new_r))
                    if len(tabu_history) > tabu_tenure:
                        tabu.remove(tabu_history.pop(0))
            current_score = int(np.sum(valid_mask))
            no_improvement += 5
        else:
            # Apply best move
            old_r = current[best_move_p]
            old_idx = _indices[best_move_p][old_r]
            new_idx = _indices[best_move_p][best_move_r]
            valid_mask[old_idx] = True
            valid_mask[new_idx] = False
            current[best_move_p] = best_move_r
            current_score = int(np.sum(valid_mask))

            # Add to tabu
            pi = primes_sorted.index(best_move_p)
            tabu.add((pi, best_move_r))
            tabu_history.append((pi, best_move_r))
            if len(tabu_history) > tabu_tenure:
                tabu.remove(tabu_history.pop(0))

            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
                no_improvement = 0
            else:
                no_improvement += 1

        # Simulated annealing step: occasionally accept worse moves
        if no_improvement > 10 and rng.rand() < 0.1:
            pi = rng.randint(len(primes_sorted))
            p = primes_sorted[pi]
            old_r = current[p]
            new_r = rng.randint(p)
            if new_r != old_r:
                old_idx = _indices[p][old_r]
                new_idx = _indices[p][new_r]
                valid_mask[old_idx] = True
                valid_mask[new_idx] = False
                current[p] = new_r
                current_score = int(np.sum(valid_mask))
                tabu.add((pi, new_r))
                tabu_history.append((pi, new_r))
                if len(tabu_history) > tabu_tenure:
                    tabu.remove(tabu_history.pop(0))
                no_improvement = 0

    return best_config, best_score


def _simulated_annealing(config, start_score, deadline, max_iter=10000):
    """
    Simulated annealing: explore the space of residue configurations.
    """
    rng = np.random.RandomState(int(time.time()) % 100000 + 12345)

    current = config.copy()
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))

    best_config = current.copy()
    best_score = start_score

    temperature = 5.0
    cooling = 0.9995
    primes_sorted = sorted(_primes_list)

    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        # Pick a random prime and a new residue
        pi = rng.randint(len(primes_sorted))
        p = primes_sorted[pi]
        old_r = current[p]
        new_r = rng.randint(p)

        if new_r == old_r:
            continue

        old_idx = _indices[p][old_r]
        new_idx = _indices[p][new_r]
        valid_mask[old_idx] = True
        valid_mask[new_idx] = False
        new_score = int(np.sum(valid_mask))

        delta = new_score - current_score

        if delta >= 0 or rng.rand() < math.exp(delta / temperature):
            current[p] = new_r
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
        else:
            # Revert
            valid_mask[old_idx] = False
            valid_mask[new_idx] = True

        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    return best_config, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # 20s margin before 300s limit

    # Phase 1: Greedy initialization with multiple random seeds
    best_config = None
    best_score = -1

    for seed in range(10):
        if time.time() > deadline:
            break
        config, score = _greedy_init(seed)
        if score > best_score:
            best_score = score
            best_config = config

    # Phase 2: Tabu search to refine the best configuration
    if time.time() < deadline:
        tabu_deadline = min(deadline, start_time + 150.0)
        best_config, best_score = _tabu_search(best_config, best_score, tabu_deadline)

    # Phase 3: Simulated annealing for further refinement
    if time.time() < deadline:
        sa_deadline = deadline
        best_config, best_score = _simulated_annealing(best_config, best_score, sa_deadline)

    # Extract the final list of points
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][best_config[p]]] = False

    result = [int(x) for x in np.where(valid_mask)[0]]
    return sorted(result)
# EVOLVE-BLOCK-END
