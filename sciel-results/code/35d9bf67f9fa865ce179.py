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
    """Greedy initialization with optional random tie-breaking, vectorized."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    active = np.ones(N + 1, dtype=bool)
    config = {}
    for p in _primes_list:
        # Vectorized: count active elements in each residue class
        # Build a 2D array where row r contains indices for residue r
        idx_list = _indices[p]
        # Use numpy to count active in each residue class efficiently
        # For small p, just loop; for larger p, still loop but with numpy sum
        residue_counts = np.array([np.sum(active[idx]) for idx in idx_list], dtype=np.int64)
        min_count = np.min(residue_counts)
        candidates = np.where(residue_counts == min_count)[0]
        if len(candidates) > 1 and rng.rand() < 0.5:
            chosen_r = int(candidates[rng.randint(len(candidates))])
        else:
            chosen_r = int(candidates[0])
        config[p] = chosen_r
        active[_indices[p][chosen_r]] = False
    return config, int(np.sum(active))


def _tabu_search(config, start_score, deadline, tabu_tenure=40, max_iter=10000):
    """
    Tabu search with improved move evaluation and diversification.
    """
    rng = np.random.RandomState(int(time.time()) % 100000)

    current = config.copy()
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))

    best_config = current.copy()
    best_score = start_score

    tabu = set()
    tabu_history = []

    primes_sorted = sorted(_primes_list)
    n_primes = len(primes_sorted)

    no_improvement = 0
    iter_count = 0

    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        best_move_score = -1
        best_move_p = -1
        best_move_r = -1
        best_aspiration = False

        for pi, p in enumerate(primes_sorted):
            old_r = current[p]
            old_idx = _indices[p][old_r]

            for new_r in range(p):
                if new_r == old_r:
                    continue

                new_idx = _indices[p][new_r]

                is_tabu = (pi, new_r) in tabu

                # Evaluate move
                valid_mask[old_idx] = True
                valid_mask[new_idx] = False
                new_score = int(np.sum(valid_mask))
                valid_mask[new_idx] = True
                valid_mask[old_idx] = False

                if is_tabu:
                    if new_score > best_score and new_score > best_move_score:
                        best_move_score = new_score
                        best_move_p = p
                        best_move_r = new_r
                        best_aspiration = True
                    continue

                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_p = p
                    best_move_r = new_r
                    best_aspiration = False

        if best_move_p == -1:
            # Diversification: random multi-move perturbation
            num_perturb = rng.randint(2, 6)
            for _ in range(num_perturb):
                pi = rng.randint(n_primes)
                p = primes_sorted[pi]
                old_r = current[p]
                new_r = rng.randint(p)
                if new_r != old_r:
                    old_idx = _indices[p][old_r]
                    new_idx = _indices[p][new_r]
                    valid_mask[old_idx] = True
                    valid_mask[new_idx] = False
                    current[p] = new_r
                    tabu.add((pi, new_r))
                    tabu_history.append((pi, new_r))
                    if len(tabu_history) > tabu_tenure:
                        tabu.remove(tabu_history.pop(0))
            current_score = int(np.sum(valid_mask))
            no_improvement += 3
        else:
            old_r = current[best_move_p]
            old_idx = _indices[best_move_p][old_r]
            new_idx = _indices[best_move_p][best_move_r]
            valid_mask[old_idx] = True
            valid_mask[new_idx] = False
            current[best_move_p] = best_move_r
            current_score = int(np.sum(valid_mask))

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

        # Escape strategy: if stuck for a while, do a larger perturbation
        if no_improvement > 20 and rng.rand() < 0.15:
            num_perturb = rng.randint(3, 8)
            for _ in range(num_perturb):
                pi = rng.randint(n_primes)
                p = primes_sorted[pi]
                old_r = current[p]
                new_r = rng.randint(p)
                if new_r != old_r:
                    old_idx = _indices[p][old_r]
                    new_idx = _indices[p][new_r]
                    valid_mask[old_idx] = True
                    valid_mask[new_idx] = False
                    current[p] = new_r
                    tabu.add((pi, new_r))
                    tabu_history.append((pi, new_r))
                    if len(tabu_history) > tabu_tenure:
                        tabu.remove(tabu_history.pop(0))
            current_score = int(np.sum(valid_mask))
            no_improvement = 0

    return best_config, best_score


def _simulated_annealing(config, start_score, deadline, max_iter=20000):
    """
    Simulated annealing with adaptive temperature and restarts.
    """
    rng = np.random.RandomState(int(time.time()) % 100000 + 12345)

    current = config.copy()
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))

    best_config = current.copy()
    best_score = start_score

    temperature = 8.0
    cooling = 0.999
    min_temp = 0.05
    primes_sorted = sorted(_primes_list)
    n_primes = len(primes_sorted)

    iter_count = 0
    no_improve_count = 0

    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        pi = rng.randint(n_primes)
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
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            valid_mask[old_idx] = False
            valid_mask[new_idx] = True
            no_improve_count += 1

        temperature *= cooling
        if temperature < min_temp:
            # Restart with fresh temperature if stuck
            if no_improve_count > 200:
                temperature = 5.0
                no_improve_count = 0
            else:
                temperature = min_temp

    return best_config, best_score


def _local_polish(config, score, deadline):
    """
    Final local polish: exhaustively try all single-prime changes repeatedly
    until no improvement is possible or deadline is reached.
    """
    current = config.copy()
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))

    best_config = current.copy()
    best_score = score

    primes_sorted = sorted(_primes_list)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in primes_sorted:
            old_r = current[p]
            old_idx = _indices[p][old_r]
            best_new_r = old_r
            best_new_score = current_score

            for new_r in range(p):
                if new_r == old_r:
                    continue
                new_idx = _indices[p][new_r]
                valid_mask[old_idx] = True
                valid_mask[new_idx] = False
                new_score = int(np.sum(valid_mask))
                valid_mask[new_idx] = True
                valid_mask[old_idx] = False

                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_r = new_r

            if best_new_r != old_r:
                # Apply the best move for this prime
                old_idx = _indices[p][old_r]
                new_idx = _indices[p][best_new_r]
                valid_mask[old_idx] = True
                valid_mask[new_idx] = False
                current[p] = best_new_r
                current_score = best_new_score
                improved = True

                if current_score > best_score:
                    best_score = current_score
                    best_config = current.copy()

    return best_config, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit

    # Phase 1: Greedy initialization with multiple random seeds
    best_config = None
    best_score = -1

    for seed in range(15):
        if time.time() > deadline:
            break
        config, score = _greedy_init(seed)
        if score > best_score:
            best_score = score
            best_config = config

    # Phase 2: Tabu search to refine the best configuration
    if time.time() < deadline:
        tabu_deadline = min(deadline, start_time + 120.0)
        best_config, best_score = _tabu_search(best_config, best_score, tabu_deadline)

    # Phase 3: Simulated annealing for further refinement
    if time.time() < deadline:
        sa_deadline = min(deadline, start_time + 240.0)
        best_config, best_score = _simulated_annealing(best_config, best_score, sa_deadline)

    # Phase 4: Local polish - exhaustive single-prime optimization
    if time.time() < deadline:
        polish_deadline = deadline
        best_config, best_score = _local_polish(best_config, best_score, polish_deadline)

    # Extract the final list of points
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][best_config[p]]] = False

    result = [int(x) for x in np.where(valid_mask)[0]]
    return sorted(result)
# EVOLVE-BLOCK-END
