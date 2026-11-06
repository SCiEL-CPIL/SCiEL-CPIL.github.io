import math
import time
import numpy as np

N = 5000

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

# Pre-compute indices for each prime and residue
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
                    # Quick score estimate
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
            current_score = best_move_score
            
            # Update tabu
            pi_of_p = primes_sorted.index(best_move_p)
            tabu.add((pi_of_p, best_move_r))
            tabu_history.append((pi_of_p, best_move_r))
            if len(tabu_history) > tabu_tenure:
                tabu.remove(tabu_history.pop(0))
            
            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
                no_improvement = 0
            else:
                no_improvement += 1
        
        if no_improvement > 100:
            # Restart tabu list and increase perturbation
            tabu.clear()
            tabu_history.clear()
            no_improvement = 0
    
    return best_config, best_score

def _simulated_annealing(config, start_score, deadline):
    """
    Simulated annealing: accept worse moves with decreasing probability.
    """
    rng = np.random.RandomState(int(time.time()) % 100000)
    
    current = config.copy()
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))
    
    best_config = current.copy()
    best_score = start_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    primes_sorted = sorted(_primes_list)
    
    while time.time() < deadline and T > T_min:
        # Pick random prime
        pi = rng.randint(len(primes_sorted))
        p = primes_sorted[pi]
        old_r = current[p]
        new_r = rng.randint(p)
        
        if new_r == old_r:
            T *= cooling
            continue
        
        old_idx = _indices[p][old_r]
        new_idx = _indices[p][new_r]
        valid_mask[old_idx] = True
        valid_mask[new_idx] = False
        new_score = int(np.sum(valid_mask))
        
        delta = new_score - current_score
        if delta >= 0:
            # Accept
            current[p] = new_r
            current_score = new_score
        else:
            # Accept with probability exp(delta/T)
            if rng.rand() < math.exp(delta / T):
                current[p] = new_r
                current_score = new_score
            else:
                # Reject: restore
                valid_mask[new_idx] = True
                valid_mask[old_idx] = False
        
        if current_score > best_score:
            best_score = current_score
            best_config = current.copy()
        
        T *= cooling
    
    return best_config, best_score

def construct():
    start_time = time.time()
    limit_time = start_time + 290.0  # Safety margin before 300s
    
    np.random.seed(42)
    
    best_config = None
    best_score = 0
    
    # Phase 1: Multiple greedy inits with tabu search
    for attempt in range(3):
        if time.time() > limit_time - 60:
            break
        
        config, score = _greedy_init(seed=42 + attempt)
        
        remaining = limit_time - time.time()
        ts_deadline = time.time() + min(remaining * 0.4, 90.0)
        
        config, score = _tabu_search(config, score, ts_deadline, tabu_tenure=40, max_iter=3000)
        
        if score > best_score:
            best_score = score
            best_config = config
    
    # Phase 2: Simulated annealing from best
    if time.time() < limit_time - 30:
        remaining = limit_time - time.time()
        sa_deadline = time.time() + min(remaining * 0.5, 120.0)
        
        config, score = _simulated_annealing(best_config, best_score, sa_deadline)
        
        if score > best_score:
            best_score = score
            best_config = config
    
    # Phase 3: More tabu search with different tenure
    if time.time() < limit_time - 20:
        remaining = limit_time - time.time()
        ts_deadline = time.time() + min(remaining * 0.7, 80.0)
        
        config, score = _tabu_search(best_config, best_score, ts_deadline, tabu_tenure=60, max_iter=4000)
        
        if score > best_score:
            best_score = score
            best_config = config
    
    # Final: extract result
    valid = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid[_indices[p][best_config[p]]] = False
    
    result = [int(x) for x in np.where(valid)[0]]
    return result
