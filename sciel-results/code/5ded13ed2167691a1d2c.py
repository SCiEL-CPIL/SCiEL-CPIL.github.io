import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 112.0  # Safety margin vs 150 s kill

def _score_from_set(U):
    """
    Compute the score for a set U (list of nonnegative ints containing 0).
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses FFT for O(M log M) computation.
    """
    if not U:
        return 0.0
    m = max(U)
    if m < 1:
        return 1.0
    
    # FFT size: next power of 2 >= 2*m + 1
    N = 1
    while N < 2 * m + 1:
        N *= 2
    
    # Indicator vector
    f = np.zeros(N, dtype=np.float64)
    for x in U:
        if 0 <= x < N:
            f[x] = 1.0
    
    F = np.fft.fft(f)
    
    # Sum set: |U+U|
    conv_sum = np.fft.ifft(F * F).real
    # Valid range for sums: [0, 2*m]
    size_sum = np.sum(conv_sum[:2 * m + 1] > 0.5)
    
    # Difference set: |U-U|
    # Autocorrelation: ifft(F * conj(F))
    auto = np.fft.ifft(F * np.conj(F)).real
    # auto[k] for k in [0, m] gives count of pairs (a,b) with a-b = k (a>=b)
    # By symmetry, |U-U| = 1 + 2 * (# of k in [1,m] with auto[k] > 0.5)
    pos_diffs = np.sum(auto[1:m + 1] > 0.5)
    size_diff = 1 + 2 * int(pos_diffs)
    
    if size_sum == 0:
        return 0.0
    
    ratio = size_diff / float(size_sum)
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
    
    return 1.0 + math.log(ratio) / denom


def _greedy_golomb(num_elements):
    """Construct a greedy Golomb ruler with num_elements marks starting at 0."""
    if num_elements <= 1:
        return [0]
    if num_elements == 2:
        return [0, 1]
    
    U = [0, 1]
    diffs = {1}
    current_max = 1
    
    for _ in range(num_elements - 2):
        candidate = current_max + 1
        while True:
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                for u in U[:-1]:
                    diffs.add(candidate - u)
                current_max = candidate
                break
            else:
                candidate += 1
    return U


def _random_initial_set(n, max_span):
    """Generate a random set of size n containing 0, within [0, max_span]."""
    if n <= 1:
        return [0]
    others = random.sample(range(1, max_span + 1), n - 1)
    return sorted([0] + others)


def _simulated_annealing(U, time_limit, max_span=None, T0=2.0, cooling=0.9995):
    """
    Simulated annealing local search on set U.
    Moves: add, remove, move, swap, perturb.
    Returns (best_U, best_score).
    """
    current_U = sorted(U)
    current_score = _score_from_set(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    if max_span is None:
        max_span = max(30, int(max(current_U) * 1.5))
    
    start_time = time.time()
    T = T0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < start_time + time_limit:
        if no_improve > max_no_improve:
            # Perturb best and restart
            current_U = best_U[:]
            # Randomly perturb 1-3 elements
            for _ in range(random.randint(1, 3)):
                if len(current_U) > 1:
                    idx = random.randint(1, len(current_U) - 1)
                    current_U[idx] = random.randint(1, max_span)
            current_U = sorted(set(current_U))
            current_score = _score_from_set(current_U)
            T = T0
            no_improve = 0
            continue
        
        # Choose move type
        r = random.random()
        new_U = current_U[:]
        
        if r < 0.25:
            # Add element
            if len(new_U) < 80:
                new_val = random.randint(1, max_span)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U.sort()
                else:
                    continue
            else:
                continue
        elif r < 0.45:
            # Remove element
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
            else:
                continue
        elif r < 0.70:
            # Move element
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                new_val = random.randint(1, max_span)
                if new_val not in new_U:
                    new_U[idx] = new_val
                    new_U.sort()
                else:
                    continue
            else:
                continue
        elif r < 0.85:
            # Swap two elements (not 0)
            if len(new_U) > 2:
                i, j = random.sample(range(1, len(new_U)), 2)
                new_U[i], new_U[j] = new_U[j], new_U[i]
            else:
                continue
        else:
            # Perturb: change 2-4 random elements
            if len(new_U) > 2:
                k = random.randint(2, min(4, len(new_U) - 1))
                indices = random.sample(range(1, len(new_U)), k)
                for idx in indices:
                    new_U[idx] = random.randint(1, max_span)
                new_U = sorted(set(new_U))
            else:
                continue
        
        # Ensure 0 is still in the set
        if 0 not in new_U:
            new_U.append(0)
            new_U.sort()
        
        new_score = _score_from_set(new_U)
        
        # Accept/reject
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
                # Expand max_span if near boundary
                if max(current_U) > max_span * 0.85:
                    max_span = min(int(max_span * 1.3), 600)
        else:
            # Simulated annealing: accept with probability exp((new - cur)/T)
            delta = new_score - current_score
            if delta > -0.1:  # Only consider accepting very small losses
                accept_prob = math.exp(delta / T) if T > 0.01 else 0.0
                if random.random() < accept_prob:
                    current_U = new_U
                    current_score = new_score
            no_improve += 1
        
        # Cool down
        T *= cooling
    
    return best_U, best_score


def construct():
    global _DEADLINE
    
    best_score = 0.0
    best_set = [0, 1]
    
    random.seed(42)
    
    # Phase 1: Diverse initial configurations
    configs = []
    
    # Random sets of various sizes and spans
    for n in [5, 8, 12, 16, 20, 25, 30, 40]:
        for span_mult in [2, 4, 8, 15]:
            M = n * span_mult
            if M > 400:
                continue
            configs.append(('random', n, M))
    
    # Golomb rulers (excellent for large |U-U|/|U+U| ratio)
    for n in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        try:
            g = _greedy_golomb(n)
            configs.append(('golomb', n, max(g) + 5))
        except:
            pass
    
    # Shuffle for diversity
    random.shuffle(configs)
    
    # Phase 2: Search each config with simulated annealing
    num_configs = len(configs)
    # Allocate time: ~1.5s per config, with minimum 0.3s
    for i, (ctype, n, M) in enumerate(configs):
        if time.time() > _DEADLINE - 8.0:
            break
        
        remaining = _DEADLINE - time.time() - 8.0
        # Time per config: distribute remaining time
        time_per = max(0.3, min(2.0, remaining / max(1, num_configs - i)))
        
        if ctype == 'random':
            U_init = _random_initial_set(n, M)
        else:  # golomb
            try:
                U_init = _greedy_golomb(n)
            except:
                U_init = _random_initial_set(n, M)
        
        U_opt, score_opt = _simulated_annealing(U_init, time_per, max_span=max(M, max(U_init) + 10))
        
        if score_opt > best_score:
            best_score = score_opt
            best_set = U_opt[:]
    
    # Phase 3: Final polish on best set with more time
    if time.time() < _DEADLINE - 5.0 and len(best_set) > 1:
        remaining = _DEADLINE - time.time() - 5.0
        # Multiple polish rounds with decreasing temperature
        current_best = best_set[:]
        current_best_score = best_score
        
        for polish_round in range(3):
            if time.time() > _DEADLINE - 5.0:
                break
            round_time = remaining / (3 - polish_round)
            T0 = 1.0 / (polish_round + 1)  # Lower temperature for later rounds
            
            U_polished, score_polished = _simulated_annealing(
                current_best, round_time,
                max_span=max(50, int(max(current_best) * 1.2)),
                T0=T0, cooling=0.9998
            )
            if score_polished > current_best_score:
                current_best_score = score_polished
                current_best = U_polished[:]
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_set = current_best[:]
    
    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
    if len(best_set) < 2:
        best_set = [0, 1]
    
    return best_set
# EVOLVE-BLOCK-END
