import random
import time

def _is_sidon(S):
    # S is a sorted list of distinct integers
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(S):
    # S is a sorted list of 12 distinct integers
    # Exhaustively find the largest subset that is Sidon
    # n=12, so 2^12 = 4096 subsets
    n = len(S)
    best = 0
    for mask in range(1 << n):
        if bin(mask).count('1') <= best:
            continue
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(S[i])
        if len(subset) <= best:
            continue
        if _is_sidon(subset):
            best = len(subset)
    return best

def _random_candidate(rng):
    # Pick 12 distinct integers in [0, 128]
    return sorted(rng.sample(range(129), 12))

def _neighborhood_move(candidate, rng):
    # Perturb: pick 1-3 random positions, replace with new random values in [0,128]
    c = list(candidate)
    n_moves = rng.randint(1, 3)
    used = set(c)
    for _ in range(n_moves):
        idx = rng.randint(0, 11)
        # Pick a new value not already in the set
        new_val = rng.randint(0, 128)
        attempts = 0
        while new_val in used and attempts < 50:
            new_val = rng.randint(0, 128)
            attempts += 1
        if new_val in used:
            continue
        used.discard(c[idx])
        c[idx] = new_val
        used.add(new_val)
    return sorted(c)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    rng = random.Random(42)
    
    # Initial candidate
    best = _random_candidate(rng)
    best_score = _max_sidon_size(best)
    
    # Also try a few structured initial candidates
    structured = [
        list(range(12)),
        [0, 1, 3, 7, 12, 20, 31, 45, 62, 82, 105, 128],
        [0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77],
        [0, 1, 4, 6, 9, 13, 18, 24, 31, 39, 48, 58],
    ]
    for sc in structured:
        s = _max_sidon_size(sc)
        if s < best_score:
            best_score = s
            best = list(sc)
    
    # Simulated annealing / hill-climbing
    current = list(best)
    current_score = best_score
    T = 0.05  # temperature
    T_min = 0.005
    T_decay = 0.9995
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        neighbor = _neighborhood_move(current, rng)
        n_score = _max_sidon_size(neighbor)
        
        if n_score <= current_score:
            # Accept (improvement or tie)
            current = neighbor
            current_score = n_score
            if n_score < best_score:
                best_score = n_score
                best = list(neighbor)
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < 2.718281828 ** (-(n_score - current_score) / T):
                current = neighbor
                current_score = n_score
        
        # Occasionally reheat if stuck
        if iters % 5000 == 0 and current_score > 5:
            T = max(T_min, T * 1.1)
        
        T = max(T_min, T * T_decay)
    
    return list(best)

# EVOLVE-BLOCK-END
