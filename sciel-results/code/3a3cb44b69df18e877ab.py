import math
import time
import random

def score_set(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a+b)
            diffs.add(a-b)
    size_sum = len(sums)
    size_diff = len(diffs)
    if size_sum == 0:
        return 0
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def lfsr_generate(taps, period, seed_val=1):
    """Generate a binary sequence using LFSR with given taps.
    taps: list of tap positions (0-indexed from LSB)
    period: number of bits to generate
    Returns a list of 0s and 1s of length period."""
    # State is a list of bits, index 0 = LSB
    state = [0] * len(taps)
    for i, b in enumerate(bin(seed_val & ((1 << len(taps)) - 1))[2:].zfill(len(taps))):
        state[i] = int(b)
    
    seq = []
    for _ in range(period):
        seq.append(state[0])
        # Compute next bit: XOR of tapped positions
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        # Shift right, insert new_bit at MSB
        state = [new_bit] + state[:-1]
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Phase 1: Try LFSR-based constructions with various parameters
    # The idea: use LFSR to generate a pseudo-random binary sequence,
    # take positions of 1s as U. This gives near-uniform distribution
    # which maximizes |U-U| while the sparsity helps limit |U+U|.
    
    # Try different LFSR configurations
    lfsr_configs = [
        # (taps, period, threshold_for_inclusion)
        # taps should form a primitive polynomial for max period
        ([0, 3], 32),      # x^4 + x + 1
        ([0, 2], 16),      # x^3 + x^2 + 1
        ([0, 4], 32),      # x^5 + x^2 + 1
        ([0, 1, 4], 32),   # x^5 + x^4 + x + 1
        ([0, 2, 5], 32),   # x^6 + x^3 + x^2 + 1
        ([0, 3, 5], 64),   # x^6 + x^4 + x^3 + 1
        ([0, 1, 3, 5], 64),
        ([0, 2, 4, 6], 64),
        ([0, 3, 6, 8], 128),
        ([0, 2, 5, 8], 128),
        ([0, 1, 4, 8], 128),
        ([0, 3, 4, 8], 128),
        ([0, 2, 3, 8], 128),
        ([0, 5, 8, 12], 256),
        ([0, 1, 7, 12], 256),
        ([0, 3, 9, 12], 256),
        ([0, 4, 10, 12], 256),
    ]
    
    for taps, period in lfsr_configs:
        if time.time() > deadline - 5:
            break
        try:
            seq = lfsr_generate(taps, period)
            # U = positions where seq has 1
            U = [i for i, bit in enumerate(seq) if bit == 1]
            if not U or 0 not in U:
                # Ensure 0 is in U
                U = [0] + U
            U = sorted(set(U))
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
        except Exception:
            continue
    
    # Phase 2: Simulated annealing to optimize U directly
    if time.time() < deadline - 10:
        random.seed(123)
        current_U = best_set[:]
        current_score = best_score
        
        # SA parameters
        T_start = 0.1
        T_end = 0.001
        cooling = 0.995
        T = T_start
        
        max_val = max(current_U) + 20
        min_val = 1
        
        while T > T_end and time.time() < deadline - 5:
            # Generate a neighbor
            move_type = random.random()
            new_U = current_U[:]
            
            if move_type < 0.4 and len(new_U) > 2:
                # Remove a random element (not 0)
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
            elif move_type < 0.8:
                # Replace a random element (not 0)
                idx = random.randint(1, len(new_U) - 1)
                new_val = random.randint(min_val, max_val)
                if new_val not in new_U:
                    new_U[idx] = new_val
                    new_U.sort()
            else:
                # Add a new element
                new_val = random.randint(min_val, max_val)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U.sort()
            
            if len(new_U) < 2 or 0 not in new_U:
                continue
            
            new_score = score_set(new_U)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_U[:]
            
            T *= cooling
    
    # Phase 3: Local refinement
    if time.time() < deadline - 3:
        U = best_set[:]
        U_score = best_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(1, len(U)):
                for delta in [-2, -1, 1, 2]:
                    new_val = U[i] + delta
                    if new_val < 1 or new_val in U:
                        continue
                    trial = U[:]
                    trial[i] = new_val
                    trial.sort()
                    s = score_set(trial)
                    if s > U_score + 1e-12:
                        U = trial
                        U_score = s
                        improved = True
                        break
                if improved:
                    break
        if U_score > best_score:
            best_score = U_score
            best_set = U[:]
    
    return best_set
