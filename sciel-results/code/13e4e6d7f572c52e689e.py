import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s buffer
    
    def score(U):
        if not U:
            return 0
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        if n <= 20:
            # Direct computation for small sets
            sums = set()
            diffs = set()
            for a in U:
                for b in U:
                    sums.add(a + b)
                    diffs.add(a - b)
            s_size = len(sums)
            d_size = len(diffs)
        else:
            # FFT-based for larger sets
            arr = np.zeros(2 * max_u + 1, dtype=np.float64)
            for u in U:
                arr[u] = 1.0
            # Sumset via convolution
            fft_size = 1
            while fft_size < 2 * max_u + 1:
                fft_size *= 2
            fft_arr = np.fft.rfft(arr, n=fft_size)
            conv = np.fft.irfft(fft_arr * fft_arr, n=fft_size)
            s_size = int(np.sum(conv[:2 * max_u + 1] > 0.5))
            # Diffset via autocorrelation-like
            # a - b ranges from -max_u to max_u
            # |a-b| = |b-a|, so we can compute autocorrelation
            auto = np.fft.irfft(fft_arr * np.conj(fft_arr), n=fft_size)
            # auto[k] = sum_x arr[x]*arr[x-k] for k>=0
            # We need count of k where there exists a,b with a-b=k
            # This is the same as sumset of U and -U shifted
            neg_arr = np.zeros(2 * max_u + 1, dtype=np.float64)
            for u in U:
                neg_arr[2 * max_u - u] = 1.0
            fft_neg = np.fft.rfft(neg_arr, n=fft_size)
            conv_diff = np.fft.irfft(fft_arr * fft_neg, n=fft_size)
            # Shift: a + (2*max_u - b) = 2*max_u + (a-b)
            # So a-b = k iff 2*max_u + k is in the sumset of U and neg_arr
            d_size = int(np.sum(conv_diff > 0.5))
        
        if s_size == 0:
            return 0
            
        ratio = d_size / s_size
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        val = 1 + math.log(ratio) / denom
        return val

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Initial candidates - include prime-step sequences
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 16, 64],
        [0, 1, 3, 9, 27],
        [0, 2, 5, 11, 23],  # prime-ish gaps
        [0, 1, 4, 10, 21],  # non-superincreasing
        [0, 3, 7, 15, 31],  # 2^k - 1
    ]
    
    best_score = -1
    best_set = [0, 1, 3]
    
    for c in candidates:
        s = score(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Local Search with simulated annealing-style acceptance
    current_set = list(best_set)
    current_score = best_score
    max_val = max(current_set)
    
    # Temperature for simulated annealing
    temperature = 0.1
    min_temperature = 0.001
    cooling = 0.995
    
    stagnant_count = 0
    
    while time.time() < deadline:
        improved = False
        max_val = max(current_set)
        
        # 1. Try adding a new element - focus on prime gaps and structured positions
        if len(current_set) < 25:
            # Generate candidate positions: primes near expected gaps
            candidates_to_try = set()
            # Positions based on current max
            for x in range(0, int(max_val * 1.5) + 10):
                if x not in current_set:
                    candidates_to_try.add(x)
            # Add prime positions
            for p in range(2, int(max_val * 1.5) + 10):
                if is_prime(p) and p not in current_set:
                    candidates_to_try.add(p)
            
            # Try a subset of candidates (randomized for diversity)
            if len(candidates_to_try) > 50:
                sample = random.sample(sorted(candidates_to_try), 50)
            else:
                sample = sorted(candidates_to_try)
            
            for x in sample:
                new_set = current_set + [x]
                s = score(new_set)
                if s > current_score:
                    current_score = s
                    current_set = new_set
                    max_val = max(current_set)
                    improved = True
                    stagnant_count = 0
                    break
                elif s > current_score - temperature:
                    # Accept worse move with probability
                    if random.random() < math.exp((s - current_score) / temperature):
                        current_score = s
                        current_set = new_set
                        max_val = max(current_set)
                        improved = True
                        stagnant_count = 0
                        break
        
        # 2. Try removing an element (if size > 2)
        if len(current_set) > 2:
            for i in range(len(current_set)):
                new_set = [x for j, x in enumerate(current_set) if j != i]
                s = score(new_set)
                if s > current_score:
                    current_score = s
                    current_set = new_set
                    max_val = max(current_set) if current_set else 0
                    improved = True
                    stagnant_count = 0
                    break
        
        # 3. Try swapping an element with prime-step adjustments
        if len(current_set) > 2:
            for i in range(len(current_set)):
                old_val = current_set[i]
                # Try prime-step replacements
                prime_steps = [2, 3, 5, 7, 11, 13]
                for step in prime_steps:
                    for direction in [-1, 1]:
                        new_val = old_val + direction * step
                        if new_val < 0 or new_val in current_set:
                            continue
                        new_set = list(current_set)
                        new_set[i] = new_val
                        s = score(new_set)
                        if s > current_score:
                            current_score = s
                            current_set = new_set
                            max_val = max(current_set)
                            improved = True
                            stagnant_count = 0
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            stagnant_count += 1
            temperature *= cooling
            if temperature < min_temperature:
                temperature = min_temperature
            
            # Stagnation: try a random jump
            if len(current_set) < 25:
                rand_x = random.randint(0, int(max_val * 2))
                if rand_x not in current_set:
                    new_set = current_set + [rand_x]
                    s = score(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        max_val = max(current_set)
                        stagnant_count = 0
            
            if stagnant_count > 200:
                # Reset temperature and try harder
                temperature = 0.1
                stagnant_count = 0
                
    return current_set
