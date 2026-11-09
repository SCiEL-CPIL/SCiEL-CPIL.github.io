import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 88.0  # 2s margin
    
    MIN_SUM = 128
    MAX_SUM = 256
    N = 128
    
    def compute_counts(A_bool):
        # A_bool is a numpy array of 0s and 1s of length N+1
        # Convolution gives ordered pair counts
        conv = np.convolve(A_bool, A_bool, mode='full')
        # conv[s] is count of (a,b) in A x A such that a+b=s
        # We need unordered pairs {a,b} with a<=b
        # Let U[s] be the count of unordered pairs.
        # If a != b, (a,b) and (b,a) are distinct in conv, but 1 in U.
        # If a == b, (a,a) is 1 in conv, and 1 in U.
        # Let D[s] be 1 if s is even and A[s/2] is 1, else 0.
        # conv[s] = 2 * (sum_{a<b, a+b=s} 1) + D[s]
        # U[s] = (sum_{a<b, a+b=s} 1) + D[s] = (conv[s] - D[s]) / 2 + D[s] = (conv[s] + D[s]) / 2
        
        # D is 1 at indices s where s is even and A[s//2] is 1
        # s ranges from 0 to 2N
        D = np.zeros(2 * N + 1, dtype=int)
        # Indices s = 0, 2, 4, ... 2N
        # s//2 = 0, 1, 2, ... N
        D[0::2] = A_bool
        
        counts = (conv + D) // 2
        return counts[MIN_SUM:MAX_SUM+1]

    def is_valid(counts):
        return np.all(counts >= 1)

    def get_peak(counts):
        return np.max(counts)

    best_peak = float('inf')
    best_A = None
    
    # Multi-start
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize A with random size
        # Start with a small set, repair to validity
        k = random.randint(18, 25)
        A_set = set(random.sample(range(N + 1), k))
        
        # Repair Phase
        A_bool = np.zeros(N + 1, dtype=int)
        for x in A_set:
            A_bool[x] = 1
            
        counts = compute_counts(A_bool)
        
        # If not valid, add elements
        max_repair_steps = 100
        steps = 0
        while not is_valid(counts) and steps < max_repair_steps:
            # Find an uncovered sum
            zero_indices = np.where(counts == 0)[0]
            if len(zero_indices) == 0:
                break
            s = MIN_SUM + random.choice(zero_indices)
            
            # Find a random pair (a, b) such that a+b=s
            # a can be from max(0, s-N) to min(N, s)
            a_min = max(0, s - N)
            a_max = min(N, s)
            if a_min > a_max:
                # Should not happen for s in [128, 256] with N=128
                # Min sum 0+0=0, Max 128+128=256.
                # So there is always a solution.
                pass
            
            a = random.randint(a_min, a_max)
            b = s - a
            
            # Add a and b to A
            if a not in A_set:
                A_set.add(a)
                A_bool[a] = 1
            if b not in A_set:
                A_set.add(b)
                A_bool[b] = 1
                
            counts = compute_counts(A_bool)
            steps += 1
            
        if not is_valid(counts):
            continue
            
        current_peak = get_peak(counts)
        
        # Optimization Phase: Simulated Annealing
        T = 5.0
        cooling = 0.995
        current_A = A_set.copy()
        current_bool = A_bool.copy()
        current_counts = counts.copy()
        
        # Local Search / SA
        while time.time() < deadline - 1.0:
            if T < 0.01:
                break
                
            # Pick a move
            # 1. Remove
            # 2. Swap
            move_type = random.randint(0, 1)
            
            if move_type == 0:
                # Remove
                if not current_A:
                    continue
                x = random.choice(list(current_A))
                
                # Try removal
                new_bool = current_bool.copy()
                new_bool[x] = 0
                new_counts = compute_counts(new_bool)
                
                if is_valid(new_counts):
                    new_peak = get_peak(new_counts)
                    delta = new_peak - current_peak
                    if delta < 0 or (delta == 0 and len(current_A) > 1 and random.random() < 0.5) or random.random() < math.exp(-delta / T):
                        # Accept
                        current_A.discard(x)
                        current_bool = new_bool
                        current_counts = new_counts
                        current_peak = new_peak
                        
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = current_A.copy()
                # If invalid, reject (do nothing)
                
            else:
                # Swap
                if not current_A:
                    continue
                x = random.choice(list(current_A))
                y = random.randint(0, N)
                if y in current_A:
                    continue
                    
                new_bool = current_bool.copy()
                new_bool[x] = 0
                new_bool[y] = 1
                new_counts = compute_counts(new_bool)
                
                if is_valid(new_counts):
                    new_peak = get_peak(new_counts)
                    delta = new_peak - current_peak
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        current_A.discard(x)
                        current_A.add(y)
                        current_bool = new_bool
                        current_counts = new_counts
                        current_peak = new_peak
                        
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = current_A.copy()
                        
            T *= cooling
            
        # End of restart
        
    if best_A is None:
        # Fallback: Return a valid set (e.g., all evens or odds? No, must cover [128, 256])
        # A = {0, 1, ..., 128} is always valid.
        return list(range(N + 1))
        
    return list(best_A)
