import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 148  # Safety margin
    
    N = 1000
    MAX_SUM = 2 * N  # 2000
    
    def run_greedy(seed=42, noise_scale=0.0):
        """
        Fast greedy using numpy for sum count checks.
        Optimized to avoid repeated list-to-array conversions.
        """
        random.seed(seed)
        chosen = []
        chosen_arr = np.array([], dtype=np.int32)
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        in_chosen = np.zeros(N + 1, dtype=bool)
        
        while True:
            if time.time() > deadline:
                break
            
            best_x = None
            best_score = float('inf')
            
            # Iterate through candidates
            for x in range(1, N + 1):
                if in_chosen[x]:
                    continue
                
                # Check 2x
                s2 = 2 * x
                if counts[s2] >= 2:
                    continue
                
                tight = 0
                if counts[s2] == 1:
                    tight += 1
                
                # Check x + y for y in chosen
                if len(chosen_arr) > 0:
                    sums = x + chosen_arr
                    # Vectorized check
                    if np.any(counts[sums] >= 2):
                        continue
                    # Count how many are exactly 1
                    tight += np.sum(counts[sums] == 1)
                
                score = tight + random.random() * noise_scale
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            # Add best_x
            chosen.append(best_x)
            in_chosen[best_x] = True
            chosen_arr = np.append(chosen_arr, best_x)
            
            # Update counts
            s2 = 2 * best_x
            counts[s2] += 1
            if len(chosen) > 1:
                prev_chosen = chosen_arr[:-1]
                new_sums = best_x + prev_chosen
                counts[new_sums] += 1
                
        return chosen

    def local_improve(S):
        """
        Try to add more elements to S if possible.
        """
        if not S:
            return S
        
        S_set = set(S)
        S_list = list(S)
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        # Rebuild counts
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i, len(S_list)):
                b = S_list[j]
                counts[a + b] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to find any x that can be added
            S_arr = np.array(S_list, dtype=np.int32)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                
                valid = True
                if counts[2*x] >= 2:
                    continue
                
                if len(S_arr) > 0:
                    sums = x + S_arr
                    if np.any(counts[sums] >= 2):
                        continue
                
                # If valid, add x
                S_set.add(x)
                S_list.append(x)
                
                # Update counts
                counts[2*x] += 1
                if len(S_arr) > 0:
                    new_sums = x + S_arr
                    counts[new_sums] += 1
                
                improved = True
                break # Restart scan with new state
        
        return S_list

    best_set = []
    best_size = 0
    
    # Phase 1: Structured Greedy
    # Run a few deterministic-ish runs
    for i in range(10):
        if time.time() > deadline:
            break
        seed = i
        noise = 0.0
        s = run_greedy(seed=seed, noise_scale=noise)
        s = local_improve(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Phase 2: Randomized Restarts
    while time.time() < deadline:
        seed = random.randint(0, 1000000)
        noise = random.uniform(0.0, 2.0)
        s = run_greedy(seed=seed, noise_scale=noise)
        s = local_improve(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return best_set
