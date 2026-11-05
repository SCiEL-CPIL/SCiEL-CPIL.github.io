import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Generate divisors
    divisors = []
    for i in range(3, base + 1, 2):
        if base % i == 0:
            divisors.append(i)
    
    best_classes = None
    best_score = base # Max possible uncovered is base
    
    # We will run multiple restarts
    while time.time() < deadline - 5:
        
        # Randomize divisor order for diversity
        current_divisors = divisors.copy()
        random.shuffle(current_divisors)
        
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        
        # 2. Greedy Initialization
        for m in current_divisors:
            if time.time() > deadline - 5:
                break
            best_r = 0
            best_gain = -1
            total_len = base // m
            
            for r in range(m):
                cov_count = np.count_nonzero(covered[r::m])
                gain = total_len - cov_count
                
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
                elif gain == best_gain:
                    # Random tie break
                    if random.random() < 0.5:
                        best_r = r
            
            classes.append([m, best_r])
            covered[best_r::m] = 1
            
        if not classes:
            continue
            
        # 3. Local Search with Simulated Annealing
        counts = np.zeros(base, dtype=np.int16)
        for m, r in classes:
            counts[r::m] += 1
        
        current_uncovered = np.count_nonzero(counts == 0)
        
        T = 10.0 # Initial temperature
        min_T = 0.1
        cooling_rate = 0.95
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle classes to avoid bias
            indices = list(range(len(classes)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                
                m, r = classes[i]
                
                # Calculate Loss for current r
                loss = np.count_nonzero(counts[r::m] == 1)
                
                accepted_move = False
                best_r_new = r
                best_delta = 0
                
                for r_new in range(m):
                    if r_new == r:
                        continue
                    gain = np.count_nonzero(counts[r_new::m] == 0)
                    delta = loss - gain # Change in uncovered count
                    
                    if delta < 0:
                        # Always accept improvement
                        accepted_move = True
                        best_r_new = r_new
                        best_delta = delta
                        break
                    else:
                        # Accept worsening with prob exp(-delta / T)
                        if random.random() < np.exp(-delta / T):
                            accepted_move = True
                            best_r_new = r_new
                            best_delta = delta
                            break
                            
                if accepted_move:
                    # Apply move
                    counts[r::m] -= 1
                    counts[best_r_new::m] += 1
                    classes[i] = [m, best_r_new]
                    current_uncovered += best_delta
                    improved = True
                    T *= cooling_rate
                    if T < min_T:
                        T = min_T
                    break # Restart sweep to refresh counts context
                
        # Evaluate score
        if current_uncovered < best_score:
            best_score = current_uncovered
            best_classes = [c[:] for c in classes]
            
    if best_classes is None:
        # Fallback to deterministic greedy if random failed or time out immediately
        best_classes = []
        covered = np.zeros(base, dtype=np.uint8)
        for m in divisors:
            best_r = 0
            best_gain = -1
            total_len = base // m
            for r in range(m):
                cov_count = np.count_nonzero(covered[r::m])
                gain = total_len - cov_count
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            best_classes.append([m, best_r])
            covered[best_r::m] = 1
            
    return best_classes
