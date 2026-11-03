"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set using FFT
    # U + U: values range from 0 to 2*max_u
    # U - U: values range from -max_u to max_u
    total_len = 4 * max_u + 1  # enough for both
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2
    
    # Create indicator vector for U
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT
    freq = np.fft.fft(indicator)
    
    # Sumset: convolution of indicator with itself
    sum_conv = np.fft.ifft(freq * freq).real
    sumset_size = np.sum(sum_conv[:2*max_u+1] > 1e-9)
    
    # Difference set: autocorrelation
    autocorr = np.fft.ifft(freq * np.conj(freq)).real
    # autocorr[k] for k=0..max_u counts pairs with difference k
    # autocorr[fft_len-k] for k=1..max_u counts pairs with difference -k
    diffset_size = 1  # difference 0 always exists
    for k in range(1, max_u + 1):
        if autocorr[k] > 1e-9:
            diffset_size += 1
        if autocorr[fft_len - k] > 1e-9:
            diffset_size += 1
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # For small sets, use direct computation
    if n <= 30:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _local_search(U, deadline, max_iter=5000, temp=1.0):
    """Local search with simulated annealing-like acceptance and random swaps."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    max_u = max(U)
    upper_bound = max_u + 100
    
    for i in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        # Choose a random move
        move_type = random.choice(['add', 'remove', 'swap', 'randomize'])
        
        new_U = U[:]
        
        if move_type == 'add':
            # Add a random element near the current range
            low = 0
            high = max_u + 20
            new_elem = random.randint(low, high)
            if new_elem not in U:
                new_U.append(new_elem)
                new_U.sort()
            else:
                continue # Skip if no change
                
        elif move_type == 'remove':
            if len(U) <= 2:
                continue
            # Remove a random non-zero element
            idx = random.randint(1, len(U) - 1)
            new_U.pop(idx)
            
        elif move_type == 'swap':
            if len(U) <= 1:
                continue
            idx = random.randint(1, len(U) - 1)
            old_elem = U[idx]
            # Replace with a random element
            new_elem = random.randint(1, upper_bound)
            if new_elem == old_elem:
                continue
            if new_elem in U:
                continue
            new_U[idx] = new_elem
            new_U.sort()
            
        elif move_type == 'randomize':
            # Occasionally randomize one element significantly to jump out of local minima
            if len(U) <= 1:
                continue
            idx = random.randint(1, len(U) - 1)
            new_elem = random.randint(1, max_u * 2 + 50)
            if new_elem in U:
                continue
            new_U[idx] = new_elem
            new_U.sort()

        new_score = _score_set(new_U)
        
        if new_score > current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
            max_u = max(U)
            upper_bound = max_u + 100
        else:
            # Accept worse moves with some probability (simulated annealing)
            # Probability decreases over time
            if random.random() < 0.05: # Fixed small probability for exploration
                U = new_U
                current_score = new_score
                max_u = max(U)
                upper_bound = max_u + 100
                
    return best_U, best_score

def _polish(U, deadline):
    """Final high-precision polish pass with tighter tolerance and more iterations."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    max_u = max(U)
    upper_bound = max_u + 50
    
    # More aggressive local search with smaller steps
    for i in range(10000):
        if time.time() > deadline - 1:
            break
        
        improved = False
        # Try swapping one element with small perturbations
        for idx in range(1, len(U)):
            old_val = U[idx]
            for offset in range(-5, 6):
                new_val = old_val + offset
                if new_val <= 0:
                    continue
                if new_val in U:
                    continue
                new_U = U[:]
                new_U[idx] = new_val
                new_U.sort()
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 100:
                for _ in range(10):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        current_score = s
                        improved = True
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3:
                    for idx in range(1, len(U)):
                        new_U = U[:idx] + U[idx+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            current_score = s
                            improved = True
                            break
        
        if not improved:
            # Small random perturbation
            if random.random() < 0.1:
                idx = random.randint(1, len(U) - 1)
                new_val = random.randint(1, upper_bound)
                if new_val not in U:
                    new_U = U[:]
                    new_U[idx] = new_val
                    new_U.sort()
                    s = _score_set(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        current_score = s
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # 1. Systematic search for small sizes
    # Size 3
    for a in range(1, 30):
        for b in range(a + 1, 50):
            if time.time() > deadline - 10:
                return best_set
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # Size 4
    for a in range(1, 20):
        for b in range(a + 1, 35):
            for c in range(b + 1, 60):
                if time.time() > deadline - 10:
                    return best_set
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
                    
    # Size 5 (limited search space)
    for a in range(1, 15):
        for b in range(a + 1, 25):
            for c in range(b + 1, 40):
                for d in range(c + 1, 60):
                    if time.time() > deadline - 10:
                        return best_set
                    U = [0, a, b, c, d]
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]

    # 2. Local search from best found
    if time.time() < deadline - 20:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=2000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score

    # 3. Local search from several promising starting points
    # Generate some structured starting points
    starting_points = []
    
    # Golomb rulers / Sidon-like sets
    for n in range(4, 10):
        for _ in range(5):
            U = [0]
            max_val = 0
            for _ in range(n - 1):
                candidate = random.randint(max_val + 1, max_val + 20)
                U.append(candidate)
                max_val = candidate
            U.sort()
            starting_points.append(U)
            
    # Add the best set found so far
    starting_points.append(best_set[:])
    
    # Add some hardcoded good candidates if known
    starting_points.extend([
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 3, 4, 8],
        [0, 1, 2, 4, 8],
    ])
    
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        s = _score_set(start)
        if s > best_score - 0.1: # Only search from promising starts
            improved_set, improved_score = _local_search(start, deadline, max_iter=1500)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
                
    # 4. Final polish pass
    if time.time() < deadline - 3:
        polished_set, polished_score = _polish(best_set, deadline)
        if polished_score > best_score:
            best_set = polished_set
            best_score = polished_score
    
    if time.time() > deadline - 1:
        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
