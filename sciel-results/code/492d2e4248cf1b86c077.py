import time
import random
from math import log
import numpy as np
from scipy import fft

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    max_u = max(U)
    # Sumset size via FFT
    sumset_size = int(np.sum(np.convolve(np.ones(n), np.ones(n), mode='full') > 0))
    
    # Difference set: |U-U| = 2*max_u + 1 - (number of missing differences)
    # Actually compute via indicator function
    # U-U has elements in range [-max_u, max_u]
    # Use FFT: diff indicator = convolve(indicator, reversed_indicator)
    ind = np.zeros(max_u + 1)
    for u in U:
        ind[u] = 1
    # Difference: for each d in [-max_u, max_u], count pairs (a,b) with a-b=d
    # This is cross-correlation of ind with itself
    # Use FFT for speed
    n_fft = 1
    target = 2 * max_u + 1
    while n_fft < target:
        n_fft *= 2
    f = fft.fft(ind, n=n_fft)
    corr = np.real(fft.ifft(f * np.conj(f)))
    # corr[k] for k=0..max_u gives count of pairs with difference k
    # corr[max_u-k] gives count with difference -k (by symmetry for real)
    # Actually for autocorrelation: corr[d] = sum_x ind[x]*ind[x-d]
    # So corr[0] = n, corr[d] for d>0 = count of pairs with diff d
    # |U-U| = 1 + 2 * sum_{d=1}^{max_u} (corr[d] > 0)
    diff_size = 1 + 2 * int(np.sum(corr[1:max_u+1] > 0))
    
    ratio = diff_size / sumset_size
    if ratio <= 1:
        return -1
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_iter=5000):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    temperature = 0.03
    cooling_rate = 0.995
    
    iteration = 0
    while time.time() < deadline and iteration < max_iter:
        iteration += 1
        n = len(current)
        
        move_type = random.randint(1, 4)
        new_set = current
        
        if move_type == 1:  # Perturb
            i = random.randint(1, n - 1)
            old_val = current[i]
            if random.random() < 0.7:
                delta = random.randint(-5, 5)
            else:
                delta = random.randint(-int(old_val*0.3), int(old_val*0.3))
            
            new_val = old_val + delta
            if new_val <= 0:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
                
        elif move_type == 2:  # Add
            if n >= 20:
                continue
            max_val = max(current)
            if random.random() < 0.5:
                new_val = random.randint(max_val + 1, max_val + 20)
            else:
                i = random.randint(1, n - 1)
                if current[i] > current[i-1] + 2:
                    new_val = random.randint(current[i-1] + 1, current[i] - 1)
                else:
                    new_val = random.randint(1, max_val + 20)
            
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        elif move_type == 3:  # Remove
            if n <= 4:
                continue
            i = random.randint(1, n - 1)
            new_set = [current[j] for j in range(n) if j != i]
            
        else:  # Swap/Replace
            i = random.randint(1, n - 1)
            old_val = current[i]
            new_val = random.randint(1, max(5, int(max(current) * 1.3)))
            if new_val == old_val:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue

        new_score = score_set(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = current_score - new_score
            if delta < 0.5 and random.random() < np.exp(-delta / max(temperature, 1e-9)):
                current = new_set
                current_score = new_score
        
        temperature *= cooling_rate

    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Generate diverse base candidates
    candidates = []
    
    # 1. Exponential growth (2^k - 1)
    for k in range(1, 18):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        candidates.append(U)
        
    # 2. Powers of 2
    for k in range(1, 18):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val
        candidates.append(U)
        
    # 3. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 50000:
        U.append(b)
        a, b = b, a + b
    candidates.append(U)
    
    # 4. Squares
    U = [i*i for i in range(20)]
    candidates.append(U)
    
    # 5. Triangular numbers
    U = [i*(i+1)//2 for i in range(20)]
    candidates.append(U)
    
    # 6. Prime numbers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    U = [0] + [i for i in range(1, 500) if is_prime(i)]
    candidates.append(U)
    
    # 7. Randomized exponential with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        random.seed(seed)
        for _ in range(10):
            U = [0]
            val = random.randint(1, 10)
            for _ in range(random.randint(5, 20)):
                U.append(val)
                val = random.randint(int(val * 1.1), int(val * 4.0))
            candidates.append(U)
    
    # 8. Geometric sequences with various ratios
    for ratio in [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0]:
        U = [0]
        val = 1
        for _ in range(15):
            U.append(val)
            val = int(val * ratio)
            if val > 200000:
                break
        candidates.append(U)
    
    # 9. Sparse random sets
    random.seed(999)
    for _ in range(30):
        n_elem = random.randint(8, 25)
        max_val = random.randint(200, 5000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # 10. Mixed growth patterns
    for seed in [11, 22, 33, 44, 55, 66, 77, 88]:
        random.seed(seed)
        U = [0]
        val = random.randint(1, 5)
        for _ in range(random.randint(10, 20)):
            if random.random() < 0.5:
                val = int(val * random.uniform(1.5, 3.0))
            else:
                val += random.randint(1, max(2, int(val * 0.5)))
            U.append(val)
        candidates.append(U)
    
    # 11. Sets with controlled gaps
    for gap_pattern in [[1, 2, 4, 8, 16], [2, 3, 5, 8, 13], [1, 3, 7, 15, 31], [2, 4, 8, 16, 32], [3, 6, 12, 24, 48]]:
        U = [0]
        val = 0
        for gap in gap_pattern * 4:
            val += gap
            U.append(val)
            if val > 100000:
                break
        candidates.append(U)
        
    # 12. Random walks with varying step sizes
    for seed in [101, 202, 303]:
        random.seed(seed)
        U = [0]
        val = 0
        for _ in range(20):
            step = random.randint(1, 20) * random.choice([1, -1])
            val += step
            if val > 0:
                U.append(val)
        U = sorted(set(U))
        candidates.append(U)
        
    # 13. F_2 vector space inspired
    for trial in range(20):
        random.seed(1000 + trial)
        U = [0]
        max_bits = 15
        available_bits = list(range(1, max_bits))
        random.shuffle(available_bits)
        
        num_elements = random.randint(5, 12)
        chosen_bits = available_bits[:num_elements]
        
        for bit in chosen_bits:
            val = 1 << bit
            for lower_bit in range(bit):
                if random.random() < 0.1:
                    val |= (1 << lower_bit)
            U.append(val)
        
        U = sorted(set(U))
        if len(U) > 1:
            candidates.append(U)
            
    # 14. Hybrid: Mix of powers of 2 and small perturbations
    for trial in range(20):
        random.seed(2000 + trial)
        U = [0]
        val = 1
        for _ in range(random.randint(5, 15)):
            U.append(val)
            next_val = val * 2
            offset = random.randint(-val//10, val//10)
            next_val += offset
            if next_val <= val:
                next_val = val + 1
            val = next_val
        candidates.append(U)

    # Evaluate and refine
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 20:
            break
        s = score_set(U)
        if s > 0:
            scored_candidates.append((s, U))
    scored_candidates.sort(reverse=True)
    
    num_candidates_to_process = min(len(scored_candidates), 25)
    time_per_candidate = (deadline - time.time() - 8) / num_candidates_to_process if num_candidates_to_process > 0 else 0
    
    for i, (score, U) in enumerate(scored_candidates[:num_candidates_to_process]):
        if time.time() > deadline - 8:
            break
        
        candidate_deadline = min(deadline - 8, time.time() + time_per_candidate * 1.5)
        
        refined, refined_score = local_search(U, candidate_deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Final high-precision polish pass on the best set
    if time.time() < deadline - 3:
        refined, refined_score = local_search(best_set, deadline - 1)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set
