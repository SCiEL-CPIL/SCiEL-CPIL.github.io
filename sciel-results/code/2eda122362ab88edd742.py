import numpy as np
import random
import time

def construct():
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        size = 2 * max_u + 1
        n_fft = 1
        while n_fft < size:
            n_fft *= 2
            
        vec = np.zeros(n_fft, dtype=np.float64)
        for x in U:
            vec[x] = 1.0
            
        fft_vec = np.fft.fft(vec)
        
        # Sumset U+U
        sum_poly = np.fft.ifft(fft_vec * fft_vec).real
        sum_counts = np.round(sum_poly).astype(int)
        size_sum = np.sum(sum_counts[:2*max_u+1] > 0)
        
        # Diffset U-U
        corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
        diff_counts = np.round(corr).astype(int)
        
        size_diff = np.sum(diff_counts[:max_u+1] > 0) + np.sum(diff_counts[n_fft - max_u:] > 0)
        
        if size_sum == 0 or size_diff == 0:
            return -np.inf
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return -np.inf
            
        score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
        return score

    start_time = time.time()
    time_limit = 115.0
    
    # Initial set: Sparse set to allow growth
    current_set = [0, 1, 3, 7, 15, 31]
    current_score = calc_score(current_set)
    best_set = current_set.copy()
    best_score = current_score
    
    random.seed(42)
    
    T = 0.5
    cooling = 0.995
    min_T = 0.01
    
    u_set = set(current_set)
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = 0.1
            current_set = best_set.copy()
            for _ in range(3):
                if len(current_set) > 1:
                    idx = random.randint(1, len(current_set)-1)
                    new_val = random.randint(0, 2 * max(current_set) + 20)
                    if new_val not in u_set:
                        current_set[idx] = new_val
            u_set = set(current_set)
            current_score = calc_score(current_set)
            if current_score < best_score:
                 current_set = best_set.copy()
                 u_set = set(current_set)
                 current_score = best_score
            continue

        op = random.choice(['add', 'remove', 'swap'])
        new_set = current_set.copy()
        new_u_set = set(new_set)
        
        if op == 'add' and len(new_set) < 150:
            max_u = max(new_set) if new_set else 0
            new_val = random.randint(0, max_u + random.randint(1, 50))
            if new_val not in new_u_set:
                new_set.append(new_val)
                new_u_set.add(new_val)
            else:
                continue
        elif op == 'remove' and len(new_set) > 2:
            idx = random.randint(1, len(new_set)-1)
            removed = new_set.pop(idx)
            new_u_set.discard(removed)
        elif op == 'swap' and len(new_set) > 1:
            idx = random.randint(1, len(new_set)-1)
            old_val = new_set[idx]
            max_u = max(new_set)
            new_val = random.randint(0, max_u + random.randint(1, 50))
            if new_val not in new_u_set or new_val == old_val:
                if new_val in new_u_set and new_val != old_val:
                    continue
                new_u_set.discard(old_val)
                new_u_set.add(new_val)
                new_set[idx] = new_val
            else:
                continue
        else:
            continue

        if 0 not in new_u_set:
             continue

        new_score = calc_score(new_set)
        
        if new_score > current_score:
            current_set = new_set
            u_set = new_u_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()
        else:
            if new_score == -np.inf:
                pass
            else:
                delta = new_score - current_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_set = new_set
                    u_set = new_u_set
                    current_score = new_score
        
        T *= cooling

    return sorted(best_set)
