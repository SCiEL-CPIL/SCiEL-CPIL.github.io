import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 129
    TARGET_LOW = 128
    TARGET_HIGH = 256
    TARGET_SLICE = slice(TARGET_LOW, TARGET_HIGH + 1)
    LAMBDA = 0.5  # Weight for variance term in composite objective
    
    best_A = None
    best_score = float('inf')
    
    def get_counts_from_vals(A_vals):
        if len(A_vals) == 0:
            return np.zeros(257, dtype=np.int64)
        sums = A_vals[:, None] + A_vals[None, :]
        i, j = np.triu_indices(len(A_vals))
        upper_sums = sums[i, j]
        valid_sums = upper_sums[(upper_sums >= TARGET_LOW) & (upper_sums <= TARGET_HIGH)]
        counts = np.bincount(valid_sums, minlength=257).astype(np.int64)
        return counts
    
    def evaluate_composite(counts):
        window = counts[TARGET_SLICE]
        if np.min(window) == 0:
            return float('inf')
        peak = int(np.max(window))
        std = float(np.std(window))
        return peak + LAMBDA * std
    
    def make_structured_seed(rng, size_hint=50):
        """2-level even/odd construction for guaranteed coverage."""
        x = np.zeros(N, dtype=np.int64)
        evens = np.array([2 * k for k in range(65)])
        odds = np.array([2 * k + 1 for k in range(64)])
        rng.shuffle(evens)
        rng.shuffle(odds)
        
        n_even = max(20, size_hint // 2)
        n_odd = max(20, size_hint - n_even)
        n_even = min(n_even, len(evens))
        n_odd = min(n_odd, len(odds))
        
        x[evens[:n_even]] = 1
        x[odds[:n_odd]] = 1
        
        # Fix coverage
        A_vals = np.where(x == 1)[0]
        counts = get_counts_from_vals(A_vals)
        while np.min(counts[TARGET_SLICE]) == 0:
            uncovered = np.where(counts[TARGET_SLICE] == 0)[0] + TARGET_LOW
            if len(uncovered) == 0:
                break
            best_c = None
            best_gain = -1
            for t in uncovered[:10]:  # Limit search
                lo = max(0, t - 128)
                hi = min(128, t)
                for c in range(lo, hi + 1):
                    if x[c] == 1:
                        continue
                    gain = 0
                    if TARGET_LOW <= c + c <= TARGET_HIGH and counts[c + c] == 0:
                        gain += 1
                    for yy in A_vals:
                        if yy == c:
                            continue
                        s = c + yy
                        if TARGET_LOW <= s <= TARGET_HIGH and counts[s] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
            if best_c is None or best_gain <= 0:
                break
            x[best_c] = 1
            A_vals = np.append(A_vals, best_c)
            counts = get_counts_from_vals(A_vals)
        
        return x, A_vals
    
    # SA Loop with Restarts
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Structured seed with varying sizes
        size = random.randint(38, 58)
        rng = random.Random(restart * 1000 + size)
        A_mask, A_vals = make_structured_seed(rng, size)
        counts = get_counts_from_vals(A_vals)
        current_score = evaluate_composite(counts)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A_mask.copy()
        
        # If invalid, try to fix quickly
        if current_score == float('inf'):
            for _ in range(50):
                if time.time() > deadline:
                    break
                uncovered = np.where(counts[TARGET_SLICE] == 0)[0] + TARGET_LOW
                if len(uncovered) == 0:
                    break
                t = uncovered[0]
                lo = max(0, t - 128)
                hi = min(128, t)
                added = False
                for c in random.sample(range(lo, hi + 1), min(10, hi - lo + 1)):
                    if not A_mask[c]:
                        new_indices = c + A_vals
                        new_indices = np.append(new_indices, 2 * c)
                        counts[new_indices] += 1
                        A_vals = np.append(A_vals, c)
                        A_mask[c] = True
                        added = True
                        break
                if not added:
                    break
            current_score = evaluate_composite(counts)
            if current_score < best_score:
                best_score = current_score
                best_A = A_mask.copy()
        
        if current_score == float('inf'):
            continue
        
        # SA with composite objective
        T = 8.0
        T_min = 0.01
        cooling = 0.9999
        last_improvement = time.time()
        
        while time.time() < deadline:
            if random.random() < 0.55:
                # Add element
                candidates = np.where(~A_mask)[0]
                if len(candidates) == 0:
                    break
                k = candidates[random.randint(0, len(candidates) - 1)]
                
                new_indices = k + A_vals
                new_indices = np.append(new_indices, 2 * k)
                counts[new_indices] += 1
                A_vals = np.append(A_vals, k)
                A_mask[k] = True
                
                new_score = evaluate_composite(counts)
                
                delta = new_score - current_score
                if delta < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / T)
                
                if accept:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A_mask.copy()
                        last_improvement = time.time()
                else:
                    counts[new_indices] -= 1
                    A_vals = A_vals[A_vals != k]
                    A_mask[k] = False
            else:
                # Remove element
                if len(A_vals) == 0:
                    break
                k_idx = random.randint(0, len(A_vals) - 1)
                k = A_vals[k_idx]
                
                old_indices = k + A_vals
                counts[old_indices] -= 1
                A_vals = np.delete(A_vals, k_idx)
                A_mask[k] = False
                
                new_score = evaluate_composite(counts)
                
                delta = new_score - current_score
                if delta < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / T)
                
                if accept:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A_mask.copy()
                        last_improvement = time.time()
                else:
                    counts[old_indices] += 1
                    A_vals = np.append(A_vals, k)
                    A_mask[k] = True
            
            T *= cooling
            if T < T_min:
                break
            if time.time() - last_improvement > 4.0:
                break
    
    if best_A is None:
        return list(range(64, 129))
    
    return np.where(best_A)[0].tolist()
