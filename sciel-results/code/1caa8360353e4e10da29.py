import numpy as np
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 128
    # A is a boolean array of length n+1 (0 to 128)
    
    def evaluate_counts(A):
        # A is bool array
        idxs = np.where(A)[0]
        counts = np.zeros(2 * n + 1, dtype=np.int64)
        if len(idxs) > 0:
            for i in range(len(idxs)):
                for j in range(i, len(idxs)):
                    counts[idxs[i] + idxs[j]] += 1
        return counts

    def get_score(counts):
        target_counts = counts[128:257]
        if np.any(target_counts == 0):
            # Invalid: heavy penalty
            peak = np.max(target_counts)
            invalid = np.sum(target_counts == 0)
            return peak * 1000 + invalid
        else:
            return np.max(target_counts)

    best_A = None
    best_score = float('inf')
    
    def update_best(A, score):
        nonlocal best_A, best_score
        if score < best_score:
            best_score = score
            best_A = A.copy()
    
    # Strategy: Simulated Annealing with multiple restarts
    # We will try to find a set A that covers [128, 256] with minimal peak multiplicity.
    # Theoretical lower bound for peak: 
    # Total pairs in A is k(k+1)/2.
    # These pairs map to sums in [0, 256].
    # We care about sums in [128, 256] (129 values).
    # Average multiplicity in target range is roughly (pairs in target) / 129.
    # If k=20, pairs=210. If most fall in target, avg ~1.6. So peak at least 2.
    # If k=16, pairs=136. Avg ~1.05. Peak at least 2?
    # Let's aim for k around 16-24.
    
    # Random restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Pick a size
        k = random.randint(16, 24)
        A = np.zeros(n + 1, dtype=bool)
        candidates = list(range(n + 1))
        random.shuffle(candidates)
        for i in range(k):
            A[candidates[i]] = True
            
        counts = evaluate_counts(A)
        current_score = get_score(counts)
        update_best(A, current_score)
        
        # Simulated Annealing
        temperature = 5.0
        cooling = 0.9995
        local_deadline = min(deadline - 5.0, time.time() + 2.0) # 2 seconds per restart
        
        # Precompute pairs for faster update? 
        # Update: toggling x changes counts for sums x+y where y in A (and x+x).
        # This is O(k) per step.
        
        while time.time() < local_deadline:
            # Pick random element to toggle
            idx = random.randint(0, n)
            
            if A[idx]:
                # Remove idx
                y_idxs = np.where(A)[0]
                # Subtract contributions: pairs {idx, y} for y in A
                # Note: when removing idx, the set of y's includes idx itself.
                # So we subtract 1 from counts[idx + y] for all y in current A.
                targets = idx + y_idxs
                counts[targets] -= 1
                A[idx] = False
            else:
                # Add idx
                y_idxs = np.where(A)[0]
                # Add contributions: pairs {idx, y} for y in current A
                # Plus {idx, idx}
                targets = idx + y_idxs
                counts[targets] += 1
                counts[2 * idx] += 1
                A[idx] = True
            
            new_score = get_score(counts)
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                update_best(A, current_score)
            else:
                # Revert
                if A[idx]:
                    # It was added, so remove it
                    y_idxs = np.where(A)[0]
                    targets = idx + y_idxs
                    counts[targets] -= 1
                    A[idx] = False
                else:
                    # It was removed, so add it back
                    y_idxs = np.where(A)[0]
                    targets = idx + y_idxs
                    counts[targets] += 1
                    counts[2 * idx] += 1
                    A[idx] = True
            
            temperature *= cooling
            if temperature < 0.01:
                break
                
    # Ensure we return a valid set
    if best_A is None:
        # Fallback: return all even numbers? Or range(n+1)?
        # range(n+1) is valid but peak is high.
        # Let's try to construct a valid set quickly if we have time.
        # Simple valid set: A = {0, 1, ..., 128} is valid.
        # Better: A = even numbers in [0, 128]?
        # Sums of two evens are even. So covers even numbers in [128, 256].
        # But we need ALL integers in [128, 256].
        # So we need both parity coverage.
        # A = [0, 1, 2, ..., 128] works.
        return list(range(n + 1))
        
    # Check validity of best_A
    counts = evaluate_counts(best_A)
    if np.any(counts[128:257] == 0):
        # If best found is invalid, try to fix or return a known valid set.
        # Let's try a quick greedy repair on the best invalid set?
        # Or just return a simple valid set.
        # To be safe, return a simple valid set if the best is invalid.
        # However, the score for invalid is high, so if we found any valid set, best_score would be low.
        # If best_score is high (invalid penalty), it means we didn't find a valid set.
        # In that case, returning range(n+1) is better than an invalid set.
        return list(range(n + 1))

    return [i for i in range(n + 1) if best_A[i]]
