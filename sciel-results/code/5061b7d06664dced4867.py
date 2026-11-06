import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 128
    # A is a boolean array of length n+1 (0 to 128)
    
    def evaluate(A):
        # Calculate counts for A
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
            # Invalid
            peak = np.max(target_counts)
            invalid = np.sum(target_counts == 0)
            return peak * 1000 + invalid
        else:
            return np.max(target_counts)

    best_A = None
    best_score = float('inf')
    
    # Try multiple random restarts
    # Estimated optimal size is around 16-20.
    # We will try sizes from 12 to 30.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Pick a size
        k = random.randint(14, 25)
        A = np.zeros(n + 1, dtype=bool)
        candidates = list(range(n + 1))
        random.shuffle(candidates)
        for i in range(k):
            A[candidates[i]] = True
            
        counts = evaluate(A)
        current_score = get_score(counts)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A.copy()
            
        if current_score == 0: # Perfect? Unlikely, min peak is at least 1
             pass

        # Local Search / SA
        temperature = 10.0
        cooling = 0.999
        local_deadline = min(deadline, time.time() + 0.5) # 0.5 seconds per restart
        
        while time.time() < local_deadline:
            # Pick random element to toggle
            idx = random.randint(0, n)
            
            if A[idx]:
                # Remove idx
                # Find all y in A
                y_idxs = np.where(A)[0]
                # Subtract contributions
                # Pairs {idx, y} for y in A
                # This includes {idx, idx}
                # So we subtract 1 from counts[idx + y] for all y in A
                targets = idx + y_idxs
                counts[targets] -= 1
                A[idx] = False
            else:
                # Add idx
                y_idxs = np.where(A)[0]
                # Add contributions
                # Pairs {idx, y} for y in A
                # Plus {idx, idx}
                targets = idx + y_idxs
                counts[targets] += 1
                counts[2 * idx] += 1
                A[idx] = True
            
            new_score = get_score(counts)
            
            delta = new_score - current_score
            
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A.copy()
            else:
                # Revert
                if A[idx]:
                    # It was added, so remove it
                    y_idxs = np.where(A)[0]
                    # Note: A[idx] is True here, so idx is in y_idxs
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
        # Fallback
        return list(range(n + 1))
        
    # Check validity of best_A
    counts = evaluate(best_A)
    if np.any(counts[128:257] == 0):
        # If best found is invalid, try to fix or return a known valid set
        # The fallback range(n+1) is valid but has high peak.
        # We should probably return the best valid set found.
        # My get_score penalizes invalid sets heavily.
        # So best_score should correspond to a valid set if one was found.
        # If no valid set was found, best_A might be invalid.
        # Let's verify.
        if np.any(counts[128:257] == 0):
             # Return a simple valid set, e.g., all even numbers?
             # Or just range(n+1)
             return list(range(n + 1))

    return [i for i in range(n + 1) if best_A[i]]
