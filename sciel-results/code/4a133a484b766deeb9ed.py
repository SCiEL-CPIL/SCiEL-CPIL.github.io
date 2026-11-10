import time
import random

def _compute_peak(A):
    """Compute the maximum representation multiplicity for targets [128,256]."""
    A_set = set(A)
    max_mult = 0
    for t in range(128, 257):
        count = 0
        for a in A:
            b = t - a
            if b in A_set and a <= b:
                count += 1
        if count == 0:
            return -1  # invalid, not covered
        if count > max_mult:
            max_mult = count
    return max_mult

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Start with the seed
    best_A = list(range(64, 129))
    best_peak = _compute_peak(best_A)
    
    if best_peak <= 0:
        return best_A
    
    random.seed(42)
    
    # Local search: try modifications
    def try_modify(A, peak):
        """Try random modifications and return improved (A, peak) or None."""
        n = len(A)
        A_list = list(A)
        A_set = set(A_list)
        
        # Random choice of operation
        op = random.random()
        
        if op < 0.4 and n > 10:
            # Remove a random element
            idx = random.randint(0, n-1)
            new_A = A_list[:idx] + A_list[idx+1:]
            new_peak = _compute_peak(new_A)
            if new_peak > 0 and new_peak < peak:
                return new_A, new_peak
        elif op < 0.8:
            # Add a random element not in A
            candidates = [x for x in range(0, 129) if x not in A_set]
            if candidates:
                new_elem = random.choice(candidates)
                new_A = A_list + [new_elem]
                new_peak = _compute_peak(new_A)
                if new_peak > 0 and new_peak < peak:
                    return new_A, new_peak
        else:
            # Swap: remove one, add another
            if n > 5:
                idx = random.randint(0, n-1)
                removed = A_list[idx]
                candidates = [x for x in range(0, 129) if x not in A_set or x == removed]
                if candidates:
                    new_elem = random.choice(candidates)
                    if new_elem != removed:
                        new_A = A_list[:idx] + [new_elem] + A_list[idx+1:]
                        new_peak = _compute_peak(new_A)
                        if new_peak > 0 and new_peak < peak:
                            return new_A, new_peak
    
    # Hill climbing with restarts
    max_iters = 50000
    for it in range(max_iters):
        if time.time() - start_time > deadline:
            break
        
        result = try_modify(best_A, best_peak)
        if result is not None:
            best_A, best_peak = result
    
    # Final polish: try all single-element swaps to see if we can improve further
    if time.time() - start_time < deadline - 5:
        A_set = set(best_A)
        improved = True
        while improved:
            improved = False
            for i in range(len(best_A)):
                if time.time() - start_time > deadline - 2:
                    break
                old_elem = best_A[i]
                for new_elem in range(0, 129):
                    if new_elem in A_set and new_elem != old_elem:
                        continue
                    new_A = best_A[:i] + [new_elem] + best_A[i+1:]
                    new_peak = _compute_peak(new_A)
                    if new_peak > 0 and new_peak < best_peak:
                        best_A = new_A
                        best_peak = new_peak
                        A_set = set(best_A)
                        improved = True
                        break
    
    return best_A
