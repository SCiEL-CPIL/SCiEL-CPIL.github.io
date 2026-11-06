import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80  # seconds, leaving 10s safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    targets = list(range(target_min, target_max + 1))
    
    def compute_mults(A_set):
        """Compute representation counts for all targets."""
        mults = np.zeros(target_max + 1, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if target_min <= s <= target_max:
                    mults[s] += 1
        return mults
    
    def is_covered(mults):
        """Check if all targets have at least one representation."""
        return all(mults[s] > 0 for s in targets)
    
    def peak_mult(mults):
        """Get the maximum multiplicity."""
        return max(mults[s] for s in targets)
    
    # Start with baseline
    A = set(range(64, 129))
    mults = compute_mults(A)
    best_A = set(A)
    best_peak = peak_mult(mults)
    
    # Local search: try removing elements
    def try_remove(A_set, mults):
        """Try removing each element and see if peak improves."""
        best_remove = None
        best_remove_peak = peak_mult(mults)
        
        for elem in sorted(A_set):
            if time.time() > deadline:
                break
            new_A = set(A_set)
            new_A.discard(elem)
            if not new_A:
                continue
            new_mults = compute_mults(new_A)
            if is_covered(new_mults):
                p = peak_mult(new_mults)
                if p < best_remove_peak:
                    best_remove_peak = p
                    best_remove = elem
        
        return best_remove, best_remove_peak
    
    # Local search: try adding elements
    def try_add(A_set, mults):
        """Try adding each element and see if peak improves."""
        best_add = None
        best_add_peak = peak_mult(mults)
        
        for elem in range(0, n + 1):
            if elem in A_set:
                continue
            if time.time() > deadline:
                break
            new_A = set(A_set)
            new_A.add(elem)
            new_mults = compute_mults(new_A)
            if is_covered(new_mults):
                p = peak_mult(new_mults)
                if p < best_add_peak:
                    best_add_peak = p
                    best_add = elem
        
        return best_add, best_add_peak
    
    # Greedy local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing
        elem_to_remove, new_peak = try_remove(A, mults)
        if elem_to_remove is not None and new_peak < best_peak:
            A.discard(elem_to_remove)
            mults = compute_mults(A)
            best_peak = new_peak
            best_A = set(A)
            improved = True
            continue
        
        # Try adding
        elem_to_add, new_peak = try_add(A, mults)
        if elem_to_add is not None and new_peak < best_peak:
            A.add(elem_to_add)
            mults = compute_mults(A)
            best_peak = new_peak
            best_A = set(A)
            improved = True
    
    # Try a few random restarts with different initial sets
    random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        
        # Generate a random initial set
        A_rand = set()
        # Ensure some structure: include elements that help cover extremes
        A_rand.add(0)
        A_rand.add(128)
        # Add random elements
        for i in range(40):
            A_rand.add(random.randint(0, 128))
        
        mults_rand = compute_mults(A_rand)
        if not is_covered(mults_rand):
            continue
        
        peak_rand = peak_mult(mults_rand)
        if peak_rand < best_peak:
            best_peak = peak_rand
            best_A = set(A_rand)
            A = set(A_rand)
            mults = mults_rand
            
            # Local search from here
            improved = True
            while improved and time.time() < deadline:
                improved = False
                elem_to_remove, new_peak = try_remove(A, mults)
                if elem_to_remove is not None and new_peak < best_peak:
                    A.discard(elem_to_remove)
                    mults = compute_mults(A)
                    best_peak = new_peak
                    best_A = set(A)
                    improved = True
                    continue
                elem_to_add, new_peak = try_add(A, mults)
                if elem_to_add is not None and new_peak < best_peak:
                    A.add(elem_to_add)
                    mults = compute_mults(A)
                    best_peak = new_peak
                    best_A = set(A)
                    improved = True
    
    return sorted(best_A)
