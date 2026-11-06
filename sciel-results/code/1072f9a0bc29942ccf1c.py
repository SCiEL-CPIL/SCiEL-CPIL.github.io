"""Improved additive basis construction with local search."""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    n = 128
    target_lo = n  # 128
    target_hi = 2 * n  # 256
    target_range = target_hi - target_lo + 1  # 129 values
    
    # Helper: compute representation counts for a given set A
    def compute_reps(A_set):
        """Compute number of unordered pairs (a,b) with a<=b from A that sum to each target."""
        A_list = sorted(A_set)
        reps = np.zeros(target_range, dtype=np.int32)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    reps[s - target_lo] += 1
        return reps
    
    def peak_mult(A_set):
        reps = compute_reps(A_set)
        return int(np.max(reps))
    
    def is_covered(A_set):
        reps = compute_reps(A_set)
        return bool(np.all(reps > 0))
    
    def get_representations(A_set):
        """Return list of (a,b) pairs for each target."""
        A_list = sorted(A_set)
        reps = {}
        for t in range(target_lo, target_hi + 1):
            reps[t] = []
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    reps[s].append((a, b))
        return reps
    
    # Initial solution: A = [64, 128]
    # This gives: for t in [128,256], we need a+b=t with a,b in [64,128]
    # Actually [64,128] means every element from 64 to 128.
    # Let's start with the seed: A = range(64, 129)
    A = set(range(64, 129))
    best_peak = peak_mult(A)
    best_A = set(A)
    
    # Try to improve via local search
    # Strategy: try removing elements that don't break coverage, 
    # and add elements to reduce peak multiplicity.
    
    random.seed(42)
    
    # First, try to find a better initial solution using a greedy approach
    # Start with all of [0,128] and try to remove elements
    
    # Let's try a different approach: start with a smaller set
    # For covering [128,256] with pairs from A ⊆ [0,128]:
    # If A contains x and y, then x+y must be in [128,256] for useful pairs.
    # Since max sum is 128+128=256 and min sum is 0+0=0,
    # we need pairs summing to exactly each value in [128,256].
    
    # Key insight: if A = [64,128], then for any t in [128,256],
    # we can write t = a + b where a = max(64, t-128), b = min(128, t-64)
    # The number of representations varies.
    
    # Let's do a local search: try swapping elements in and out
    
    def evaluate_swap(A_set, remove_elem, add_elem):
        """Evaluate the effect of removing remove_elem and adding add_elem."""
        if remove_elem == add_elem:
            return None
        new_A = set(A_set)
        if remove_elem in new_A:
            new_A.remove(remove_elem)
        if add_elem in new_A:
            return None  # already there
        new_A.add(add_elem)
        
        if not is_covered(new_A):
            return None
        return peak_mult(new_A)
    
    # Local search with time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A_list = list(A)
        random.shuffle(A_list)
        
        for elem in A_list:
            if time.time() >= deadline:
                break
            # Try removing elem
            if len(A) <= 2:
                continue
            new_A = set(A)
            new_A.remove(elem)
            if is_covered(new_A):
                new_peak = peak_mult(new_A)
                if new_peak < best_peak:
                    A = new_A
                    best_peak = new_peak
                    best_A = set(A)
                    improved = True
                    break
            # Try swapping elem with another element
            for other in range(n + 1):
                if time.time() >= deadline:
                    break
                if other == elem or other in A:
                    continue
                new_A = set(A)
                new_A.remove(elem)
                new_A.add(other)
                if is_covered(new_A):
                    new_peak = peak_mult(new_A)
                    if new_peak < best_peak:
                        A = new_A
                        best_peak = new_peak
                        best_A = set(A)
                        improved = True
                        break
            if improved:
                break
    
    # Try adding elements to reduce peak (if we can reduce peak by adding)
    # This is less common but possible
    if time.time() < deadline:
        for other in range(n + 1):
            if time.time() >= deadline:
                break
            if other in A:
                continue
            new_A = set(A)
            new_A.add(other)
            if is_covered(new_A):
                new_peak = peak_mult(new_A)
                if new_peak < best_peak:
                    A = new_A
                    best_peak = new_peak
                    best_A = set(A)
    
    # Return the best solution found
    return sorted(best_A)
