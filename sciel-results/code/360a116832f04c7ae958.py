import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    n = 128
    target_min = 128
    target_max = 256
    targets = list(range(target_min, target_max + 1))
    
    def compute_multiplicities(A):
        """Compute multiplicity of each target sum using unordered pairs (a,b) with a <= b."""
        A_set = set(A)
        mults = {t: 0 for t in targets}
        A_sorted = sorted(A)
        for i in range(len(A_sorted)):
            for j in range(i, len(A_sorted)):
                s = A_sorted[i] + A_sorted[j]
                if target_min <= s <= target_max:
                    mults[s] += 1
        return mults
    
    def score(A):
        """Return (max_multiplicity, coverage_ok). Lower max_mult is better."""
        mults = compute_multiplicities(A)
        max_mult = max(mults.values())
        # Check coverage
        for t in targets:
            if mults[t] == 0:
                return (float('inf'), False)
        return (max_mult, True)
    
    def is_valid(A):
        """Check if A covers all targets."""
        A_set = set(A)
        for t in targets:
            found = False
            for a in A:
                b = t - a
                if b in A_set and a <= b:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Initial solution: try a few heuristics
    best_A = None
    best_score = float('inf')
    
    # Heuristic 1: Use a subset that's somewhat dense
    # Try A as every other number in a range
    for offset in range(0, 10):
        A = list(range(offset, n+1, 2))
        if not A:
            continue
        if is_valid(A):
            s, ok = score(A)
            if ok and s < best_score:
                best_score = s
                best_A = list(A)
    
    # Heuristic 2: Random sets of various sizes
    for trial in range(50):
        size = random.randint(25, 45)
        A = random.sample(range(0, n+1), size)
        if is_valid(A):
            s, ok = score(A)
            if ok and s < best_score:
                best_score = s
                best_A = list(A)
    
    # If no valid solution found yet, use a larger set
    if best_A is None:
        # Use a denser set
        A = list(range(0, n+1, 2))  # every other number
        if not is_valid(A):
            A = list(range(0, n+1))
        best_A = list(A)
        best_score, _ = score(best_A)
    
    # Local search: try to swap elements to reduce peak multiplicity
    current_A = list(best_A)
    current_score = best_score
    
    def try_improve(A, max_iterations=10000):
        A = list(A)
        cur_score, _ = score(A)
        best_local = list(A)
        best_local_score = cur_score
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Pick a random element to remove and a random element to add
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            
            # Try adding a random new element
            new_elem = random.randint(0, n)
            
            A[idx] = new_elem
            
            if len(set(A)) != len(A):
                A[idx] = old_elem
                continue
            
            new_score, ok = score(A)
            if ok and new_score < best_local_score:
                best_local_score = new_score
                best_local = list(A)
            else:
                A[idx] = old_elem
        
        return best_local, best_local_score
    
    # Run local search with multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best
        A = list(best_A)
        perturbations = random.randint(1, 3)
        for _ in range(perturbations):
            idx = random.randint(0, len(A) - 1)
            A[idx] = random.randint(0, n)
        
        if len(set(A)) != len(A):
            continue
        
        if not is_valid(A):
            continue
        
        improved_A, improved_score = try_improve(A, max_iterations=5000)
        
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
    
    # Final polish: try single-element swaps more carefully
    if time.time() < deadline - 5:
        A = list(best_A)
        for _ in range(20000):
            if time.time() > deadline - 2:
                break
            
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            new_elem = random.randint(0, n)
            
            if new_elem in A:
                continue
            
            A[idx] = new_elem
            
            new_score, ok = score(A)
            if ok and new_score < best_score:
                best_score = new_score
                best_A = list(A)
            else:
                A[idx] = old_elem
    
    return list(best_A)
