"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _compute_peak(A):
    """Compute the maximum representation multiplicity in [128, 256]."""
    A_set = set(A)
    max_mult = 0
    for target in range(128, 257):
        count = 0
        for a in A:
            b = target - a
            if b in A_set:
                if a <= b:
                    count += 1
        if count > max_mult:
            max_mult = count
    return max_mult

def _is_valid(A):
    """Check if every integer in [128, 256] is representable as sum of two elements of A."""
    A_set = set(A)
    for target in range(128, 257):
        found = False
        for a in A:
            b = target - a
            if b in A_set and a <= b:
                found = True
                break
        if not found:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial solution: the seed
    best_A = list(range(64, 129))
    best_peak = _compute_peak(best_A)
    
    # Try a few different initializations
    random.seed(42)
    
    # Strategy: start with a reasonable set and do local search
    # The set should be spread out to avoid clustering
    
    # Try different sizes
    for size in [50, 55, 60, 65, 70]:
        if time.time() > deadline - 10:
            break
        
        # Create initial set: spread elements across [0, 128]
        # We need elements that can sum to cover [128, 256]
        # If we have elements in [0, 128], sums range from 0 to 256
        # To cover [128, 256], we need good coverage
        
        # Initialize with a mix of low and high values
        A = set()
        # Add some low values
        for i in range(size // 4):
            A.add(random.randint(0, 40))
        # Add some mid values
        for i in range(size // 2):
            A.add(random.randint(40, 90))
        # Add some high values
        for i in range(size - len(A)):
            A.add(random.randint(90, 128))
        
        A = list(A)
        if len(A) < 10:
            continue
        
        # Ensure validity first
        if not _is_valid(A):
            # Try to fix by adding missing elements
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for target in range(128, 257):
                    found = False
                    A_set = set(A)
                    for a in A:
                        b = target - a
                        if b in A_set and a <= b:
                            found = True
                            break
                    if not found:
                        # Add a random element that might help
                        if time.time() > deadline - 5:
                            break
                        a_new = random.randint(0, 128)
                        if a_new not in A_set:
                            A.append(a_new)
                            changed = True
            if not _is_valid(A):
                continue
        
        peak = _compute_peak(A)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A)
        
        # Local search: try to improve by swapping elements
        iterations = 0
        while time.time() < deadline - 5 and iterations < 5000:
            iterations += 1
            # Pick a random element to remove
            idx = random.randint(0, len(A) - 1)
            old_elem = A[idx]
            A[idx] = -1  # mark as removed
            A_set = set(x for x in A if x >= 0)
            
            # Try adding a new element
            new_elem = random.randint(0, 128)
            if new_elem in A_set:
                A[idx] = old_elem
                continue
            
            A[idx] = new_elem
            if not _is_valid(A):
                A[idx] = old_elem
                continue
            
            new_peak = _compute_peak(A)
            if new_peak < peak:
                peak = new_peak
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            else:
                A[idx] = old_elem
        
        if time.time() > deadline - 5:
            break
    
    # Final check
    if not _is_valid(best_A):
        return list(range(64, 129))
    
    return best_A
# EVOLVE-BLOCK-END
