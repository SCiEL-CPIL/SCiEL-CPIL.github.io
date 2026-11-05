import time
import random
import itertools
import numpy as np

def is_sidon_fast(subset):
    """Fast Sidon check using numpy for vectorized pair sum computation."""
    n = len(subset)
    if n <= 2:
        return True
    arr = np.array(subset, dtype=np.int64)
    # Compute all pair sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    # Check for duplicates
    return len(np.unique(sums)) == len(sums)

def is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def has_sidon_subset(A, k):
    """
    Check if A has ANY Sidon subset of size k.
    Returns True if found, False otherwise.
    This is faster than max_sidon_size when we only care about a threshold.
    """
    n = len(A)
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def count_sidon_subsets(A, k):
    """
    Count the number of Sidon subsets of size k in A.
    Lower count means "harder" to find a Sidon subset, which is good for us.
    """
    n = len(A)
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            count += 1
    return count

def count_sidon_subsets_fast(A, k):
    """Faster count using numpy for the Sidon check."""
    n = len(A)
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon_fast(combo):
            count += 1
    return count

def compute_collision_score(A, target_k):
    """
    Compute a smooth surrogate: for each k-subset, measure how close it is to being Sidon.
    Lower score = more collisions = harder to find Sidon subsets.
    """
    n = len(A)
    total = 0.0
    for combo in itertools.combinations(A, target_k):
        arr = np.array(combo, dtype=np.int64)
        m = len(arr)
        i_idx, j_idx = np.triu_indices(m)
        sums = arr[i_idx] + arr[j_idx]
        # Count number of duplicate pairs
        unique_sums = np.unique(sums)
        num_dupes = len(sums) - len(unique_sums)
        # Score: higher = more collisions = better for us
        total += num_dupes
    return total

def construct():
    start_time = time.time()
    # Safety margin: stop 5 seconds before 60s limit
    deadline = start_time + 55.0
    
    # Initial candidate: Try to create many sum collisions
    # A good strategy: use values that create arithmetic progressions
    # which naturally create sum collisions
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = max_sidon_size(best_A)
    
    # If we start at 4, we are looking for 3.
    # Track soft metric: collision score for Sidon subsets of size best_val
    target_k = best_val
    current_score = compute_collision_score(current_A, target_k)
    best_score = current_score
    
    # Search parameters
    iterations = 0
    no_improve_counter = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Create a perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Strategy: Try to create collisions by picking values that
        # are symmetric around existing values or create APs
        if random.random() < 0.4:
            # Pick a value that creates a collision with existing sums
            # Find a pair (i,j) and pick new_val such that new_val + A[i] = A[j] + A[k]
            # or new_val + new_val = A[i] + A[j]
            if len(A_new) >= 2:
                i, j = random.sample(range(12), 2)
                # Option 1: new_val = A[j] + A[k] - A[i] for some k
                k = random.randint(0, 11)
                if k != i and k != j:
                    new_val = A_new[j] + A_new[k] - A_new[i]
                else:
                    new_val = A_new[j] + A_new[k] - A_new[i]
                # Option 2: new_val = (A[i] + A[j]) / 2 (if integer)
                s = A_new[i] + A_new[j]
                if s % 2 == 0:
                    new_val = s // 2
            else:
                new_val = random.randint(0, 128)
        elif random.random() < 0.7:
            # Small step to create near-collisions
            step = random.randint(-3, 3)
            new_val = A_new[idx] + step
        else:
            # Random value in a small window
            min_val = min(current_A)
            max_val = max(current_A)
            window = max(5, (max_val - min_val) // 2)
            new_val = random.randint(max(0, min_val - window), min(128, max_val + window))
        
        # Ensure validity: [0, 128] and distinct
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)
            
        if new_val in A_new:
            # Try to resolve conflict
            attempts = 0
            while new_val in A_new and attempts < 10:
                new_val = random.randint(0, 128)
                attempts += 1
            if new_val in A_new:
                continue
        
        A_new[idx] = new_val
        
        # Evaluate A_new
        # First, check if we improved the max Sidon size
        if has_sidon_subset(A_new, best_val):
            # Max size is still at least best_val
            # Use smooth surrogate: collision score
            new_score = compute_collision_score(A_new, best_val)
            
            # Accept if smooth metric is better (more collisions)
            if new_score > best_score:
                current_A = A_new
                current_score = new_score
                best_score = new_score
                no_improve_counter = 0
            else:
                # Accept with small probability to escape local minima
                if random.random() < 0.15:
                    current_A = A_new
                    current_score = new_score
                no_improve_counter += 1
        else:
            # No Sidon subset of size best_val
            # This means max_sidon_size(A_new) < best_val
            # This is an improvement!
            
            new_val_score = max_sidon_size(A_new)
            
            if new_val_score < best_val:
                best_val = new_val_score
                best_A = A_new
                current_A = A_new
                no_improve_counter = 0
                
                # Update target_k and scores for the new best_val
                target_k = best_val
                current_score = compute_collision_score(current_A, target_k)
                best_score = current_score
                
                # If we reached 3, try a few more iterations to see if we can get to 2
                if best_val <= 3:
                    # Run a quick local search to try to break 3-element Sidon sets
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        A_try = list(current_A)
                        idx2 = random.randint(0, 11)
                        # Try to create a collision in a 3-element subset
                        # Pick a 3-subset and try to break it
                        if has_sidon_subset(A_try, 3):
                            # Find a 3-subset that is Sidon and try to break it
                            found = False
                            for combo in itertools.combinations(A_try, 3):
                                if is_sidon(combo):
                                    # Try to modify one element to create a collision
                                    for pos in range(3):
                                        old_val = combo[pos]
                                        # Try values close to old_val
                                        for delta in range(-5, 6):
                                            if delta == 0:
                                                continue
                                            new_v = old_val + delta
                                            if 0 <= new_v <= 128 and new_v not in A_try:
                                                A_try2 = list(A_try)
                                                # Find index of old_val in A_try
                                                for i2, v2 in enumerate(A_try):
                                                    if v2 == old_val:
                                                        A_try2[i2] = new_v
                                                        break
                                                if not has_sidon_subset(A_try2, 3):
                                                    # Success! We broke all 3-Sidon sets
                                                    new_vs = max_sidon_size(A_try2)
                                                    if new_vs < best_val:
                                                        best_val = new_vs
                                                        best_A = A_try2
                                                        current_A = A_try2
                                                        target_k = best_val
                                                        current_score = compute_collision_score(current_A, target_k)
                                                        best_score = current_score
                                                        found = True
                                                        break
                                        if found:
                                            break
                                    if found:
                                        break
                            if found:
                                break
                    if best_val <= 2:
                        return best_A
            else:
                pass
        
        # Periodic reset if stuck
        if no_improve_counter > 300:
            # Reset to best_A and perturb
            current_A = list(best_A)
            current_score = best_score
            no_improve_counter = 0
            
            # Apply a few random perturbations
            for _ in range(5):
                idx = random.randint(0, 11)
                A_new = list(current_A)
                A_new[idx] = random.randint(0, 128)
                while A_new.count(A_new[idx]) > 1:
                    A_new[idx] = random.randint(0, 128)
                current_A = A_new
                current_score = compute_collision_score(current_A, target_k)

    return best_A
