import random
import time

def construct():
    N = 1000
    start_time = time.time()
    limit = 115  # Leave 5s buffer
    
    # Helper to initialize counts for a set A
    def get_counts(A):
        counts = {}
        for i, a in enumerate(A):
            for b in A[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    # Helper to check if x can be added to A given counts
    def is_valid_add(A, x, counts):
        if x in A:
            return False
        if counts.get(2*x, 0) >= 2:
            return False
        for y in A:
            if counts.get(x+y, 0) >= 2:
                return False
        return True

    # Helper to compute valid array
    # valid[x] = 1 if x not in A and is_valid_add(A, x, counts)
    def compute_valid(A, counts):
        valid = [0] * (N + 1)
        A_set = set(A)
        for x in range(1, N + 1):
            if x not in A_set:
                if is_valid_add(A, x, counts):
                    valid[x] = 1
        return valid

    # Initialize with Greedy Ascending
    A = []
    counts = {}
    for x in range(1, N + 1):
        if is_valid_add(A, x, counts):
            A.append(x)
            counts[2*x] = counts.get(2*x, 0) + 1
            for y in A[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
    best_size = len(A)
    best_A = list(A)
    
    # Initialize validity array
    valid = compute_valid(A, counts)
    
    # Local Search
    while time.time() - start_time < limit:
        # Try to add
        if any(valid):
            # Pick a random valid index
            valid_indices = [i for i in range(1, N + 1) if valid[i]]
            x = random.choice(valid_indices)
            
            # Add x
            A.append(x)
            counts[2*x] = counts.get(2*x, 0) + 1
            for y in A[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
            # Update valid array
            # This is O(N)
            valid = compute_valid(A, counts)
            
            if len(A) > best_size:
                best_size = len(A)
                best_A = list(A)
        else:
            # Local optimum, try to escape
            # Remove a random element
            if not A:
                break
            x = random.choice(A)
            A.remove(x)
            
            # Update counts
            # Remove sums involving x
            counts[2*x] -= 1
            if counts[2*x] == 0:
                del counts[2*x]
            for y in A:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            
            # Update valid array
            valid = compute_valid(A, counts)
            
            # Now try to add any valid element (greedy fill step)
            # We can add multiple if we want, but let's do one step at a time
            # Actually, if we removed one, we might be able to add one or more.
            # Let's just let the next loop iteration handle the add.
            # To make it more aggressive, we can try to add immediately.
            
            # If we just removed, we are smaller. We want to get back.
            # The loop will try to add in the next iteration.
            # But we might oscillate. 
            # Let's force an add if possible in the same step?
            # No, the `if any(valid)` block handles adds.
            # So this `else` block just performs a removal.
            # The next iteration will likely add something back.
            # This is a random walk in size.
            
            # To bias towards larger sets, we should only remove if we are "stuck" 
            # and maybe accept smaller sets with low probability (SA style).
            # But since we don't have a temperature, let's just do it.
            # It acts as a perturbation.

    return best_A
