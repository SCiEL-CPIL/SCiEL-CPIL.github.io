import time
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Fast sum and diff calculation using numpy
    # U is a numpy array of int64
    # Sums: U[i] + U[j] for i <= j
    # Diffs: U[j] - U[i] for all i, j (includes 0 and negatives)
    
    # To handle large U efficiently, we use the property that 
    # |U+U| is the number of unique sums.
    # |U-U| is the number of unique differences.
    
    # For n up to ~50, O(n^2) is fast enough with numpy.
    
    # Generate all pairs
    # For sums: i <= j
    # For diffs: all i, j
    
    # A simpler and often faster way for small n is to use sets in Python, 
    # but numpy might be faster for the sorting/unique step if n is larger.
    # Given the 120s limit, we can afford O(n^2) Python sets for n < 100.
    # Let's stick to the robust Python set method but optimize the loop slightly.
    
    sums = set()
    diffs = set()
    
    # Optimization: use local variables for speed
    U_list = U
    
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sums.add(ui + U_list[j])
        
        # Diffs: U[j] - U[i]
        # We can just do all pairs for diffs
        # Actually, diffs is symmetric. U-U contains 0.
        # Let's just compute all differences.
        
    # Recompute diffs properly
    diffs = set()
    for i in range(n):
        ui = U_list[i]
        for j in range(n):
            diffs.add(U_list[j] - ui)
            
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U_list)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    # Import log here to avoid global import overhead in tight loop? 
    # No, import is fine.
    from math import log
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Initial seed: The geometric set found by the previous best approach
    # Or a simple geometric progression
    # Let's start with a geometric progression: 1, 3, 9, 27, 81...
    # U = [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561]
    # This has huge max U, which hurts the log(D) term.
    # We want a compact set with high ratio.
    
    # A known good structure for sum/diff sets is related to Sidon sets or similar.
    # Let's initialize with a small random set or a specific pattern.
    
    # Strategy: Simulated Annealing on the set elements.
    # Representation: Sorted list of integers.
    # Moves: 
    # 1. Change an element to a nearby value.
    # 2. Add a new element.
    # 3. Remove an element.
    
    # Initial candidate: 
    # Let's try a set based on powers of 2 but compressed?
    # Or just start with [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]?
    
    # Let's use the "geometric" insight from the parent as a starting point.
    # The parent found something around score 1.026.
    # A typical good set might be like [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]?
    # Squares have |U+U| ~ n^2, |U-U| ~ n^2. Ratio ~ 1.
    # We need |U-U| > |U+U|.
    
    # Let's try a set with rapidly increasing gaps.
    # U = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # This is powers of 2.
    # Sums: unique sums of distinct powers are all numbers < 2^(n+1). Count ~ 2^n.
    # Diffs: differences of powers.
    
    # Let's just start with a random set of size 10 in range [0, 1000] and let SA find its way.
    # But random might be too far.
    
    # Let's try a "greedy" start.
    # Start with [0, 1].
    # Add elements that maximize score.
    
    current_set = [0, 1]
    current_score = score_set(current_set)
    
    # Greedy expansion
    max_val = 100
    while len(current_set) < 15 and time.time() < deadline - 60:
        best_next = None
        best_next_score = current_score
        for candidate in range(2, max_val + 1):
            if candidate in current_set:
                continue
            new_set = sorted(current_set + [candidate])
            s = score_set(new_set)
            if s > best_next_score:
                best_next_score = s
                best_next = candidate
        if best_next is None:
            break
        current_set.append(best_next)
        current_set.sort()
        current_score = best_next_score
        max_val = max(current_set) * 2 + 10 # Expand search range
    
    best_set = list(current_set)
    best_score = current_score
    
    # Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    min_T = 1e-4
    
    # We will perform moves on the list of elements.
    # To make it faster, we can precompute something? 
    # No, just fast evaluation.
    
    # We need to handle the case where the set size changes.
    
    # Let's define a neighbor generator.
    # 1. Perturb an element: pick i, change U[i] to U[i] + delta.
    # 2. Add element: pick random value.
    # 3. Remove element: pick i, remove U[i].
    
    # We must ensure 0 is always in the set.
    
    steps = 0
    max_steps = 20000 # Adjust based on speed
    
    while time.time() < deadline - 10 and T > min_T:
        steps += 1
        
        # Generate neighbor
        # 80% perturb, 10% add, 10% remove
        move_type = random.random()
        
        new_set = list(current_set)
        
        if move_type < 0.8:
            # Perturb
            if len(new_set) < 2:
                continue
            i = random.randint(1, len(new_set) - 1) # Don't change 0
            delta = random.randint(-10, 10)
            new_val = new_set[i] + delta
            if new_val <= 0:
                continue
            if new_val in new_set:
                continue
            new_set[i] = new_val
            new_set.sort()
        elif move_type < 0.9:
            # Add
            if len(new_set) >= 20: # Cap size
                continue
            # Add near existing elements or in gaps
            if random.random() < 0.5:
                # Near max
                new_val = max(new_set) + random.randint(1, 20)
            else:
                # Random in range
                new_val = random.randint(1, max(new_set) + 50)
            
            if new_val in new_set:
                continue
            new_set.append(new_val)
            new_set.sort()
        else:
            # Remove
            if len(new_set) <= 3:
                continue
            i = random.randint(1, len(new_set) - 1)
            new_set.pop(i)
        
        if new_set == current_set:
            continue
            
        new_score = score_set(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Metropolis criterion
            delta_E = current_score - new_score
            if delta_E < 0:
                # Should not happen if we check > above, but just in case
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
        
        T *= cooling
        
        # Adaptive cooling: if stuck, cool faster? Or keep going?
        # Standard cooling is fine.
        
    # Final local search polish on the best set found
    # Try small perturbations on the best set
    current = list(best_set)
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        n = len(current)
        for i in range(1, n):
            old_val = current[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if new_val in current:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set.sort()
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding
        if not improved and n < 15:
            for new_val in range(1, max(current) + 10):
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try removing
        if not improved and n > 3:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

    if current_score > best_score:
        best_score = current_score
        best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
