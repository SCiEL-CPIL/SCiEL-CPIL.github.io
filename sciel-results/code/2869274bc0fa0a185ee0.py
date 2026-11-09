import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0  # Safety margin
    
    N = 4096
    MAX_K = 80
    
    # Maximum possible sum is 3 * 4096 = 12288
    MAX_SUM = 3 * N
    # Use a bytearray for fast sum checking
    # index 0..MAX_SUM
    # We will use a version of this for incremental checks
    
    def can_add(x, S, occupied_sums):
        """
        Checks if x can be added to S given the current occupied_sums.
        occupied_sums is a bytearray where occupied_sums[s] == 1 if sum s is used.
        Returns (bool, new_occupied_sums) if successful, or (False, None).
        Note: We don't actually need to return the new array if we just want to check, 
        but for the greedy loop, we need to update it.
        """
        k = len(S)
        
        # Check 3x
        s = 3 * x
        if occupied_sums[s]:
            return False, None
        
        # Check 2x + y
        for y in S:
            s = 2 * x + y
            if occupied_sums[s]:
                return False, None
        
        # Check x + y + z
        for i in range(k):
            y = S[i]
            for j in range(i, k):
                z = S[j]
                s = x + y + z
                if occupied_sums[s]:
                    return False, None
        
        return True, None

    def add_element(x, S, occupied_sums):
        """
        Adds x to S and updates occupied_sums.
        Assumes x can be added.
        """
        S.append(x)
        k = len(S) - 1 # index of x
        
        # Mark 3x
        occupied_sums[3 * x] = 1
        
        # Mark 2x + y
        for y in S[:-1]:
            occupied_sums[2 * x + y] = 1
            
        # Mark x + y + z
        for i in range(k):
            y = S[i]
            for j in range(i, k):
                z = S[j]
                occupied_sums[x + y + z] = 1

    def remove_element(x, S, occupied_sums):
        """
        Removes x from S and clears its sums in occupied_sums.
        Note: This is tricky because other triples might also sum to the same value.
        So we can't just clear the bits. We need to recount or use a counter array.
        For simplicity in SA, we might just rebuild the occupied_sums from scratch 
        if we use a swap move, or use a counter array.
        """
        # Find index
        idx = S.index(x)
        S.pop(idx)
        
        # To be safe and simple, we'll rebuild occupied_sums in the SA loop 
        # if we do swaps, or we use a counter array.
        # Let's use a counter array for robustness in SA.
        pass

    # Better approach for SA: Use a counter array `sum_counts`
    # sum_counts[s] = number of triples in S that sum to s.
    # A set is valid if all sum_counts[s] <= 1.
    
    def get_sum_counts(S):
        counts = np.zeros(MAX_SUM + 1, dtype=np.uint8)
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = S[i] + S[j] + S[k]
                    counts[s] += 1
        return counts

    def count_conflicts_from_counts(counts):
        # Number of sums with count > 1
        return np.sum(counts > 1)

    # Greedy Construction
    def greedy_random_restart(attempts=50):
        best_set = []
        best_size = 0
        
        for _ in range(attempts):
            if time.time() > DEADLINE - 5.0:
                break
                
            S = []
            # Use a bytearray for speed in greedy phase
            occupied = bytearray(MAX_SUM + 1)
            
            # Randomize candidate order
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added
                # We need to check against current S and occupied
                # We'll do the check inline for speed
                k = len(S)
                valid = True
                
                # 3x
                if occupied[3 * x]:
                    valid = False
                else:
                    # 2x + y
                    for y in S:
                        if occupied[2 * x + y]:
                            valid = False
                            break
                    if valid:
                        # x + y + z
                        for i in range(k):
                            y = S[i]
                            for j in range(i, k):
                                z = S[j]
                                if occupied[x + y + z]:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    # Add x
                    S.append(x)
                    # Update occupied
                    occupied[3 * x] = 1
                    for y in S[:-1]:
                        occupied[2 * x + y] = 1
                    for i in range(len(S) - 1):
                        y = S[i]
                        for j in range(i, len(S) - 1):
                            z = S[j]
                            occupied[x + y + z] = 1
                    
                    if len(S) >= MAX_K:
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
                
        return best_set

    # Simulated Annealing
    def simulated_annealing(initial_set):
        S = initial_set[:]
        if not S:
            S = [1, 3, 9]
            
        # Calculate initial conflicts
        counts = get_sum_counts(S)
        current_conflicts = count_conflicts_from_counts(counts)
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        best_set = S[:]
        best_conflicts = current_conflicts
        best_size = len(S)
        
        # We want to minimize conflicts, and among zero-conflict sets, maximize size.
        # Objective: cost = conflicts * 1000 - size
        
        while time.time() < DEADLINE - 2.0 and T > min_T:
            # Generate neighbor: swap one element
            if len(S) > 0:
                idx = random.randint(0, len(S) - 1)
                old_val = S[idx]
                
                # Pick a new random value
                new_val = random.randint(1, N)
                if new_val == old_val:
                    T *= cooling
                    continue
                    
                S[idx] = new_val
                
                # Check for duplicates in S
                if S.count(new_val) > 1:
                    S[idx] = old_val
                    T *= cooling
                    continue
                    
                # Recalculate conflicts
                # This is O(K^3). For K=40, it's fast.
                new_counts = get_sum_counts(S)
                new_conflicts = count_conflicts_from_counts(new_counts)
                
                delta_conflicts = new_conflicts - current_conflicts
                delta_size = len(S) - len(S) # Size doesn't change in swap
                
                # Cost difference
                # We prioritize reducing conflicts.
                # If conflicts are 0, we might want to allow swaps that keep 0 conflicts 
                # but potentially allow future growth? 
                # Since size is fixed in a swap, we just minimize conflicts.
                
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(S) > best_size):
                        best_conflicts = current_conflicts
                        best_size = len(S)
                        best_set = S[:]
                elif new_conflicts == current_conflicts:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.1:
                        current_conflicts = new_conflicts
                        if current_conflicts == 0:
                            best_set = S[:]
                else:
                    # Accept if exp(-delta/T)
                    if delta_conflicts > 0:
                        if random.random() < np.exp(-delta_conflicts / T):
                            current_conflicts = new_conflicts
                    else:
                        # Should not happen if logic is correct
                        current_conflicts = new_conflicts
            
            T *= cooling
            
        return best_set

    # Run Greedy
    print("Running Greedy...")
    greedy_set = greedy_random_restart(20)
    print(f"Greedy best size: {len(greedy_set)}")
    
    # Run SA
    print("Running SA...")
    sa_set = simulated_annealing(greedy_set)
    
    if len(sa_set) > len(greedy_set):
        final_set = sa_set
    else:
        final_set = greedy_set
        
    # Final check and cleanup
    # Ensure distinctness
    final_set = list(set(final_set))
    
    # Verify validity
    def is_valid(S):
        counts = get_sum_counts(S)
        return count_conflicts_from_counts(counts) == 0
        
    if not is_valid(final_set):
        # Fallback to a known small valid set
        final_set = [1, 3, 9, 27]
        
    return final_set

# Fallback if construct fails or times out unexpectedly
if __name__ == "__main__":
    res = construct()
    print(res)
