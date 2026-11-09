import time
import random
import numpy as np

N = 512
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute all 4-term arithmetic progressions
    # For each element x, store the other 3 elements in each 4-AP containing x
    aps_containing = {i: [] for i in range(1, N + 1)}
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))
    
    # Convert to numpy arrays for faster checking
    # For each x, store as a 2D array of shape (num_aps, 3)
    aps_array = {}
    for x in range(1, N + 1):
        if aps_containing[x]:
            aps_array[x] = np.array(aps_containing[x], dtype=np.int32)
        else:
            aps_array[x] = np.empty((0, 3), dtype=np.int32)
    
    def can_add_vectorized(x, current_bits):
        """Check if adding x creates a 4-AP using numpy"""
        aps = aps_array[x]
        if len(aps) == 0:
            return True
        # current_bits is a numpy bool array of length N+1
        # For each AP (a,b,c), check if current_bits[a] & current_bits[b] & current_bits[c]
        # We need: NOT EXISTS (a,b,c) in aps such that all three are in current_bits
        # Equivalent: FOR ALL (a,b,c), NOT (all three in current_bits)
        # i.e., FOR ALL (a,b,c), at least one of a,b,c is NOT in current_bits
        cols = aps  # shape (num_aps, 3)
        # Extract bits for all three positions
        bits_a = current_bits[cols[:, 0]]
        bits_b = current_bits[cols[:, 1]]
        bits_c = current_bits[cols[:, 2]]
        # If any row has all three True, we cannot add x
        if np.any(bits_a & bits_b & bits_c):
            return False
        return True
    
    def greedy_randomized(current_bits=None):
        """Greedy with random ordering, returns numpy bool array"""
        if current_bits is None:
            current_bits = np.zeros(N + 1, dtype=bool)
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if can_add_vectorized(x, current_bits):
                current_bits[x] = True
        return current_bits
    
    def local_search(current_bits):
        """
        Simulated annealing style local search with efficient constraint checking.
        """
        current_bits = current_bits.copy()
        current_size = np.sum(current_bits)
        best_bits = current_bits.copy()
        best_size = current_size
        temperature = 10.0
        cooling = 0.995
        
        while time.time() - start_time < TIME_LIMIT - 5:
            # Try to add a random element
            if random.random() < 0.7:
                # Add phase
                candidates = [x for x in range(1, N + 1) if not current_bits[x]]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add_vectorized(x, current_bits):
                    current_bits[x] = True
                    new_size = current_size + 1
                    if new_size > best_size:
                        best_size = new_size
                        best_bits = current_bits.copy()
                    current_size = new_size
                else:
                    # Can't add, try removing an element that's blocking
                    # Find which APs are blocking
                    aps = aps_array[x]
                    if len(aps) > 0:
                        bits_a = current_bits[aps[:, 0]]
                        bits_b = current_bits[aps[:, 1]]
                        bits_c = current_bits[aps[:, 2]]
                        blocking = bits_a & bits_b & bits_c
                        if np.any(blocking):
                            # Pick a random blocking AP
                            blocking_idx = np.where(blocking)[0]
                            idx = random.choice(blocking_idx)
                            # Remove one of the three elements in that AP
                            remove_choices = [aps[idx, 0], aps[idx, 1], aps[idx, 2]]
                            remove_x = random.choice(remove_choices)
                            if current_bits[remove_x]:
                                # Check if removing this element allows adding x
                                trial_bits = current_bits.copy()
                                trial_bits[remove_x] = False
                                if can_add_vectorized(x, trial_bits):
                                    trial_bits[x] = True
                                    new_size = current_size - 1 + 1
                                    # Accept if better or with probability based on temperature
                                    if new_size > current_size:
                                        current_bits = trial_bits
                                        current_size = new_size
                                        if new_size > best_size:
                                            best_size = new_size
                                            best_bits = current_bits.copy()
                                    else:
                                        # Same size, accept with probability
                                        delta_e = 0  # no change in size
                                        if random.random() < np.exp(-delta_e / temperature):
                                            current_bits = trial_bits
                                            # current_size unchanged
            else:
                # Remove phase: remove a random element and try to refill
                if np.sum(current_bits) == 0:
                    continue
                remove_x = random.choice([x for x in range(1, N + 1) if current_bits[x]])
                trial_bits = current_bits.copy()
                trial_bits[remove_x] = False
                # Try to fill with new elements
                added = 0
                candidates = [x for x in range(1, N + 1) if not trial_bits[x]]
                random.shuffle(candidates)
                for x in candidates:
                    if can_add_vectorized(x, trial_bits):
                        trial_bits[x] = True
                        added += 1
                new_size = current_size - 1 + added
                if new_size > current_size:
                    current_bits = trial_bits
                    current_size = new_size
                    if new_size > best_size:
                        best_size = new_size
                        best_bits = current_bits.copy()
                elif new_size == current_size:
                    # Accept with some probability to escape local optima
                    if random.random() < 0.1:
                        current_bits = trial_bits
                # else: reject
            
            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01
        
        return best_bits
    
    best_bits = None
    best_size = 0
    
    # Run multiple restarts
    for i in range(200):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        bits = greedy_randomized()
        bits = local_search(bits)
        size = int(np.sum(bits))
        
        if size > best_size:
            best_size = size
            best_bits = bits.copy()
    
    # Also try starting from a structured initial set
    if time.time() - start_time < TIME_LIMIT - 10:
        # Try starting from all odd numbers
        odd_bits = np.zeros(N + 1, dtype=bool)
        odd_bits[1::2] = True
        # Remove any 4-APs
        for d in range(1, (N - 1) // 3 + 1):
            for a in range(1, N - 3 * d + 1, 2):  # only odd starts
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if odd_bits[a] and odd_bits[b] and odd_bits[c] and odd_bits[e]:
                    # Remove one element
                    odd_bits[e] = False
        bits = local_search(odd_bits)
        size = int(np.sum(bits))
        if size > best_size:
            best_size = size
            best_bits = bits.copy()
    
    # Convert to list
    if best_bits is None:
        return [1]
    return [int(x) for x in range(1, N + 1) if best_bits[x]]

