import numpy as np
import time
import random

def construct():
    n = 128
    target_min = 128
    target_max = 256
    target_range = range(target_min, target_max + 1)
    
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin
    
    # State
    # A is a set of integers
    A = set()
    # counts[s] = number of pairs {u, v} in A such that u + v = s
    counts = np.zeros(257, dtype=np.int32)
    
    def get_peak():
        return np.max(counts[target_min:target_max+1])
        
    def get_uncovered():
        return np.sum(counts[target_min:target_max+1] == 0)
        
    def get_energy():
        peak = get_peak()
        unc = get_uncovered()
        if unc > 0:
            return 1000 * unc + peak
        else:
            # Add a small variance penalty to encourage uniform distribution
            # This helps escape local minima where peak is same but distribution is skewed
            # Sum of squares is a proxy for variance
            target_counts = counts[target_min:target_max+1]
            return peak + 0.01 * np.sum(target_counts * target_counts)

    def add_element(x):
        A.add(x)
        for u in A:
            if u == x:
                s = 2 * x
                if s <= 256:
                    counts[s] += 1
            else:
                s = u + x
                if s <= 256:
                    counts[s] += 1

    def remove_element(x):
        A.remove(x)
        s_xx = 2 * x
        if s_xx <= 256:
            counts[s_xx] -= 1
        for u in A:
            s = u + x
            if s <= 256:
                counts[s] -= 1

    def make_greedy_seed():
        nonlocal A, counts
        A = set()
        counts = np.zeros(257, dtype=np.int32)
        
        # Greedily add elements to cover all targets
        # We want to minimize the number of elements first? Or just get a valid set?
        # Let's just get a valid set.
        while get_uncovered() > 0:
            best_x = -1
            best_gain = -1
            for x in range(n + 1):
                if x in A:
                    continue
                # Calculate gain
                gain = 0
                s_xx = 2 * x
                if target_min <= s_xx <= target_max and counts[s_xx] == 0:
                    gain += 1
                for u in A:
                    s = x + u
                    if target_min <= s <= target_max and counts[s] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                # Stuck, shouldn't happen for this problem size
                break
            add_element(best_x)
            
    make_greedy_seed()
    
    current_energy = get_energy()
    best_energy = current_energy
    best_A = set(A)
    best_counts = counts.copy()
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    num_iterations = 0
    
    # Precompute possible moves? No, generate on the fly.
    # Elements: 0..128
    
    while time.time() - start_time < time_limit:
        # Generate a random move
        # Types: Add, Remove, Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        old_energy = current_energy
        
        if move_type == 'add':
            if len(A) == n + 1:
                continue
            # Pick random x not in A
            # Efficient way: pick random index, if in A, pick another?
            # Or just try a few times
            x = random.randint(0, n)
            attempts = 0
            while x in A and attempts < 10:
                x = random.randint(0, n)
                attempts += 1
            if x in A:
                continue
            
            add_element(x)
            new_energy = get_energy()
            
            if new_energy < old_energy or (random.random() < np.exp(-(new_energy - old_energy) / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_A = set(A)
                    best_counts = counts.copy()
            else:
                remove_element(x)
                
        elif move_type == 'remove':
            if len(A) == 0:
                continue
            x = random.choice(list(A))
            remove_element(x)
            new_energy = get_energy()
            
            if new_energy < old_energy or (random.random() < np.exp(-(new_energy - old_energy) / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_A = set(A)
                    best_counts = counts.copy()
            else:
                add_element(x)
                
        else: # swap
            if len(A) == 0:
                continue
            x = random.choice(list(A))
            y = random.randint(0, n)
            attempts = 0
            while y in A and attempts < 10:
                y = random.randint(0, n)
                attempts += 1
            if y in A:
                continue
            
            remove_element(x)
            add_element(y)
            new_energy = get_energy()
            
            if new_energy < old_energy or (random.random() < np.exp(-(new_energy - old_energy) / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_A = set(A)
                    best_counts = counts.copy()
            else:
                remove_element(y)
                add_element(x)
                
        num_iterations += 1
        
        # Cooling
        T = max(T * cooling, T_min)
        
        # Restart if stuck for too long?
        # Simple restart logic: if T is very low and no improvement for a while, reheat
        if num_iterations % 1000 == 0:
            if current_energy == best_energy and T < 0.1:
                # Reheat slightly
                T = 1.0
                # Optional: Perturb state
                # For now, just reheat

    # Return the best valid solution found
    # Ensure best_A is valid
    # It should be, as we track best_energy which includes penalty for invalidity
    # But just in case, check
    # If best_A is invalid, we should return a valid one.
    # The greedy seed was valid. So best_A is at least as good as the greedy seed.
    
    return list(best_A)
