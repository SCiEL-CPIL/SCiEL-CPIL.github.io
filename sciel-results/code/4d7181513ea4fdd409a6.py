import random
import time
import math

def construct():
    # Constants
    MAX_VAL = 128
    MIN_VAL = 2
    CANDIDATES = list(range(MIN_VAL, MAX_VAL + 1))
    
    def count_collisions(seq):
        """Count the number of duplicate products in consecutive subarrays."""
        if not seq:
            return 0
        prods = []
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        # Count duplicates
        from collections import Counter
        counts = Counter(prods)
        # Number of pairs that are duplicates? Or just number of excess items?
        # If a product appears k times, it contributes k-1 to the "collision count" if we define it as "number of elements that are not unique".
        # For penalty, sum(k-1) for all k>1 is a good measure of "how invalid".
        collisions = sum(k - 1 for k in counts.values() if k > 1)
        return collisions

    def get_energy(seq):
        """
        Energy to be minimized.
        We want to maximize length.
        We want to minimize collisions.
        If valid (collisions == 0), energy = -len(seq).
        If invalid, energy = -len(seq) + large_penalty * collisions.
        """
        collisions = count_collisions(seq)
        if collisions == 0:
            return -len(seq)
        else:
            # Penalty needs to be large enough to discourage invalid states, 
            # but allow moving through them if it leads to a better valid state.
            # Let's say each collision costs 100 in length-equivalent.
            return -len(seq) + 100 * collisions

    def greedy_init():
        """Generate a valid initial sequence using greedy with random shuffles."""
        best_seq = []
        # Try a few random orders for the greedy fill to get a good start
        for _ in range(5):
            shuffled = CANDIDATES[:]
            random.shuffle(shuffled)
            seq = []
            for x in shuffled:
                # Try to insert x into seq
                # Since we want sorted, we just check if adding x to the set and sorting is valid
                # Optimization: Just check validity of sorted(seq + [x])
                # This is O(N^2) per check.
                # We can speed up by checking only new subarrays involving x?
                # For now, simple check.
                new_seq = sorted(seq + [x])
                if count_collisions(new_seq) == 0:
                    seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq
        return best_seq

    # Simulated Annealing
    current_seq = greedy_init()
    current_energy = get_energy(current_seq)
    
    best_seq = current_seq[:]
    best_energy = current_energy
    
    # Parameters
    T0 = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iters = 5000
    start_time = time.time()
    
    T = T0
    for i in range(max_iters):
        if time.time() - start_time > 50: # Leave 10s buffer
            break
            
        # Generate neighbor
        r = random.random()
        if r < 0.4:
            # Add
            unused = [x for x in CANDIDATES if x not in current_seq]
            if unused:
                x = random.choice(unused)
                new_seq = sorted(current_seq + [x])
            else:
                new_seq = current_seq[:]
        elif r < 0.8:
            # Remove
            if current_seq:
                x = random.choice(current_seq)
                new_seq = [v for v in current_seq if v != x]
            else:
                new_seq = current_seq[:]
        else:
            # Swap
            if current_seq:
                x = random.choice(current_seq)
                unused = [v for v in CANDIDATES if v not in current_seq]
                if unused:
                    y = random.choice(unused)
                    new_seq = sorted([v for v in current_seq if v != x] + [y])
                else:
                    new_seq = current_seq[:]
            else:
                new_seq = current_seq[:]
        
        new_energy = get_energy(new_seq)
        delta = new_energy - current_energy
        
        # Metropolis Acceptance
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_seq = new_seq
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_seq = current_seq[:]
                best_energy = current_energy
        
        T *= cooling_rate

    return best_seq
