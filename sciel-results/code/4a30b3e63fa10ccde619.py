"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    random.seed(42)
    
    # Precompute scaled values to use integers instead of Fractions
    # LCM of 1..22
    L = 232792560
    vals = [0] * 23
    for i in range(1, 23):
        vals[i] = L // i
        
    def _count_collisions(subset):
        """
        Returns the number of duplicate subset sums.
        0 means dissociated.
        """
        if not subset:
            return 0
        current_sums = {0}
        for x in subset:
            new_sums = set()
            vx = vals[x]
            for s in current_sums:
                new_sums.add(s)
                new_sums.add(s + vx)
            current_sums = new_sums
        return (1 << len(subset)) - len(current_sums)

    # Try to find a set of size K, starting from high K
    for K in range(12, 0, -1):
        # Simulated Annealing
        current_set = set(random.sample(range(1, 23), K))
        current_cost = _count_collisions(list(current_set))
        
        if current_cost == 0:
            return list(current_set)
            
        T = 10.0
        cooling_rate = 0.99
        min_T = 0.01
        steps = 0
        max_steps = 5000
        
        while T > min_T and steps < max_steps:
            # Pick a random element in set and random element outside
            in_elem = random.choice(list(current_set))
            while True:
                out_elem = random.randint(1, 22)
                if out_elem not in current_set:
                    break
            
            # Propose swap
            new_set = current_set.copy()
            new_set.remove(in_elem)
            new_set.add(out_elem)
            
            new_cost = _count_collisions(list(new_set))
            
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_set = new_set
                current_cost = new_cost
                if current_cost == 0:
                    return list(current_set)
            
            T *= cooling_rate
            steps += 1
            
    # Fallback
    return [1]

# EVOLVE-BLOCK-END
