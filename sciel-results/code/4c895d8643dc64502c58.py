"""Improved program: maximize cardinality of subset of [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a large subset of [1,100] with no subset summing to exactly 200.
    
    Strategy:
    1. Start with the greedy seed (take large elements first, avoiding 200).
    2. Use local search (swap, add, remove) to try to increase cardinality.
    3. Use simulated annealing to escape local optima.
    4. Return the best solution found.
    """
    deadline = time.time() + 80  # Safety margin before 90s kill
    TARGET = 200
    UNIVERSE = list(range(1, 101))
    
    # Helper: check if a set has a subset summing to TARGET
    def has_target_sum(s):
        """Return True if any subset of s sums to TARGET."""
        reachable = {0}
        for x in s:
            new = {v + x for v in reachable if v + x <= TARGET}
            if TARGET in new:
                return True
            reachable |= new
        return TARGET in reachable
    
    # Helper: get reachable sums up to TARGET
    def get_reachable(s):
        reachable = {0}
        for x in s:
            new = {v + x for v in reachable if v + x <= TARGET}
            reachable |= new
        return reachable
    
    # Step 1: Greedy seed - take elements from largest to smallest
    selected = set()
    reachable = {0}
    for x in sorted(UNIVERSE, reverse=True):
        new_reach = {s + x for s in reachable if s + x <= TARGET}
        if TARGET not in new_reach:
            selected.add(x)
            reachable |= new_reach
    
    best = set(selected)
    best_size = len(best)
    
    # Step 2: Local search with simulated annealing
    random.seed(42)
    
    def energy(s):
        """Negative of cardinality (we want to maximize)."""
        return -len(s)
    
    current = set(selected)
    current_energy = energy(current)
    
    # Precompute for efficiency
    def try_add(s, x):
        """Try adding x to s, return new set or None if invalid."""
        if x in s:
            return None
        new_reach = {v + x for v in get_reachable(s) if v + x <= TARGET}
        if TARGET in new_reach:
            return None
        return s | {x}
    
    def try_remove(s, x):
        """Remove x from s, return new set."""
        if x not in s:
            return None
        return s - {x}
    
    # Simulated annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    iterations = 0
    while time.time() < deadline and T > min_T:
        iterations += 1
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random element
            candidates = [x for x in UNIVERSE if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = try_add(current, x)
            if new_set is None:
                continue
            delta_e = energy(new_set) - current_energy
            if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / T):
                current = new_set
                current_energy = energy(current)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
                    
        elif move_type < 0.8:
            # Try to remove a random element
            if not current:
                continue
            x = random.choice(list(current))
            new_set = try_remove(current, x)
            if new_set is None:
                continue
            delta_e = energy(new_set) - current_energy
            if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / T):
                current = new_set
                current_energy = energy(current)
                
        else:
            # Try swap: remove one, add another
            if not current:
                continue
            x_remove = random.choice(list(current))
            candidates = [x for x in UNIVERSE if x not in current]
            if not candidates:
                continue
            x_add = random.choice(candidates)
            
            new_set = current - {x_remove}
            # Try to add x_add
            new_reach = {v + x_add for v in get_reachable(new_set) if v + x_add <= TARGET}
            if TARGET not in new_reach:
                new_set = new_set | {x_add}
                delta_e = energy(new_set) - current_energy
                if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / T):
                    current = new_set
                    current_energy = energy(current)
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(best)
        
        T *= cooling
        
        # Occasionally restart from best
        if iterations % 5000 == 0 and len(best) > len(current):
            current = set(best)
            current_energy = energy(current)
    
    # Step 3: Final greedy improvement pass on the best solution
    # Try to add any element that doesn't create a target sum
    for x in sorted(UNIVERSE, reverse=True):
        if x not in best:
            new_reach = {v + x for v in get_reachable(best) if v + x <= TARGET}
            if TARGET not in new_reach:
                best = best | {x}
    
    # Ensure nonempty
    if not best:
        best = {100}
    
    return list(best)
# EVOLVE-BLOCK-END
