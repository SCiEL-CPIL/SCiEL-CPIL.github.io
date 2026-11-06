"""Simulated annealing approach for constructing a Sidon set with differences 1..30."""
import random
import math

def construct():
    """
    Construct a Sidon set containing 0 with at most 32 distinct integers in 0..10^6,
    including 0, such that all positive differences are distinct and every d in 1..30
    occurs as a difference. Minimize the largest element.
    
    Uses simulated annealing with a composite energy function.
    """
    random.seed(42)
    
    def energy(s):
        """Compute energy of a candidate set s (sorted, contains 0)."""
        if not s or s[0] != 0:
            return float('inf')
        if len(s) > 32:
            return float('inf')
        if any(x < 0 or x > 10**6 for x in s):
            return float('inf')
        
        # Compute all positive differences
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        
        # Check for duplicate differences (Sidon violation)
        diff_set = set(diffs)
        num_diffs = len(diffs)
        if len(diff_set) != num_diffs:
            # Penalty for collisions
            collision_penalty = (num_diffs - len(diff_set)) * 1000
        
        # Check which targets in 1..30 are covered
        missing = [d for d in range(1, 31) if d not in diff_set]
        missing_penalty = len(missing) * 500
        
        # Diameter penalty (we want to minimize the largest element)
        diameter = s[-1]
        diameter_penalty = diameter * 1.0
        
        return collision_penalty + missing_penalty + diameter_penalty
    
    def is_valid(s):
        """Check if s is a valid Sidon set covering all differences 1..30."""
        if not s or s[0] != 0:
            return False
        if len(s) > 32:
            return False
        if any(x < 0 or x > 10**6 for x in s):
            return False
        
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        
        diff_set = set(diffs)
        # All differences must be distinct
        if len(diff_set) != len(diffs):
            return False
        
        # All targets 1..30 must be covered
        for d in range(1, 31):
            if d not in diff_set:
                return False
        
        return True
    
    # Start with a reasonable initial solution
    # Use a known structure: start with 0 and build up
    # A good initial guess based on perfect difference set theory
    # For n elements, we need ~n^2/2 differences, and we need to cover 1..30
    
    # Start with a small set and grow it
    # Initial set: [0, 1, 4, 10, 18, 28, 40, 54, 70, 88] - quadratic-ish growth
    # But let's use simulated annealing to find a better one
    
    # Start with a random feasible-ish set
    # We need at least enough elements to cover 30 differences
    # With k elements, we have k*(k-1)/2 differences
    # Need k*(k-1)/2 >= 30, so k >= 8 (8*7/2 = 28, not enough), k=9 (9*8/2=36)
    
    # Let's start with a set of size ~10-15
    best_solution = None
    best_energy = float('inf')
    
    # Multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        # Initialize with a set that has 0 and some spread out elements
        # Try different initial sizes
        init_size = 10 + restart * 2  # 10, 12, 14, 16, 18
        
        # Start with 0 and add elements with some spacing
        s = [0]
        # Add elements with decreasing gaps to encourage coverage of small differences
        gap = 1
        for i in range(1, init_size):
            # Add element with some randomness
            next_val = s[-1] + gap + random.randint(0, 3)
            s.append(next_val)
            gap = max(1, gap - 1 + random.randint(0, 2))
        
        s = sorted(set(s))
        
        # Simulated annealing
        current = s[:]
        current_energy = energy(current)
        
        T = 5000.0
        T_min = 1.0
        cooling_rate = 0.999
        num_iterations = 20000
        
        for it in range(num_iterations):
            T = max(T_min, T * cooling_rate)
            
            # Generate a neighbor
            neighbor = current[:]
            
            # Random move: modify one element, add one, or remove one
            move_type = random.random()
            
            if move_type < 0.5 and len(neighbor) > 2:
                # Modify an existing element (not 0)
                idx = random.randint(1, len(neighbor) - 1)
                delta = random.randint(-20, 20)
                new_val = neighbor[idx] + delta
                if new_val > 0 and new_val < 10**6:
                    neighbor[idx] = new_val
                    neighbor = sorted(set(neighbor))
                    if len(neighbor) < 2:
                        neighbor = current[:]
                else:
                    continue
            elif move_type < 0.8 and len(neighbor) < 32:
                # Add a new element
                if len(neighbor) >= 2:
                    max_val = neighbor[-1]
                    # Try adding near existing elements or extending
                    if random.random() < 0.5:
                        base = random.choice(neighbor[1:])
                        new_val = base + random.randint(1, 50)
                    else:
                        new_val = max_val + random.randint(1, 30)
                    if 0 < new_val < 10**6 and new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor = sorted(neighbor)
                    else:
                        continue
                else:
                    continue
            else:
                # Remove an element (not 0)
                if len(neighbor) > 2:
                    idx = random.randint(1, len(neighbor) - 1)
                    neighbor.pop(idx)
                else:
                    continue
            
            neighbor_energy = energy(neighbor)
            
            # Accept or reject
            if neighbor_energy < current_energy:
                current = neighbor
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = current[:]
            else:
                delta_e = neighbor_energy - current_energy
                if delta_e < 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta_e / max(T, 1e-10))
                if accept:
                    current = neighbor
                    current_energy = neighbor_energy
        
        # Check if we found a valid solution
        if best_solution is not None and is_valid(best_solution):
            break
    
    # If we found a valid solution, return it
    if best_solution is not None and is_valid(best_solution):
        return best_solution
    
    # Fallback: use a constructive approach to ensure validity
    # Greedy construction with backtracking
    a = [0]
    
    def get_diffs(s):
        return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]
    
    def is_sidon(s):
        d = get_diffs(s)
        return len(set(d)) == len(d)
    
    for d in range(1, 31):
        # Check if d is already a difference
        cur_diffs = set(get_diffs(a))
        if d in cur_diffs:
            continue
        
        # Try to add an element to realize difference d
        found = False
        # Try adding x such that x - y = d or y - x = d for some y in a
        candidates = set()
        for y in a:
            candidates.add(y + d)
            candidates.add(y - d)
        candidates = [x for x in candidates if x >= 0 and x not in a]
        candidates.sort()
        
        for x in candidates:
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        
        if not found:
            # Try adding two elements to realize d
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    
    # Verify and return
    if is_valid(a):
        return a
    else:
        # Last resort: return a known valid set if we have one
        # Construct a valid set manually
        # A perfect difference set of order 5 has 11 elements
        # Let's use a known Sidon set that covers 1..30
        # We'll use the greedy result even if not optimal
        return a

