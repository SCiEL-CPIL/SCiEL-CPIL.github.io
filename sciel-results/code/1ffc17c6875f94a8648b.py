import random
import math
import time

def construct():
    # Target differences
    targets = set(range(1, 31))
    
    def calculate_energy(A):
        """
        Calculate the energy of the set A.
        A is a sorted list of unique integers including 0.
        """
        if not A or A[0] != 0:
            return float('inf')
        
        # Calculate all positive differences
        diffs = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                if d > 0:
                    diffs.append(d)
        
        # Collision penalty: differences must be distinct
        diff_counts = {}
        for d in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = 0
        for count in diff_counts.values():
            if count > 1:
                collisions += (count - 1)
        
        # Coverage penalty: all d in 1..30 must be present
        missing = 0
        for t in targets:
            if t not in diff_counts:
                missing += 1
        
        # Diameter penalty
        diameter = A[-1]
        
        # Size penalty
        size_penalty = max(0, 9 - len(A)) * 100.0 + max(0, len(A) - 12) * 10.0
        
        energy = (100.0 * collisions) + (50.0 * missing) + (1.0 * diameter) + size_penalty
        return energy

    def is_valid(A):
        """
        Check if A is a valid Sidon set covering 1..30.
        """
        if not A or A[0] != 0:
            return False
        if len(A) < 9:
            return False
        
        diffs = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(A[j] - A[i])
        
        if len(diffs) != len(set(diffs)):
            return False
            
        diff_set = set(diffs)
        for t in targets:
            if t not in diff_set:
                return False
        return True

    best_solution = None
    best_energy = float('inf')
    
    # Multiple restarts with different initializations
    max_restarts = 5
    start_time = time.time()
    timeout = 110.0  # Leave buffer for 120s limit
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with different strategies
        if restart == 0:
            # Greedy-ish: start with small values
            A = [0, 1, 3, 7, 12, 18, 25, 33, 42]
        elif restart == 1:
            # Spread out more
            A = [0, 2, 5, 10, 17, 26, 37, 50, 65]
        elif restart == 2:
            # Random small
            A = [0]
            for _ in range(8):
                A.append(random.randint(1, 80))
            A = sorted(set(A))
        elif restart == 3:
            # Based on known Sidon patterns
            A = [0, 1, 4, 10, 18, 28, 40, 54, 70]
        else:
            # Wider spread
            A = [0, 3, 8, 15, 24, 35, 48, 63, 80]
        
        # Ensure unique and sorted
        A = sorted(set(A))
        if A[0] != 0:
            A.insert(0, 0)
        
        current_energy = calculate_energy(A)
        
        # Annealing parameters
        T_start = 50.0
        T_end = 0.01
        cooling_rate = 0.997
        max_iters = 30000
        
        T = T_start
        
        for i in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            # Generate a neighbor using continuous-style perturbations
            new_A = A[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Adjust a single element with Gaussian-like step
                if len(new_A) > 1:
                    idx = random.randint(1, len(new_A) - 1)
                    # Step size decreases with temperature
                    step_scale = max(1, int(20 * (T / T_start) + 1))
                    delta = int(random.gauss(0, step_scale))
                    new_val = new_A[idx] + delta
                    if new_val > 0 and new_val not in new_A:
                        new_A[idx] = new_val
                        new_A.sort()
                    else:
                        # Try a random small change
                        new_val = new_A[idx] + random.randint(-5, 5)
                        if new_val > 0 and new_val not in new_A:
                            new_A[idx] = new_val
                            new_A.sort()
                        else:
                            continue
            elif move_type < 0.7:
                # Adjust two elements
                if len(new_A) > 2:
                    idx1 = random.randint(1, len(new_A) - 1)
                    idx2 = random.randint(1, len(new_A) - 1)
                    if idx1 != idx2:
                        step_scale = max(1, int(15 * (T / T_start) + 1))
                        delta1 = int(random.gauss(0, step_scale))
                        delta2 = int(random.gauss(0, step_scale))
                        new_val1 = new_A[idx1] + delta1
                        new_val2 = new_A[idx2] + delta2
                        if new_val1 > 0 and new_val2 > 0 and new_val1 not in new_A and new_val2 not in new_A and new_val1 != new_val2:
                            new_A[idx1] = new_val1
                            new_A[idx2] = new_val2
                            new_A.sort()
                        else:
                            continue
            elif move_type < 0.85:
                # Insert a new element
                if len(new_A) < 14:
                    # Try to insert near existing elements or at a reasonable distance
                    if len(new_A) > 1:
                        base = random.choice(new_A[1:])
                        new_val = base + random.randint(-20, 20)
                    else:
                        new_val = random.randint(1, 100)
                    if new_val > 0 and new_val not in new_A:
                        new_A.append(new_val)
                        new_A.sort()
                    else:
                        continue
            else:
                # Remove an element
                if len(new_A) > 9:
                    idx = random.randint(1, len(new_A) - 1)
                    new_A.pop(idx)
            
            # Calculate new energy
            new_energy = calculate_energy(new_A)
            
            # Acceptance probability with Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or (T > T_end and random.random() < math.exp(-delta_e / T)):
                A = new_A
                current_energy = new_energy
                
                # Check if this is a valid solution
                if is_valid(A):
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = A[:]
            
            # Cool down
            T *= cooling_rate
        
        # If we found a valid solution, we can try to improve it further
        if best_solution:
            # Run a shorter SA to refine
            A = best_solution[:]
            current_energy = calculate_energy(A)
            T = 10.0
            cooling_rate = 0.995
            for i in range(10000):
                if time.time() - start_time > timeout:
                    break
                
                new_A = A[:]
                if len(new_A) > 1:
                    idx = random.randint(1, len(new_A) - 1)
                    delta = random.randint(-3, 3)
                    new_val = new_A[idx] + delta
                    if new_val > 0 and new_val not in new_A:
                        new_A[idx] = new_val
                        new_A.sort()
                
                new_energy = calculate_energy(new_A)
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    A = new_A
                    current_energy = new_energy
                    
                    if is_valid(A) and current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = A[:]
                
                T *= cooling_rate
    
    if best_solution:
        return best_solution
    else:
        # Fallback: deterministic greedy construction
        a = [0]
        def get_diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        
        for d in range(1, 31):
            if d in get_diffs(sorted(a)):
                continue
            found = False
            # Try to add one element
            candidates = set()
            for v in a:
                candidates.add(v + d)
                candidates.add(v - d)
            candidates = {x for x in candidates if x > 0 and x not in a}
            
            for x in sorted(candidates):
                trial = sorted(a + [x])
                diffs = get_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    found = True
                    break
            
            if not found:
                # Try to add two elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    diffs = get_diffs(trial)
                    if len(set(diffs)) == len(diffs):
                        a = trial
                        break
                    y += 1
        return a

# Run construct
if __name__ == "__main__":
    res = construct()
    print(res)
