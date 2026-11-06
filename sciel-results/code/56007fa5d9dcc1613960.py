import random
import math
import time

def construct():
    # Target differences
    targets = set(range(1, 31))
    
    # Energy weights
    W_COLLISION = 100.0
    W_COVERAGE = 50.0
    W_DIAMETER = 1.0
    W_SIZE = 10.0
    
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
        # Count occurrences of each difference
        diff_counts = {}
        for d in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = 0
        for count in diff_counts.values():
            if count > 1:
                # Penalize extra occurrences
                collisions += (count - 1)
        
        # Coverage penalty: all d in 1..30 must be present
        missing = 0
        for t in targets:
            if t not in diff_counts:
                missing += 1
        
        # Diameter penalty
        diameter = A[-1]
        
        # Size penalty (prefer smaller sets, but need enough to cover 30 diffs)
        # Max diffs for size k is k*(k-1)/2. Need >= 30.
        # k=8 -> 28 diffs (not enough)
        # k=9 -> 36 diffs (enough)
        # So size should be at least 9.
        size_penalty = max(0, 9 - len(A)) * 100.0 + max(0, len(A) - 12) * 10.0
        
        energy = (W_COLLISION * collisions) + (W_COVERAGE * missing) + (W_DIAMETER * diameter) + size_penalty
        return energy

    def is_valid(A):
        """
        Check if A is a valid Sidon set covering 1..30.
        """
        if not A or A[0] != 0:
            return False
        if len(A) < 9: # 8 elements give max 28 diffs, need 30
            return False
        
        diffs = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(A[j] - A[i])
        
        # Check distinctness
        if len(diffs) != len(set(diffs)):
            return False
            
        # Check coverage
        diff_set = set(diffs)
        for t in targets:
            if t not in diff_set:
                return False
        return True

    # Simulated Annealing
    best_solution = None
    best_energy = float('inf')
    
    # Initial solution: Greedy-ish random
    # Start with 0 and a few random small numbers
    A = [0]
    # Add random elements to get size ~9
    for _ in range(8):
        A.append(random.randint(1, 50))
    A = sorted(set(A))
    # Ensure 0 is first
    if A[0] != 0:
        A.insert(0, 0)
    
    current_energy = calculate_energy(A)
    
    # Annealing parameters
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iters = 50000
    start_time = time.time()
    timeout = 100.0 # Leave some buffer for 120s limit
    
    T = T_start
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        # Generate a neighbor
        new_A = A[:]
        move_type = random.random()
        
        if move_type < 0.7:
            # Adjust a value
            if len(new_A) > 1:
                idx = random.randint(1, len(new_A) - 1)
                # Change by a small amount
                delta = random.randint(-10, 10)
                new_val = new_A[idx] + delta
                if new_val > 0: # Must be positive and distinct
                    # Check if new_val collides with existing
                    if new_val not in new_A:
                        new_A[idx] = new_val
                        new_A.sort()
                        if new_A[0] != 0:
                            new_A[0] = 0 # This shouldn't happen if we only change idx>0
                    else:
                        # If collision, try a different random value
                        new_val = random.randint(1, 200)
                        if new_val not in new_A:
                            new_A[idx] = new_val
                            new_A.sort()
                        else:
                            continue # Skip this move if we can't place it easily
        elif move_type < 0.85:
            # Insert a new element
            if len(new_A) < 15:
                new_val = random.randint(1, 200)
                if new_val not in new_A:
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
        
        # Acceptance probability
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            A = new_A
            current_energy = new_energy
            
            # Check if this is a valid solution
            if is_valid(A):
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = A[:]
        
        # Cool down
        T *= cooling_rate
    
    if best_solution:
        return best_solution
    else:
        # Fallback: return the current best if valid, or a known valid set if found during search
        # If no valid solution found, we might return the current A if it's valid, 
        # otherwise we need to ensure we return *something* valid.
        # Given the difficulty, we should probably run multiple restarts or a more robust search.
        # Let's add a simple fallback: a known small Sidon set for 1..30?
        # Actually, let's just return the best found even if not perfectly valid? No, contract requires valid.
        # If SA fails to find a valid one, we can try a deterministic construction as a last resort.
        
        # Deterministic Fallback: 
        # A simple way to get a Sidon set is using powers of 2, but diameter grows fast.
        # Or use the greedy algorithm from the seed but improved.
        
        # Let's try a quick greedy construction as fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            # Try to add one element
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                # Try to add two elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

# Run construct
if __name__ == "__main__":
    res = construct()
    print(res)
