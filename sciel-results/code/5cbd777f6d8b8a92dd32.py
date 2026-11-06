"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Use simulated annealing on the set of selected indices to find a strictly
    increasing sequence of integers in [1,256] such that all nonempty consecutive
    block sums are distinct. Maximize sequence length.
    """
    
    def count_conflicts(seq):
        """Return the number of duplicate consecutive block sums."""
        n = len(seq)
        if n <= 1:
            return 0
        sums = set()
        conflicts = 0
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    conflicts += 1
                else:
                    sums.add(s)
        return conflicts

    def is_valid(seq):
        """Check if sequence is strictly increasing and all consecutive block sums are distinct."""
        if len(seq) == 0:
            return True
        if len(seq) == 1:
            return 1 <= seq[0] <= 256
        
        # Check strictly increasing and bounds
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        if seq[0] < 1 or seq[-1] > 256:
            return False
            
        return count_conflicts(seq) == 0

    def greedy_construct():
        values = []
        used_sums = set()
        for x in range(1, 257):
            new_sums = [x]
            t = x
            for a in reversed(values):
                t += a
                new_sums.append(t)
            if not any(s in used_sums for s in new_sums):
                values.append(x)
                used_sums.update(new_sums)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    def simulated_annealing(initial_seq, time_limit=45.0):
        current = list(initial_seq)
        current_conflicts = count_conflicts(current)
        best = list(current)
        best_conflicts = current_conflicts
        best_len = len(current)
        
        temp = 10.0
        cooling = 0.9995
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to insert
                if len(current) == 0:
                    new_val = random.randint(1, 256)
                    new_seq = [new_val]
                else:
                    pos = random.randint(0, len(current))
                    low = current[pos-1] + 1 if pos > 0 else 1
                    high = current[pos] - 1 if pos < len(current) else 256
                    if low <= high:
                        new_val = random.randint(low, high)
                        new_seq = current[:pos] + [new_val] + current[pos:]
                    else:
                        continue
            elif move_type < 0.7:
                # Try to remove
                if len(current) <= 1:
                    continue
                pos = random.randint(0, len(current) - 1)
                new_seq = current[:pos] + current[pos+1:]
            elif move_type < 0.9:
                # Try to swap value at pos with a random unused value
                if len(current) == 0:
                    continue
                pos = random.randint(0, len(current) - 1)
                old_val = current[pos]
                low = current[pos-1] + 1 if pos > 0 else 1
                high = current[pos+1] - 1 if pos < len(current) - 1 else 256
                if low < high:
                    candidates = [v for v in range(low, high+1) if v != old_val]
                    if candidates:
                        new_val = random.choice(candidates)
                        new_seq = current[:pos] + [new_val] + current[pos+1:]
                    else:
                        continue
                else:
                    continue
            else:
                # Try to move a value (swap positions logic is tricky for increasing seq, 
                # so we simulate by removing and re-inserting)
                if len(current) < 2:
                    continue
                pos = random.randint(0, len(current) - 1)
                val = current[pos]
                new_seq = current[:pos] + current[pos+1:]
                # Insert back at a different position
                if len(new_seq) == 0:
                    new_seq = [val]
                else:
                    new_pos = random.randint(0, len(new_seq))
                    low = new_seq[new_pos-1] + 1 if new_pos > 0 else 1
                    high = new_seq[new_pos] - 1 if new_pos < len(new_seq) else 256
                    if low <= val <= high:
                        new_seq = new_seq[:new_pos] + [val] + new_seq[new_pos:]
                    else:
                        continue

            new_conflicts = count_conflicts(new_seq)
            new_len = len(new_seq)
            
            # Energy: Minimize Conflicts, then Maximize Length
            curr_energy = current_conflicts * 1000 - len(current)
            new_energy = new_conflicts * 1000 - new_len
            
            delta_e = new_energy - curr_energy
            
            if delta_e < 0:
                current = new_seq
                current_conflicts = new_conflicts
                if new_conflicts == 0 and new_len > best_len:
                    best = new_seq
                    best_len = new_len
                    best_conflicts = new_conflicts
            else:
                if temp > 0.01:
                    # Avoid overflow/underflow in exp
                    if delta_e / temp < -500:
                        prob = 0.0
                    elif delta_e / temp > 500:
                        prob = 1.0
                    else:
                        prob = np.exp(-delta_e / temp)
                    if random.random() < prob:
                        current = new_seq
                        current_conflicts = new_conflicts
            
            temp *= cooling
            
        return best

    random.seed(42)
    
    # Run SA starting from greedy
    res1 = simulated_annealing(best_seq, time_limit=20.0)
    if len(res1) > len(best_seq):
        best_seq = res1

    # Run SA from a random valid start
    def random_valid_start():
        for _ in range(100):
            length = random.randint(5, 10)
            candidates = random.sample(range(1, 257), length)
            candidates.sort()
            if is_valid(candidates):
                return candidates
        return []

    start2 = random_valid_start()
    if start2:
        res2 = simulated_annealing(start2, time_limit=20.0)
        if len(res2) > len(best_seq):
            best_seq = res2

    return best_seq

# EVOLVE-BLOCK-END
