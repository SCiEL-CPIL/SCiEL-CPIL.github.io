import random
import math
import time

def construct():
    # Target differences
    targets = set(range(1, 31))
    
    # Energy weights
    W_COLLISION = 1000.0
    W_COVERAGE = 500.0
    W_DIAMETER = 10.0
    W_WASTED = 0.1
    
    def calculate_energy(A):
        if not A or A[0] != 0:
            return float('inf')
        
        diffs = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                if d > 0:
                    diffs.append(d)
        
        diff_counts = {}
        for d in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = 0
        for count in diff_counts.values():
            if count > 1:
                collisions += (count - 1)
        
        missing = 0
        for t in targets:
            if t not in diff_counts:
                missing += 1
        
        diameter = A[-1]
        wasted = sum(d - 30 for d in diffs if d > 30)
        
        size_penalty = max(0, 9 - len(A)) * 1000.0 + max(0, len(A) - 12) * 50.0
        
        energy = (W_COLLISION * collisions) + (W_COVERAGE * missing) + (W_DIAMETER * diameter) + (W_WASTED * wasted) + size_penalty
        return energy

    def is_valid(A):
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

    # Initial solution: Random small set
    A = [0]
    for _ in range(8):
        A.append(random.randint(1, 50))
    A = sorted(set(A))
    if A[0] != 0:
        A.insert(0, 0)
    
    current_energy = calculate_energy(A)
    
    best_solution = None
    best_energy = float('inf')
    
    if is_valid(A):
        best_solution = A[:]
        best_energy = current_energy
    
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.999
    max_iters = 20000
    start_time = time.time()
    timeout = 110.0
    
    T = T_start
    no_improve_count = 0
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        new_A = A[:]
        move_type = random.random()
        
        if move_type < 0.8:
            if len(new_A) > 1:
                idx = random.randint(1, len(new_A) - 1)
                if random.random() < 0.9:
                    delta = random.randint(-5, 5)
                else:
                    delta = random.randint(-50, 50)
                new_val = new_A[idx] + delta
                if new_val > 0:
                    if new_val not in new_A:
                        new_A[idx] = new_val
                        new_A.sort()
                    else:
                        continue
        elif move_type < 0.9:
            if len(new_A) < 15:
                new_val = random.randint(1, 200)
                if new_val not in new_A:
                    new_A.append(new_val)
                    new_A.sort()
                else:
                    continue
        else:
            if len(new_A) > 9:
                idx = random.randint(1, len(new_A) - 1)
                new_A.pop(idx)
        
        new_energy = calculate_energy(new_A)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and random.random() < math.exp(-delta_e / T)):
            A = new_A
            current_energy = new_energy
            
            if is_valid(A):
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = A[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling_rate
        
        if no_improve_count > 5000:
            if best_solution:
                A = best_solution[:]
                current_energy = best_energy
            else:
                A = [0]
                for _ in range(8):
                    A.append(random.randint(1, 50))
                A = sorted(set(A))
                if A[0] != 0:
                    A.insert(0, 0)
                current_energy = calculate_energy(A)
            T = T_start * 0.5
            no_improve_count = 0
    
    if best_solution:
        return best_solution
    else:
        # Fallback
        if is_valid(A):
            return A
        else:
            # Try to fix A locally
            # This is a best-effort fallback
            return A

# Run construct
if __name__ == "__main__":
    res = construct()
    print(res)
