import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129 targets
    
    def compute_multiplicities(A_set):
        """Compute multiplicity for each target value using FFT for speed."""
        # Create binary vector
        x = np.zeros(n + 1)
        for a in A_set:
            x[a] = 1.0
        # Autocorrelation via FFT
        size = 2 * (n + 1)
        X = np.fft.fft(x, size)
        ac = np.real(np.fft.ifft(X * np.conj(X)))
        ac = np.round(ac).astype(int)
        # Extract target range
        mult = ac[target_lo:target_hi+1]
        return mult
    
    def objective(A_set):
        """Return max multiplicity, or infinity if invalid."""
        mult = compute_multiplicities(A_set)
        if np.any(mult == 0):
            return float('inf')
        return int(np.max(mult))
    
    def is_valid(A_set):
        mult = compute_multiplicities(A_set)
        return np.all(mult > 0)
    
    # Initial solution
    best_A = set(range(64, 129))
    best_obj = objective(best_A)
    
    # Try structured initial solutions
    for size in range(16, 35):
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            A_set = set(random.sample(range(0, n + 1), size))
            obj = objective(A_set)
            if obj < best_obj:
                best_obj = obj
                best_A = set(A_set)
    
    # Simulated annealing with FFT-based objective
    current_A = set(best_A)
    current_obj = best_obj
    best_obj_sa = best_obj
    best_A_sa = set(best_A)
    
    T = 5.0
    cooling = 0.9998
    min_T = 0.01
    
    step_count = 0
    while time.time() < deadline and T > min_T:
        step_count += 1
        
        # Generate a neighbor
        op = random.random()
        A_list = list(current_A)
        
        if op < 0.25 and len(current_A) > 16:
            # Remove an element
            idx = random.randint(0, len(A_list) - 1)
            new_A = set(A_list[:idx] + A_list[idx+1:])
        elif op < 0.5:
            # Add an element
            candidates = [x for x in range(0, n + 1) if x not in current_A]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_A = set(current_A)
            new_A.add(x)
        elif op < 0.75:
            # Swap an element
            idx = random.randint(0, len(A_list) - 1)
            old_x = A_list[idx]
            candidates = [x for x in range(0, n + 1) if x not in current_A]
            if not candidates:
                continue
            new_x = random.choice(candidates)
            new_A = set(current_A)
            new_A.discard(old_x)
            new_A.add(new_x)
        else:
            # Move an element to a random new position
            idx = random.randint(0, len(A_list) - 1)
            old_x = A_list[idx]
            candidates = [x for x in range(0, n + 1) if x not in current_A]
            if not candidates:
                continue
            new_x = random.choice(candidates)
            new_A = set(current_A)
            new_A.discard(old_x)
            new_A.add(new_x)
        
        new_obj = objective(new_A)
        
        # Acceptance criterion
        if new_obj <= current_obj:
            current_A = new_A
            current_obj = new_obj
            if new_obj < best_obj_sa:
                best_obj_sa = new_obj
                best_A_sa = set(new_A)
        else:
            # Accept with probability exp(-(new_obj - current_obj) / T)
            delta = new_obj - current_obj
            if delta < 100:  # avoid overflow
                if random.random() < np.exp(-delta / T):
                    current_A = new_A
                    current_obj = new_obj
        
        T *= cooling
    
    # Final polish: try to reduce max multiplicity with targeted moves
    if time.time() < deadline:
        polish_deadline = min(deadline, time.time() + 5.0)
        A_list = list(best_A_sa)
        for _ in range(2000):
            if time.time() > polish_deadline:
                break
            # Find the target with max multiplicity
            mult = compute_multiplicities(best_A_sa)
            max_mult = int(np.max(mult))
            if max_mult <= 2:
                break
            # Try removing an element that contributes to high-multiplicity targets
            for idx in range(len(A_list)):
                x = A_list[idx]
                new_A = set(best_A_sa)
                new_A.discard(x)
                if len(new_A) < 16:
                    continue
                obj = objective(new_A)
                if obj < best_obj_sa:
                    best_obj_sa = obj
                    best_A_sa = new_A
                    A_list = list(best_A_sa)
                    break
    
    return sorted(best_A_sa)
