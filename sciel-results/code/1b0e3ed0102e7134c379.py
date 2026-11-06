import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    random.seed(42)
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    def compute_reps_full(A_list):
        """Compute representation counts for all targets [128,256].
        A_list is a sorted list of set elements.
        Returns reps array of length 129."""
        reps = np.zeros(num_targets, dtype=np.int32)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            for j in range(i, m):
                s = ai + A_list[j]
                if target_low <= s <= target_high:
                    reps[s - target_low] += 1
        return reps
    
    def get_peak(reps):
        """Get peak multiplicity, return inf if any target uncovered."""
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    def repair_coverage(A_list):
        """Try to fix coverage by adding elements.
        Returns None if failed to fix within a few steps, else new A_list."""
        A = list(A_list)
        reps = compute_reps_full(A)
        
        # We try to fix zeros
        for _ in range(50):
            missing = np.where(reps == 0)[0]
            if len(missing) == 0:
                return A
            
            # Pick a random missing target
            target_idx = random.choice(missing)
            target_sum = target_idx + target_low
            
            # Try to find a pair (a, b) such that a+b = target_sum
            # We can try to add a new element x such that x + a = target_sum for some a in A
            # Or x + x = target_sum
            candidates = set()
            for a in A:
                x = target_sum - a
                if 0 <= x <= n:
                    candidates.add(x)
            if target_sum % 2 == 0:
                x = target_sum // 2
                if 0 <= x <= n:
                    candidates.add(x)
            
            if not candidates:
                return None
            
            # If x is already in A, the pair (x, a) already exists, so sum would be covered.
            # Thus, for a missing sum, no candidate x can be in A.
            # So we can just pick any candidate.
            x = random.choice(list(candidates))
            
            A.append(x)
            A.sort()
            
            # Update reps
            reps = compute_reps_full(A)
            
        return A
    
    def local_search(A_list, max_iters=2000):
        """Local search to minimize peak."""
        A = list(A_list)
        reps = compute_reps_full(A)
        peak = get_peak(reps)
        if peak == float('inf'):
            return None, None, None
        
        best_A = list(A)
        best_peak = peak
        
        T = 2.0
        cooling = 0.99
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random swap
            old_val = random.choice(A)
            new_val = random.randint(0, n)
            
            if new_val == old_val or new_val in A:
                continue
            
            # Calculate new reps
            # Remove old_val contributions
            new_reps = reps.copy()
            for a in A:
                s = old_val + a
                if target_low <= s <= target_high:
                    new_reps[s - target_low] -= 1
            
            # Add new_val contributions
            # We don't need to sort A_temp for the sum calculation, just iterate
            # But we need to replace old_val with new_val in the list for the next iteration
            # We can create a new list
            A_temp = A[:]
            idx = A.index(old_val)
            A_temp[idx] = new_val
            
            for a in A_temp:
                s = new_val + a
                if target_low <= s <= target_high:
                    new_reps[s - target_low] += 1
            
            new_peak = get_peak(new_reps)
            
            # Acceptance
            if new_peak < peak:
                accept = True
            elif new_peak == peak:
                accept = random.random() < 0.1
            else:
                # Simulated Annealing
                delta = new_peak - peak
                if delta < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / T)
            
            if accept:
                A = A_temp
                reps = new_reps
                peak = new_peak
                
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            
            T *= cooling
        
        return best_A, best_peak, reps
    
    best_A = None
    best_peak = float('inf')
    
    # Try various sizes, starting small
    for size in [16, 18, 20, 22, 25, 30]:
        if time.time() > deadline:
            break
        
        max_restarts = 20 if size < 25 else 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            # Random start
            A = sorted(random.sample(range(0, n + 1), size))
            
            # Repair
            A_repaired = repair_coverage(A)
            if A_repaired is None:
                # Try a structured start?
                # E.g., step size 2
                step = max(1, int(n / size))
                A_struct = list(range(0, n + 1, step))
                if len(A_struct) > size:
                    A_struct = random.sample(A_struct, size)
                A_repaired = repair_coverage(A_struct)
                
            if A_repaired is None:
                continue
                
            A = A_repaired
            
            # Local Search
            res_A, res_peak, _ = local_search(A)
            
            if res_A is not None and res_peak < best_peak:
                best_peak = res_peak
                best_A = res_A
                
            if best_peak <= 2: # Can't do better than 2 if not all 1s? 
                # Actually 1 is possible? 
                # If peak is 1, it's a perfect code.
                # Break early if we find a very good solution
                if best_peak == 1:
                    return best_A

    # Final polish on the best found
    if best_A is not None:
        # Run local search again with more iterations on the best found
        res_A, res_peak, _ = local_search(best_A, max_iters=5000)
        if res_A is not None and res_peak < best_peak:
            best_peak = res_peak
            best_A = res_A

    if best_A is None:
        # Fallback: a guaranteed valid set (might have high peak)
        return list(range(64, 129))
    
    return best_A
