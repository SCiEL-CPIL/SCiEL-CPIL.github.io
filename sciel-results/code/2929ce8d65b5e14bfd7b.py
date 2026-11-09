"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 130
    prescribed = [1, 2, 5, 11]
    N = 284
    
    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Greedy initial solution
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    for x in range(12, N + 1):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    best_size = len(a)
    best_a = list(a)
    
    # SA with incremental O(k) conflict updates
    def build_diff_counts(arr):
        max_diff = N
        diff_counts = [0] * (max_diff + 1)
        num_conflicts = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                diff_counts[d] += 1
                if diff_counts[d] > 1:
                    num_conflicts += 1
        return diff_counts, num_conflicts
    
    def apply_move(arr, diff_counts, num_conflicts, idx, new_val):
        """Apply a move: set arr[idx] = new_val. Returns (new_num_conflicts) and modifies diff_counts in place.
        Caller must revert if move is rejected."""
        n = len(arr)
        old_val = arr[idx]
        
        # Remove old differences involving idx
        for j in range(n):
            if j == idx:
                continue
            if arr[j] > old_val:
                d = arr[j] - old_val
            else:
                d = old_val - arr[j]
            diff_counts[d] -= 1
            if diff_counts[d] == 1:
                num_conflicts -= 1
        
        # Add new differences involving idx
        for j in range(n):
            if j == idx:
                continue
            if arr[j] > new_val:
                d = arr[j] - new_val
            else:
                d = new_val - arr[j]
            diff_counts[d] += 1
            if diff_counts[d] == 2:
                num_conflicts += 1
        
        arr[idx] = new_val
        return num_conflicts
    
    def revert_move(arr, diff_counts, num_conflicts, idx, new_val, old_val):
        """Revert a move: set arr[idx] back to old_val."""
        n = len(arr)
        
        # Remove new differences involving idx (which are now with new_val)
        for j in range(n):
            if j == idx:
                continue
            if arr[j] > new_val:
                d = arr[j] - new_val
            else:
                d = new_val - arr[j]
            diff_counts[d] -= 1
            if diff_counts[d] == 1:
                num_conflicts -= 1
        
        # Add old differences involving idx
        for j in range(n):
            if j == idx:
                continue
            if arr[j] > old_val:
                d = arr[j] - old_val
            else:
                d = old_val - arr[j]
            diff_counts[d] += 1
            if diff_counts[d] == 2:
                num_conflicts += 1
        
        arr[idx] = old_val
        return num_conflicts
    
    random.seed(42)
    
    # Try different target sizes
    for target_k in [min(20, N), best_size + 1, best_size]:
        if time.time() > deadline - 15:
            break
        if target_k <= len(prescribed):
            continue
        
        # Initialize
        remaining = target_k - len(prescribed)
        greedy_extra = [x for x in a if x not in prescribed]
        chosen_extra = list(greedy_extra[:remaining]) if len(greedy_extra) >= remaining else list(greedy_extra)
        if len(chosen_extra) < remaining:
            available = [x for x in range(1, N + 1) if x not in prescribed and x not in chosen_extra]
            random.shuffle(available)
            chosen_extra += available[:remaining - len(chosen_extra)]
        
        sol = list(prescribed) + list(chosen_extra)
        
        diff_counts, num_conflicts = build_diff_counts(sol)
        
        if num_conflicts == 0:
            if target_k > best_size:
                best_size = target_k
                best_a = list(sol)
            break
        
        # SA parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.9999
        max_iters = 200000
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            if num_conflicts == 0:
                if target_k > best_size:
                    best_size = target_k
                    best_a = list(sol)
                break
            
            # Pick random variable index (4 to target_k-1)
            idx = random.randint(4, target_k - 1)
            old_val = sol[idx]
            
            # Choose new value
            new_val = random.randint(1, N)
            
            # Check uniqueness
            valid = True
            for i in range(target_k):
                if i != idx and sol[i] == new_val:
                    valid = False
                    break
            if not valid:
                continue
            
            # Apply move
            new_conflicts = apply_move(sol, diff_counts, num_conflicts, idx, new_val)
            delta = new_conflicts - num_conflicts
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                num_conflicts = new_conflicts
            else:
                num_conflicts = revert_move(sol, diff_counts, num_conflicts, idx, new_val, old_val)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if num_conflicts == 0 and target_k > best_size:
            best_size = target_k
            best_a = list(sol)
    
    # Verify and return
    if is_valid_sidon(best_a):
        return sorted(best_a)
    if is_valid_sidon(a):
        return sorted(a)
    return sorted(prescribed)
# EVOLVE-BLOCK-END
