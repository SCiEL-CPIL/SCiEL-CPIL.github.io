import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for 60s limit
    
    n = 128
    target_start = 128
    target_end = 256
    num_targets = target_end - target_start + 1  # 129
    
    # A is a set of integers in [0, 128]
    # We represent it as a boolean list for fast lookup and sum calculation
    # mult[s] = number of pairs {a,b} in A such that a+b=s
    
    # Helper to compute multiplicities from scratch
    def compute_mults(A_set):
        mults = np.zeros(2 * n + 1, dtype=np.int32)
        A_list = sorted(A_set)
        L = len(A_list)
        for i in range(L):
            a = A_list[i]
            for j in range(i, L):
                b = A_list[j]
                s = a + b
                if target_start <= s <= target_end:
                    mults[s] += 1
        return mults
    
    # Helper to compute smooth objective (log-sum-exp based)
    def smooth_objective(mults, beta=2.0):
        """
        Smooth surrogate for min-max: use log-sum-exp as a smooth max approximation.
        Also incorporate variance as a secondary term.
        """
        target_mults = mults[target_start:target_end+1].astype(np.float64)
        # Log-sum-exp for smooth max
        max_val = np.max(target_mults)
        if max_val == 0:
            return float('inf')
        lse = max_val + np.log(np.sum(np.exp(target_mults - max_val)))
        # Variance term
        mean = np.mean(target_mults)
        var = np.var(target_mults)
        return lse + 0.01 * var
    
    # Helper to check coverage
    def is_covered(mults):
        return np.all(mults[target_start:target_end+1] > 0)
    
    # Helper to get peak
    def get_peak(mults):
        return int(np.max(mults[target_start:target_end+1]))
    
    # Helper to get variance
    def get_var(mults):
        tm = mults[target_start:target_end+1]
        return float(np.var(tm))
    
    random.seed(42)
    np.random.seed(42)
    
    best_A = set()
    best_peak = float('inf')
    best_var = float('inf')
    best_smooth = float('inf')
    
    # Greedy initialization with structured start
    # Start with a quasi-random spread to ensure good coverage potential
    A = set()
    
    # First, add elements that help cover the target range
    # Use a greedy approach to build initial set
    mults = np.zeros(2 * n + 1, dtype=np.int32)
    
    # Add a base set of elements spread across the range
    base_elements = list(range(0, n+1, 3))  # Every 3rd number
    random.shuffle(base_elements)
    
    # Select initial elements greedily
    for x in base_elements:
        if time.time() > start_time + 3:
            break
        A.add(x)
        # Update mults incrementally
        for y in A:
            if y == x:
                s = x + x
                if target_start <= s <= target_end:
                    mults[s] += 1
            elif y > x:
                s = x + y
                if target_start <= s <= target_end:
                    mults[s] += 1
    
    # Check coverage and fill gaps
    missing = [s for s in range(target_start, target_end+1) if mults[s] == 0]
    while missing and time.time() < start_time + 8:
        best_elem = None
        best_gain = 0
        for x in range(n+1):
            if x in A:
                continue
            gain = 0
            for s in missing:
                y = s - x
                if 0 <= y <= n and y in A:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = x
        if best_elem is None or best_gain == 0:
            break
        A.add(best_elem)
        # Update mults
        for y in A:
            if y == best_elem:
                s = best_elem + best_elem
                if target_start <= s <= target_end:
                    mults[s] += 1
            elif y > best_elem:
                s = best_elem + y
                if target_start <= s <= target_end:
                    mults[s] += 1
        missing = [s for s in range(target_start, target_end+1) if mults[s] == 0]
    
    if not missing:
        peak = get_peak(mults)
        var = get_var(mults)
        smooth = smooth_objective(mults)
        best_A = set(A)
        best_peak = peak
        best_var = var
        best_smooth = smooth
    
    if not best_A:
        # Fallback to seed
        return list(range(64, 129))
        
    # Local Search with incremental updates and smooth objective
    T = 5.0
    cooling = 0.9995
    
    A = set(best_A)
    mults = compute_mults(A)
    peak = best_peak
    var = best_var
    smooth = best_smooth
    
    # Precompute list of elements for faster random selection
    A_list = list(A)
    
    # Counters for move types
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        T *= cooling
        if T < 0.005:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.25:
            # Add
            not_in_A = [x for x in range(n+1) if x not in A]
            if not not_in_A:
                continue
            x = random.choice(not_in_A)
            
            # Incremental update: add x to A
            new_mults = mults.copy()
            for y in A_list:
                s = x + y
                if target_start <= s <= target_end:
                    new_mults[s] += 1
            # Add self-pair
            s = x + x
            if target_start <= s <= target_end:
                new_mults[s] += 1
            
            # Check coverage
            if not is_covered(new_mults):
                continue
                
            new_peak = get_peak(new_mults)
            new_var = get_var(new_mults)
            new_smooth = smooth_objective(new_mults)
            
            if new_smooth < smooth:
                A.add(x)
                A_list.append(x)
                mults = new_mults
                peak = new_peak
                var = new_var
                smooth = new_smooth
            else:
                delta = new_smooth - smooth
                prob = math.exp(-delta / T) if delta > 0 else 1.0
                if random.random() < prob:
                    A.add(x)
                    A_list.append(x)
                    mults = new_mults
                    peak = new_peak
                    var = new_var
                    smooth = new_smooth
                    
        elif move_type < 0.50:
            # Remove
            if len(A) < 35:
                continue
            x = random.choice(A_list)
            
            # Incremental update: remove x from A
            new_mults = mults.copy()
            for y in A_list:
                if y == x:
                    continue
                s = x + y
                if target_start <= s <= target_end:
                    new_mults[s] -= 1
            # Remove self-pair
            s = x + x
            if target_start <= s <= target_end:
                new_mults[s] -= 1
            
            # Check coverage
            if not is_covered(new_mults):
                continue
                
            new_peak = get_peak(new_mults)
            new_var = get_var(new_mults)
            new_smooth = smooth_objective(new_mults)
            
            if new_smooth < smooth:
                A.remove(x)
                A_list.remove(x)
                mults = new_mults
                peak = new_peak
                var = new_var
                smooth = new_smooth
            else:
                delta = new_smooth - smooth
                prob = math.exp(-delta / T) if delta > 0 else 1.0
                if random.random() < prob:
                    A.remove(x)
                    A_list.remove(x)
                    mults = new_mults
                    peak = new_peak
                    var = new_var
                    smooth = new_smooth
                    
        else:
            # Swap
            if len(A) < 35:
                continue
            x = random.choice(A_list)
            not_in_A = [y for y in range(n+1) if y not in A]
            if not not_in_A:
                continue
            y = random.choice(not_in_A)
            
            # Incremental update: remove x, add y
            new_mults = mults.copy()
            
            # Remove pairs involving x
            for z in A_list:
                if z == x:
                    continue
                s = x + z
                if target_start <= s <= target_end:
                    new_mults[s] -= 1
            s = x + x
            if target_start <= s <= target_end:
                new_mults[s] -= 1
            
            # Add pairs involving y
            for z in A_list:
                if z == x:
                    continue  # x is being removed, skip
                s = y + z
                if target_start <= s <= target_end:
                    new_mults[s] += 1
            # Add y-y pair
            s = y + y
            if target_start <= s <= target_end:
                new_mults[s] += 1
            
            # Check coverage
            if not is_covered(new_mults):
                continue
                
            new_peak = get_peak(new_mults)
            new_var = get_var(new_mults)
            new_smooth = smooth_objective(new_mults)
            
            if new_smooth < smooth:
                A.remove(x)
                A.add(y)
                A_list[A_list.index(x)] = y
                mults = new_mults
                peak = new_peak
                var = new_var
                smooth = new_smooth
            else:
                delta = new_smooth - smooth
                prob = math.exp(-delta / T) if delta > 0 else 1.0
                if random.random() < prob:
                    A.remove(x)
                    A.add(y)
                    A_list[A_list.index(x)] = y
                    mults = new_mults
                    peak = new_peak
                    var = new_var
                    smooth = new_smooth
                
        # Update global best
        if peak < best_peak or (peak == best_peak and var < best_var):
            best_A = set(A)
            best_peak = peak
            best_var = var
            best_smooth = smooth
    
    return list(best_A)
