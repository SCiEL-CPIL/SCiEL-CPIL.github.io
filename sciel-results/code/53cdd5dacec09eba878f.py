import time
import random
import numpy as np
from scipy.signal import fftconvolve

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    random.seed(42)
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    DOMAIN_SIZE = 257  # 0..256

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs."""
        if not A:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        v = np.zeros(DOMAIN_SIZE, dtype=np.float64)
        for x in A:
            v[x] = 1.0
        
        # Ordered pairs count
        c = fftconvolve(v, v, mode='full')
        
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_set = A
        
        for i, s in enumerate(range(N_LOW, N_HIGH + 1)):
            ord_count = int(c[s])
            eq_count = 1 if (s % 2 == 0 and s // 2 in A_set) else 0
            reps[i] = (ord_count + eq_count) // 2
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def get_sum_interference(A, reps):
        """
        For each element x in A, compute:
        - unique_coverage: number of sums in [128,256] where x is the ONLY element 
          that can form this sum with some other element in A
        - interference: number of sums where x contributes to a high-multiplicity sum
        Returns dict mapping x -> (unique_cov, interference)
        """
        unique_cov = {}
        interference = {}
        
        # Precompute for each sum s, which elements in A can pair to form s
        # sum_pairs[s] = set of elements x in A such that there exists y in A with x+y=s
        sum_pairs = {}
        for i, s in enumerate(range(N_LOW, N_HIGH + 1)):
            pairs = set()
            for x in A:
                y = s - x
                if 0 <= y <= MAX_VAL and y in A:
                    pairs.add(x)
            sum_pairs[s] = pairs
        
        # For each element, count unique coverage and interference
        peak = get_peak(reps)
        for x in A:
            uc = 0
            inf = 0
            for i, s in enumerate(range(N_LOW, N_HIGH + 1)):
                if x in sum_pairs[s]:
                    # x can form sum s
                    if len(sum_pairs[s]) == 1:
                        uc += 1  # x is the only element that can form this sum
                    if reps[i] >= peak and peak > 1:
                        inf += 1  # x contributes to a peak-multiplicity sum
            unique_cov[x] = uc
            interference[x] = inf
        
        return unique_cov, interference

    def greedy_remove_interference(A, reps, max_iterations=200):
        """
        Structural approach: greedily remove elements that have high interference 
        and low unique coverage, maintaining coverage.
        """
        A = set(A)
        reps = calc_reps(A)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if not is_covered(reps):
                break
            
            peak = get_peak(reps)
            if peak <= 2:
                break  # Can't do much better than peak 2
            
            unique_cov, interference = get_sum_interference(A, reps)
            
            # Score each element: we want to remove elements with high interference 
            # and low unique coverage
            # Score = interference - lambda * unique_cov
            # Higher score = more likely to remove
            lambda_val = 0.5
            scores = {}
            for x in A:
                scores[x] = interference[x] - lambda_val * unique_cov[x]
            
            # Try removing the element with highest score
            if not scores:
                break
                
            # Sort by score descending, try removing from highest score
            sorted_elems = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            
            removed = False
            for x in sorted_elems:
                new_A = set(A)
                new_A.remove(x)
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak <= peak:
                        A = new_A
                        reps = new_reps
                        removed = True
                        break
            
            if not removed:
                break
        
        return A, reps

    def local_search_swap(A, reps, max_steps=500):
        """
        Local search: try swaps that reduce peak multiplicity while maintaining coverage.
        """
        A = set(A)
        reps = calc_reps(A)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not is_covered(reps):
                break
            
            peak = get_peak(reps)
            if peak <= 2:
                break
            
            # Find elements that contribute to peak sums
            peak_sums = [s for i, s in enumerate(range(N_LOW, N_HIGH + 1)) if reps[i] == peak]
            
            # For each peak sum, find elements involved
            peak_elements = set()
            for s in peak_sums:
                for x in A:
                    y = s - x
                    if 0 <= y <= MAX_VAL and y in A:
                        peak_elements.add(x)
            
            if not peak_elements:
                break
            
            # Try swapping each peak element with a random element
            improved = False
            for x in list(peak_elements):
                if time.time() > deadline:
                    break
                
                # Try a few random swaps
                for _ in range(5):
                    new_x = random.randint(0, MAX_VAL)
                    if new_x == x:
                        continue
                    
                    new_A = set(A)
                    new_A.remove(x)
                    new_A.add(new_x)
                    
                    new_reps = calc_reps(new_A)
                    
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        if new_peak < peak:
                            A = new_A
                            reps = new_reps
                            peak = new_peak
                            improved = True
                            break
                        elif new_peak == peak and random.random() < 0.3:
                            # Accept equal peak with some probability to escape local minima
                            A = new_A
                            reps = new_reps
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try removing low-contribution elements
                unique_cov, interference = get_sum_interference(A, reps)
                low_contrib = [x for x in A if unique_cov[x] == 0 and interference[x] == 0]
                if low_contrib and len(A) > 15:
                    x_to_remove = random.choice(low_contrib)
                    new_A = set(A)
                    new_A.remove(x_to_remove)
                    new_reps = calc_reps(new_A)
                    if is_covered(new_reps):
                        A = new_A
                        reps = new_reps
                else:
                    break
        
        return A, reps

    def full_set_init():
        """Start with full set [0, 128] for maximum redundancy."""
        return set(range(0, MAX_VAL + 1))

    def random_subset_init(size):
        """Random subset of given size."""
        A = set()
        # Always include some key elements for coverage
        A.add(0)
        A.add(MAX_VAL)
        while len(A) < size:
            A.add(random.randint(0, MAX_VAL))
        return A

    best_A = None
    best_peak = float('inf')
    
    def update_best(A, reps):
        nonlocal best_A, best_peak
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)

    # Strategy 1: Start from full set, greedily remove interference
    if time.time() < deadline:
        A = full_set_init()
        reps = calc_reps(A)
        update_best(A, reps)
        A, reps = greedy_remove_interference(A, reps, max_iterations=300)
        update_best(A, reps)
        A, reps = local_search_swap(A, reps, max_steps=1000)
        update_best(A, reps)

    # Strategy 2: Multiple random subset inits with different sizes
    for size in [50, 40, 35, 30]:
        if time.time() > deadline:
            break
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            A = random_subset_init(size)
            # Ensure coverage
            reps = calc_reps(A)
            tries = 0
            while not is_covered(reps) and tries < 50:
                missing = [s for i, s in enumerate(range(N_LOW, N_HIGH + 1)) if reps[i] == 0]
                if not missing:
                    break
                s = random.choice(missing)
                candidates = [x for x in range(0, MAX_VAL + 1) 
                             if (s - x) in A and x not in A]
                if candidates:
                    A.add(random.choice(candidates))
                else:
                    A.add(random.randint(0, MAX_VAL))
                reps = calc_reps(A)
                tries += 1
            
            if not is_covered(reps):
                continue
            
            update_best(A, reps)
            A, reps = greedy_remove_interference(A, reps, max_iterations=200)
            update_best(A, reps)
            A, reps = local_search_swap(A, reps, max_steps=500)
            update_best(A, reps)

    # Final polish on best solution
    if best_A and time.time() < deadline:
        A = set(best_A)
        reps = calc_reps(A)
        
        # Additional greedy removal
        for _ in range(100):
            if time.time() > deadline:
                break
            if not is_covered(reps):
                break
            
            peak = get_peak(reps)
            if peak <= 2:
                break
            
            unique_cov, interference = get_sum_interference(A, reps)
            scores = {x: interference[x] - 0.5 * unique_cov[x] for x in A}
            sorted_elems = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            
            removed = False
            for x in sorted_elems:
                new_A = set(A)
                new_A.remove(x)
                new_reps = calc_reps(new_A)
                if is_covered(new_reps) and get_peak(new_reps) <= peak:
                    A = new_A
                    reps = new_reps
                    removed = True
                    break
            
            if not removed:
                break
        
        update_best(A, reps)
        
        # Final local search
        A, reps = local_search_swap(A, reps, max_steps=500)
        update_best(A, reps)

    if best_A is None:
        return list(range(64, 129))
        
    return best_A
