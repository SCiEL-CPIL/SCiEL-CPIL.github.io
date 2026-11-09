import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for 60s limit (90s wall-clock kill)
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129 values
    MAX_VAL = 128
    
    # Precompute: for each possible sum s in [128,256], list of pairs (i,j) with i<=j
    # such that a[i]+a[j] = s. But a is variable, so we compute on the fly.
    
    def calc_reps(A):
        """
        A: sorted list of ints in [0,128]
        Returns: np.array of length RANGE_SIZE where reps[k] = number of unordered pairs (i,j) with i<=j, a[i]+a[j] = N_LOW+k
        """
        A_arr = np.array(A, dtype=np.int32)
        n = len(A_arr)
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for i in range(n):
            ai = A_arr[i]
            # For j >= i, sums are ai + A_arr[i:]
            sums = ai + A_arr[i:]
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            if np.any(mask):
                valid_sums = sums[mask]
                indices = valid_sums - N_LOW
                # Count occurrences
                # Use np.bincount for speed
                counts = np.bincount(indices, minlength=RANGE_SIZE)
                reps += counts[:RANGE_SIZE]
        return reps
    
    def is_covered(reps):
        """Check that every value in [N_LOW, N_HIGH] has at least one representation."""
        return np.all(reps > 0)
    
    def get_peak(reps):
        """Return the maximum multiplicity."""
        return int(np.max(reps))
    
    # --- Phase 1: Greedy construction to get a valid covering set ---
    # Start with a reasonable seed and improve.
    
    # Strategy: Start with a set that guarantees coverage, then optimize.
    # A simple valid set: [0, 128] gives sums {0, 128, 256} - not enough.
    # We need every sum in [128, 256].
    
    # Let's use a randomized local search approach.
    # Start with a random valid set.
    
    def random_valid_set():
        """Generate a random set that covers [128, 256]."""
        # Start with a known valid set and perturb.
        # The seed [64, 128] covers [128, 256] with high multiplicity.
        # Let's start with a subset of that or a random set and fix coverage.
        
        # Actually, let's start with a random set of moderate size and check coverage.
        # If not covered, add elements greedily.
        
        # Start with random elements
        A = set()
        # Add some random elements
        for _ in range(20):
            A.add(random.randint(0, 128))
        
        # Ensure coverage by adding elements
        A = list(A)
        A.sort()
        reps = calc_reps(A)
        
        # While there are uncovered sums, add a random element that helps
        max_iter = 200
        for _ in range(max_iter):
            if is_covered(reps):
                break
            # Find uncovered sums
            uncovered = np.where(reps == 0)[0]
            if len(uncovered) == 0:
                break
            # Pick a random uncovered sum
            target_sum = N_LOW + random.choice(uncovered)
            # Find a pair (x, y) in [0, 128] with x <= y, x+y = target_sum
            # Add one of them to A
            x = random.randint(0, 128)
            y = target_sum - x
            if 0 <= y <= 128:
                if x <= y:
                    # Add x or y
                    if random.random() < 0.5:
                        A.append(x)
                    else:
                        A.append(y)
                else:
                    if random.random() < 0.5:
                        A.append(y)
                    else:
                        A.append(x)
            A = list(set(A))
            A.sort()
            reps = calc_reps(A)
        
        # If still not covered, fall back to the seed
        if not is_covered(reps):
            return list(range(64, 129))
        
        return A
    
    # --- Phase 2: Local search to minimize peak multiplicity ---
    
    def local_search(A, time_limit):
        """
        Local search: try to reduce peak multiplicity.
        Moves: remove an element, add an element, swap an element.
        Accept if peak decreases or coverage is maintained with same peak.
        """
        A = list(A)
        A.sort()
        reps = calc_reps(A)
        if not is_covered(reps):
            return A, get_peak(reps)
        
        best_peak = get_peak(reps)
        best_A = list(A)
        
        while time.time() < time_limit:
            improved = False
            
            # Try removing elements
            for i in range(len(A)):
                if time.time() >= time_limit:
                    break
                new_A = A[:i] + A[i+1:]
                if len(new_A) == 0:
                    continue
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_peak:
                        A = new_A
                        reps = new_reps
                        best_peak = new_peak
                        best_A = list(A)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try adding elements
            for x in range(0, 129):
                if time.time() >= time_limit:
                    break
                if x in A:
                    continue
                new_A = A + [x]
                new_A.sort()
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_peak:
                        A = new_A
                        reps = new_reps
                        best_peak = new_peak
                        best_A = list(A)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping elements
            for i in range(len(A)):
                if time.time() >= time_limit:
                    break
                old_val = A[i]
                for x in range(0, 129):
                    if x == old_val:
                        continue
                    if x in A:
                        continue
                    new_A = A[:i] + [x] + A[i+1:]
                    new_A.sort()
                    new_reps = calc_reps(new_A)
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        if new_peak < best_peak:
                            A = new_A
                            reps = new_reps
                            best_peak = new_peak
                            best_A = list(A)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Stuck. Try a random restart with perturbation.
                # Perturb: remove a random element and add a random one.
                if len(A) > 5:
                    i = random.randint(0, len(A)-1)
                    A = A[:i] + A[i+1:]
                    x = random.randint(0, 128)
                    A.append(x)
                    A = list(set(A))
                    A.sort()
                    reps = calc_reps(A)
                    if is_covered(reps):
                        peak = get_peak(reps)
                        if peak < best_peak:
                            best_peak = peak
                            best_A = list(A)
                    else:
                        # Revert
                        A = list(best_A)
                        reps = calc_reps(A)
                        best_peak = get_peak(reps)
                else:
                    break
        
        return best_A, best_peak
    
    # --- Main search ---
    
    # Start with a few random valid sets and run local search on the best.
    
    # First, generate a good starting point
    best_A = None
    best_peak = float('inf')
    
    # Try a few random starts
    for trial in range(5):
        if time.time() > deadline:
            break
        
        A = random_valid_set()
        reps = calc_reps(A)
        if not is_covered(reps):
            continue
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A)
    
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    # Run local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_A, best_peak = local_search(best_A, deadline - 5)
    
    return best_A

