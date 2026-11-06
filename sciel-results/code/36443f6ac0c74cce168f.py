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
    target_range = np.arange(target_start, target_end + 1)
    
    # Helper: compute multiplicities from a boolean array (True = in A)
    # Uses convolution for speed.
    # A_bool is length n+1.
    # Convolution of A_bool with itself gives counts of ordered pairs (a,b) such that a+b=k.
    # We need unordered pairs with equal summands allowed.
    # Ordered count C[k] = sum_{a+b=k} A[a]A[b].
    # Unordered count U[k] = (C[k] + D[k]) / 2, where D[k] = 1 if k is even and A[k/2] is True, else 0.
    # Wait, the problem says "Pair representations are unordered and permit equal summands".
    # This is equivalent to counting the number of pairs {a,b} with a,b in A, a<=b, a+b=k.
    # Let's verify the formula.
    # If a != b, the ordered pairs are (a,b) and (b,a). Both contribute to C[k]. They form 1 unordered pair.
    # If a == b, the ordered pair is (a,a). It contributes 1 to C[k]. It forms 1 unordered pair.
    # So, let C[k] be the number of ordered pairs.
    # Let E[k] be 1 if k is even and A[k/2] is in A, else 0.
    # The number of unordered pairs is (C[k] - E[k]) / 2 + E[k] = (C[k] + E[k]) / 2.
    
    def compute_mults(A_bool):
        # A_bool is numpy array of bools, length n+1
        # Convert to int for convolution
        A_int = A_bool.astype(np.int32)
        # Convolution
        conv = np.convolve(A_int, A_int, mode='full')
        # conv[k] is count of ordered pairs summing to k
        # We only care about k in [128, 256]
        mults = np.zeros(2 * n + 1, dtype=np.int32)
        for k in range(target_start, target_end + 1):
            c = conv[k]
            # Check if k/2 is in A and k is even
            e = 0
            if k % 2 == 0:
                mid = k // 2
                if 0 <= mid <= n and A_bool[mid]:
                    e = 1
            mults[k] = (c + e) // 2
        return mults

    def get_objective(mults):
        target_mults = mults[target_start:target_end+1]
        if np.any(target_mults == 0):
            # Penalize missing coverage heavily
            return float('inf'), 0, 0
        peak = int(np.max(target_mults))
        # We want to minimize peak. Secondary objective: minimize variance to balance loads?
        # The task says "Minimize the largest representation multiplicity".
        # So peak is the primary metric.
        mean = np.mean(target_mults)
        var = float(np.sum((target_mults - mean)**2))
        # Score function: primarily peak, secondarily variance
        return peak, var, float(peak * 10000.0 + var)

    random.seed(42)
    
    best_peak = float('inf')
    best_var = float('inf')
    best_score = float('inf')
    best_A_bool = None
    
    # Initialization: Greedy construction to ensure coverage
    A_bool = np.zeros(n+1, dtype=bool)
    
    # Start with a random set of reasonable size
    initial_size = 60
    initial_indices = random.sample(range(n+1), initial_size)
    for i in initial_indices:
        A_bool[i] = True
        
    mults = compute_mults(A_bool)
    peak, var, score = get_objective(mults)
    
    # If not covered, greedily add elements to cover gaps
    if peak == float('inf'):
        while time.time() < start_time + 5:
            missing = target_range[mults[target_range] == 0]
            if len(missing) == 0:
                break
            
            best_elem = None
            best_gain = -1
            # Try adding each element not in A
            candidates = [x for x in range(n+1) if not A_bool[x]]
            # Shuffle to break ties randomly
            random.shuffle(candidates)
            
            for x in candidates:
                # Calculate gain: how many missing sums does x cover?
                # x covers s if s-x is in A (and s-x >= x? No, unordered means {x, s-x}.
                # If s-x == x, it's {x,x}.
                # If s-x != x, we need s-x in A.
                # Note: if s-x < x, the pair is {s-x, x}. Since we are adding x, we check if s-x is in A.
                # This works for both cases.
                gain = 0
                for s in missing:
                    y = s - x
                    if 0 <= y <= n:
                        if y == x:
                            # Pair {x,x} is formed if we add x. It's always available once x is in A.
                            # But wait, if x is not in A yet, {x,x} is not formed.
                            # Once we add x, {x,x} is formed. So if 2*x == s, gain += 1.
                            gain += 1
                        else:
                            # Pair {x, y}. Requires y in A.
                            if A_bool[y]:
                                gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            
            if best_elem is None or best_gain == 0:
                # No single element can improve coverage? 
                # This shouldn't happen if we start with empty and A is small, 
                # but if A is large and gaps are isolated, maybe.
                # In that case, we might need to add multiple or swap. 
                # For now, break and rely on local search.
                break
                
            A_bool[best_elem] = True
            mults = compute_mults(A_bool)
            peak, var, score = get_objective(mults)
            
    if peak != float('inf'):
        best_A_bool = A_bool.copy()
        best_peak = peak
        best_var = var
        best_score = score

    if best_A_bool is None:
        # Fallback: return a valid set if we failed to construct one properly
        # A set of all even numbers covers all even sums. 
        # To cover odd sums, we need at least one odd.
        # Actually, just return a large set to be safe, though it might have high peak.
        # [0, 128] covers everything. Peak will be high.
        return list(range(0, 129))

    # Local Search: Simulated Annealing or Hill Climbing
    # We use a simple Hill Climbing with random restarts or SA.
    # Given the speed of compute_mults (convolution), we can do many steps.
    
    cur_A_bool = best_A_bool.copy()
    cur_mults = compute_mults(cur_A_bool)
    cur_peak, cur_var, cur_score = get_objective(cur_mults)
    
    best_A_bool = cur_A_bool.copy()
    best_peak = cur_peak
    best_var = cur_var
    best_score = cur_score
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Pre-compute indices for faster access
    all_indices = list(range(n+1))
    
    while time.time() < deadline:
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Generate a move
        move_type = random.random()
        new_A_bool = cur_A_bool.copy()
        
        if move_type < 0.3:
            # Add
            candidates = [x for x in all_indices if not cur_A_bool[x]]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_A_bool[x] = True
        elif move_type < 0.6:
            # Remove
            in_set = [x for x in all_indices if cur_A_bool[x]]
            if len(in_set) < 30: # Don't get too small
                continue
            x = random.choice(in_set)
            new_A_bool[x] = False
        else:
            # Swap
            in_set = [x for x in all_indices if cur_A_bool[x]]
            out_set = [x for x in all_indices if not cur_A_bool[x]]
            if not in_set or not out_set:
                continue
            x = random.choice(in_set)
            y = random.choice(out_set)
            new_A_bool[x] = False
            new_A_bool[y] = True
        
        new_mults = compute_mults(new_A_bool)
        new_peak, new_var, new_score = get_objective(new_mults)
        
        # Acceptance
        if new_score < cur_score:
            cur_A_bool = new_A_bool
            cur_mults = new_mults
            cur_peak = new_peak
            cur_var = new_var
            cur_score = new_score
            
            if new_score < best_score:
                best_A_bool = cur_A_bool.copy()
                best_peak = new_peak
                best_var = new_var
                best_score = new_score
        else:
            delta = new_score - cur_score
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / max(T, 1e-10))
            
            if random.random() < prob:
                cur_A_bool = new_A_bool
                cur_mults = new_mults
                cur_peak = new_peak
                cur_var = new_var
                cur_score = new_score

    # Convert best_A_bool to list of ints
    result = [int(i) for i in np.nonzero(best_A_bool)[0]]
    
    # Final sanity check: Ensure coverage
    # If not covered, we should have handled it, but let's double check.
    # If this check fails, it means our logic has a bug.
    # For safety, if not covered, return a known valid set (e.g., all elements)
    # But the evaluator rejects invalid results.
    # Let's verify coverage of result.
    res_set = set(result)
    covered = True
    for s in range(128, 257):
        found = False
        for a in res_set:
            b = s - a
            if b in res_set:
                found = True
                break
        if not found:
            covered = False
            break
            
    if not covered:
        # Emergency fallback: return a set that definitely covers.
        # A = [0, 128] covers 128.
        # A = [0..128] covers all.
        # Let's return a dense set to ensure validity, even if peak is high.
        return list(range(0, 129))
        
    return result
