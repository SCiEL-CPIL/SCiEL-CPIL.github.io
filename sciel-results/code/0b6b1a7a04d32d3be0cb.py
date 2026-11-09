import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        # Fast convolution-based representation counting
        arr = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A:
            arr[a] = 1
        
        conv = np.convolve(arr, arr, mode='full')
        # conv[s] = number of ordered pairs (i,j) with i+j=s
        # For unordered pairs: if i != j, count is conv[s]/2; if i==j (s even, i=s/2), add 1
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx in range(RANGE_SIZE):
            s = N_LOW + idx
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= MAX_VAL:
                    # Ordered count includes (half, half) once
                    # Unordered: (conv[s] - 1) / 2 + 1 = (conv[s] + 1) / 2
                    reps[idx] = (conv[s] + 1) // 2
                else:
                    reps[idx] = conv[s] // 2
            else:
                # No equal summands possible
                reps[idx] = conv[s] // 2
        return reps

    def get_metrics(reps):
        peak = int(np.max(reps))
        covered = bool(np.all(reps > 0))
        # Sum of squares penalizes uneven distribution
        sum_sq = int(np.sum(reps * reps))
        return peak, covered, sum_sq

    def get_cost(peak, covered, sum_sq):
        if not covered:
            return 10000 + peak * 100 + sum_sq
        return peak * 1000 + sum_sq

    best_A = None
    best_peak = float('inf')
    best_sum_sq = float('inf')
    best_cost = float('inf')
    
    # Try multiple random starts with varying set sizes
    # Smaller sets have fewer pairs, potentially lower peak if well-distributed
    # But must cover 129 sums. Min pairs ~ 129.
    # k=16 -> 136 pairs. k=15 -> 120 pairs (cannot cover 129 sums with peak 1, but can with peak 2).
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random size between 15 and 25
        size = random.randint(15, 25)
        A = set(random.sample(range(MAX_VAL + 1), size))
        
        # Ensure coverage first
        reps = calc_reps(A)
        covered = bool(np.all(reps > 0))
        
        # If not covered, add elements greedily until covered
        while not covered and time.time() < deadline:
            missing_indices = np.where(reps == 0)[0]
            if len(missing_indices) == 0:
                break
            # Pick a random missing sum
            idx = random.choice(missing_indices)
            S = N_LOW + idx
            # Try to add one or two elements to cover S
            found = False
            # Try adding one element that pairs with existing
            for a in A:
                b = S - a
                if 0 <= b <= MAX_VAL and b not in A:
                    A.add(b)
                    found = True
                    break
            if not found:
                # Try adding two new elements
                for a in range(MAX_VAL + 1):
                    b = S - a
                    if 0 <= b <= MAX_VAL and a not in A and b not in A:
                        A.add(a)
                        A.add(b)
                        found = True
                        break
            if not found:
                # Fallback: add random
                A.add(random.randint(0, MAX_VAL))
            
            reps = calc_reps(A)
            covered = bool(np.all(reps > 0))
            
        if not covered:
            continue
        
        # SA Optimization
        current_A = set(A)
        current_reps = calc_reps(current_A)
        cur_peak, _, cur_sum_sq = get_metrics(current_reps)
        cur_cost = get_cost(cur_peak, True, cur_sum_sq)
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline and T > T_min:
            # Move: Swap an element
            a = random.choice(list(current_A))
            b = random.randint(0, MAX_VAL)
            if b in current_A:
                continue
            
            new_A = current_A - {a} | {b}
            new_reps = calc_reps(new_A)
            new_peak, new_covered, new_sum_sq = get_metrics(new_reps)
            new_cost = get_cost(new_peak, new_covered, new_sum_sq)
            
            delta = new_cost - cur_cost
            
            accept = False
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_A = new_A
                current_reps = new_reps
                cur_peak = new_peak
                cur_sum_sq = new_sum_sq
                cur_cost = new_cost
                
                if new_covered:
                    # Update best if better
                    if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                        best_peak = new_peak
                        best_sum_sq = new_sum_sq
                        best_A = set(current_A)
                        best_cost = new_cost
            
            T *= cooling
            
            # Occasional removal attempt to shrink set
            if random.random() < 0.05 and len(current_A) > 15:
                a_to_remove = random.choice(list(current_A))
                new_A = current_A - {a_to_remove}
                new_reps = calc_reps(new_A)
                new_peak, new_covered, new_sum_sq = get_metrics(new_reps)
                new_cost = get_cost(new_peak, new_covered, new_sum_sq)
                
                delta = new_cost - cur_cost
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_A = new_A
                    current_reps = new_reps
                    cur_peak = new_peak
                    cur_sum_sq = new_sum_sq
                    cur_cost = new_cost
                    
                    if new_covered:
                        if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                            best_peak = new_peak
                            best_sum_sq = new_sum_sq
                            best_A = set(current_A)
                            best_cost = new_cost

    # Final Polish: Try to remove elements from best_A
    if best_A:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in list(best_A):
                new_A = best_A - {a}
                if not new_A:
                    continue
                new_reps = calc_reps(new_A)
                new_peak, covered, new_sum_sq = get_metrics(new_reps)
                if covered:
                    if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                        best_A = new_A
                        best_peak = new_peak
                        best_sum_sq = new_sum_sq
                        improved = True

    if best_A is None:
        # Fallback: Dense set
        return list(range(0, 129, 2))
        
    return list(best_A)

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
