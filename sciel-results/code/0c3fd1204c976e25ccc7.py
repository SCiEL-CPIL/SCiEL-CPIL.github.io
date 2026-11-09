import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    random.seed(42)
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs."""
        A_arr = np.array(sorted(A), dtype=np.int32)
        n = len(A_arr)
        if n == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Use triu indices for unordered pairs with i<=j
        idx = np.triu_indices(n)
        upper_sums = A_arr[idx[0]] + A_arr[idx[1]]
        
        mask = (upper_sums >= N_LOW) & (upper_sums <= N_HIGH)
        if not np.any(mask):
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        valid_sums = upper_sums[mask]
        indices = (valid_sums - N_LOW).astype(np.int32)
        counts = np.bincount(indices, minlength=RANGE_SIZE)
        return counts[:RANGE_SIZE].astype(np.int32)

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def element_contribution(x, A, reps):
        """For element x in set A, return (num_peak_contrib, coverage_risk)."""
        A_arr = np.array(sorted(A), dtype=np.int32)
        peak = get_peak(reps)
        
        sums = x + A_arr
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if not np.any(mask):
            return 0, 0
        
        valid_sums = sums[mask]
        indices = valid_sums - N_LOW
        rep_vals = reps[indices]
        
        peak_count = int(np.sum(rep_vals == peak))
        coverage_risk = int(np.sum(rep_vals == 1))
        
        return peak_count, coverage_risk

    def find_peak_elements(A, reps):
        """Find elements that contribute most to peak sums."""
        peak = get_peak(reps)
        if peak == 0:
            return []
        
        A_list = list(A)
        scores = {}
        for x in A_list:
            pc, _ = element_contribution(x, A, reps)
            scores[x] = pc
        
        sorted_elems = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_elems

    def random_init():
        A = set()
        A.add(128)
        A.add(0)
        for x in [1, 2, 3, 4, 124, 125, 126, 127]:
            A.add(x)
        for x in [64, 63, 65]:
            A.add(x)
        while len(A) < 45:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        """Add elements until all sums in [128,256] are covered."""
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 100:
            missing = np.where(reps == 0)[0] + N_LOW
            best_candidate = None
            best_gain = 0
            for s in missing:
                candidates = set()
                for x in range(0, 129):
                    y = s - x
                    if 0 <= y <= 128:
                        candidates.add(x)
                candidates -= A
                for c in list(candidates)[:20]:
                    new_A = A | {c}
                    new_reps = calc_reps(new_A)
                    new_missing = np.sum(new_reps == 0)
                    gain = len(missing) - new_missing
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c
            if best_candidate is not None:
                A.add(best_candidate)
                reps = calc_reps(A)
            else:
                A.add(random.randint(0, 128))
                reps = calc_reps(A)
            tries += 1
        return A, reps

    def greedy_removal_pass(A, reps, peak):
        """Try to remove elements that don't increase peak or break coverage."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in list(A):
                if time.time() > deadline:
                    break
                new_A = A - {a}
                if not new_A:
                    continue
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak <= peak:
                        A = new_A
                        reps = new_reps
                        peak = new_peak
                        improved = True
        return A, reps, peak

    best_A = None
    best_peak = float('inf')
    
    def update_best(A, reps):
        nonlocal best_A, best_peak
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)

    # Phase 1: Multi-start Local Search with peak-aware moves
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline: break
        
        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)
        
        if not is_covered(current_reps):
            continue

        # Greedy removal to shrink initial set
        current_A, current_reps, current_peak = greedy_removal_pass(current_A, current_reps, get_peak(current_reps))
        
        update_best(current_A, current_reps)
        
        # Local Search: Peak-aware Simulated Annealing
        T = 1.5
        cooling = 0.997
        min_T = 0.03
        stuck_counter = 0
        max_stuck = 200
        
        while time.time() < deadline and T > min_T:
            if stuck_counter > max_stuck:
                break
            
            current_peak = get_peak(current_reps)
            move_type = random.random()
            
            if len(current_A) > 15 and move_type < 0.4:
                # Try Remove: prioritize elements with high peak contribution
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    weights = []
                    for x in peak_elems:
                        pc, _ = element_contribution(x, current_A, current_reps)
                        weights.append(pc + 1)
                    total_w = sum(weights)
                    r_val = random.random() * total_w
                    cum = 0
                    item_to_remove = peak_elems[-1]
                    for i, x in enumerate(peak_elems):
                        cum += weights[i]
                        if r_val <= cum:
                            item_to_remove = x
                            break
                else:
                    item_to_remove = random.choice(list(current_A))
                
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    
                    if new_peak < current_peak:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    elif new_peak == current_peak:
                        if random.random() < 0.3:
                            current_A = new_A
                            current_reps = new_reps
                            update_best(current_A, current_reps)
                            stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            elif move_type < 0.7:
                # Try Swap
                if not current_A:
                    break
                a = random.choice(list(current_A))
                while True:
                    b = random.randint(0, MAX_VAL)
                    if b not in current_A:
                        break
                
                new_A = set(current_A)
                new_A.remove(a)
                new_A.add(b)
                
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    delta = new_peak - current_peak
                    
                    accept = False
                    if delta < 0:
                        accept = True
                    elif delta == 0:
                        accept = random.random() < 0.5
                    else:
                        if T > 0:
                            accept = random.random() < math.exp(-delta / T)
                    
                    if accept:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Try Add
                b = random.randint(0, MAX_VAL)
                if b not in current_A:
                    new_A = set(current_A)
                    new_A.add(b)
                    new_reps = calc_reps(new_A)
                    
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        delta = new_peak - current_peak
                        
                        accept = False
                        if delta < 0:
                            accept = True
                        elif delta == 0:
                            accept = random.random() < 0.3
                        else:
                            if T > 0:
                                accept = random.random() < math.exp(-delta / T)
                        
                        if accept:
                            current_A = new_A
                            current_reps = new_reps
                            update_best(current_A, current_reps)
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
            
            T *= cooling
        
        # Final polish for this restart
        current_A, current_reps, current_peak = greedy_removal_pass(current_A, current_reps, get_peak(current_reps))
        update_best(current_A, current_reps)

    if best_A is None:
        # Fallback
        return list(range(64, 129))

    # Final global polish
    best_A_set = set(best_A)
    best_reps = calc_reps(best_A_set)
    best_peak_val = get_peak(best_reps)
    best_A_set, best_reps, best_peak_val = greedy_removal_pass(best_A_set, best_reps, best_peak_val)

    return list(best_A_set)

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
