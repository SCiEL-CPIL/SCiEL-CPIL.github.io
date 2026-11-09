import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 128
    TARGET_LO = 128
    TARGET_HI = 256
    
    def compute_mults(A_arr):
        k = len(A_arr)
        mults = np.zeros(TARGET_HI + 1, dtype=np.int32)
        for i in range(k):
            ai = A_arr[i]
            for j in range(i, k):
                s = ai + A_arr[j]
                if s > TARGET_HI:
                    break
                if s >= TARGET_LO:
                    mults[s] += 1
        return mults
    
    def incremental_swap_mults(mults, A_set, a_out, a_in):
        """Compute new mults after swapping a_out (in A) with a_in (not in A)."""
        new_mults = mults.copy()
        A_list = sorted(A_set)
        
        # Remove a_out's contributions
        # Pair (a_out, a_out)
        s = 2 * a_out
        if TARGET_LO <= s <= TARGET_HI:
            new_mults[s] -= 1
        # Pairs (a_out, b) for b in A, b != a_out
        for b in A_list:
            if b == a_out:
                continue
            s = a_out + b
            if TARGET_LO <= s <= TARGET_HI:
                new_mults[s] -= 1
        
        # Add a_in's contributions
        s = 2 * a_in
        if TARGET_LO <= s <= TARGET_HI:
            new_mults[s] += 1
        # Pairs (a_in, b) for b in A_set - {a_out}
        for b in A_list:
            if b == a_out:
                continue
            s = a_in + b
            if TARGET_LO <= s <= TARGET_HI:
                new_mults[s] += 1
        
        return new_mults
    
    def is_valid_from_mults(mults):
        return bool(np.all(mults[TARGET_LO:TARGET_HI+1] > 0))
    
    def get_peak_from_mults(mults):
        return int(np.max(mults[TARGET_LO:TARGET_HI+1]))
    
    def get_sos_from_mults(mults):
        m = mults[TARGET_LO:TARGET_HI+1]
        return int(np.sum(m ** 2))
    
    def combined_from_mults(mults):
        m = mults[TARGET_LO:TARGET_HI+1]
        return int(np.max(m)) * 100 + int(np.sum(m ** 2))
    
    # Initial seed
    seed_A = set(range(64, 129))
    seed_arr = np.array(sorted(seed_A), dtype=np.int32)
    seed_mults = compute_mults(seed_arr)
    seed_peak = get_peak_from_mults(seed_mults)
    
    best_A = set(seed_A)
    best_mults = seed_mults.copy()
    best_peak = seed_peak
    
    def sa_and_polish(A_init, time_budget):
        t0 = time.time()
        current_A = set(A_init)
        current_arr = np.array(sorted(current_A), dtype=np.int32)
        current_mults = compute_mults(current_arr)
        current_sos = get_sos_from_mults(current_mults)
        
        best_local_A = set(current_A)
        best_local_mults = current_mults.copy()
        best_local_peak = get_peak_from_mults(current_mults)
        best_local_sos = current_sos
        
        T = 500.0
        T_min = 0.5
        cool = 0.997
        
        while T > T_min and (time.time() - t0) < time_budget * 0.75:
            A_list = sorted(current_A)
            not_in_A = [x for x in range(N + 1) if x not in current_A]
            if not not_in_A:
                break
            
            a_out = random.choice(A_list)
            a_in = random.choice(not_in_A)
            
            new_mults = incremental_swap_mults(current_mults, current_A, a_out, a_in)
            
            if not is_valid_from_mults(new_mults):
                T *= cool
                continue
            
            new_sos = get_sos_from_mults(new_mults)
            delta = new_sos - current_sos
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-9)):
                current_A = set(current_A)
                current_A.discard(a_out)
                current_A.add(a_in)
                current_mults = new_mults
                current_sos = new_sos
                
                new_peak = get_peak_from_mults(current_mults)
                if new_peak < best_local_peak or (new_peak == best_local_peak and current_sos < best_local_sos):
                    best_local_peak = new_peak
                    best_local_sos = current_sos
                    best_local_A = set(current_A)
                    best_local_mults = current_mults.copy()
            
            T *= cool
        
        # Local search polish
        remaining = time_budget - (time.time() - t0)
        if remaining > 0.5:
            best_local_A, best_local_mults = local_search(best_local_A, best_local_mults, remaining)
            best_local_peak = get_peak_from_mults(best_local_mults)
        
        return best_local_A, best_local_mults, best_local_peak
    
    def local_search(A_set, mults, time_budget):
        t0 = time.time()
        current_A = set(A_set)
        current_mults = mults.copy()
        current_obj = combined_from_mults(current_mults)
        
        while (time.time() - t0) < time_budget:
            improved = False
            A_list = sorted(current_A)
            not_in_A = [x for x in range(N + 1) if x not in current_A]
            
            for a_out in A_list:
                if (time.time() - t0) >= time_budget:
                    break
                for a_in in not_in_A:
                    new_mults = incremental_swap_mults(current_mults, current_A, a_out, a_in)
                    if not is_valid_from_mults(new_mults):
                        continue
                    new_obj = combined_from_mults(new_mults)
                    if new_obj < current_obj:
                        current_A = set(current_A)
                        current_A.discard(a_out)
                        current_A.add(a_in)
                        current_mults = new_mults
                        current_obj = new_obj
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_A, current_mults
    
    # Run SA on seed
    if (time.time() - start_time) < DEADLINE - 5:
        budget = min(35.0, DEADLINE - (time.time() - start_time) - 5)
        A_res, mults_res, peak_res = sa_and_polish(seed_A, budget)
        if peak_res < best_peak:
            best_peak = peak_res
            best_A = A_res
            best_mults = mults_res
    
    # Try random initializations
    for size in [55, 60, 65, 70, 75]:
        if (time.time() - start_time) > DEADLINE - 15:
            break
        
        for _ in range(2):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            
            A_init = set(random.sample(range(N + 1), size))
            A_arr = np.array(sorted(A_init), dtype=np.int32)
            m = compute_mults(A_arr)
            if not is_valid_from_mults(m):
                continue
            
            budget = min(15.0, DEADLINE - (time.time() - start_time) - 5)
            if budget < 3.0:
                break
            
            A_res, mults_res, peak_res = sa_and_polish(A_init, budget)
            if peak_res < best_peak:
                best_peak = peak_res
                best_A = A_res
                best_mults = mults_res
    
    return sorted(best_A)
