import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safe margin for 60s limit
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                v[i] += 1
                temp //= p
        if temp > 1:
            for i in range(num_primes):
                if primes[i] == temp:
                    v[i] += 1
                    break
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Global ID registry for vectors to speed up set operations
    vec_to_id = {zero_vec: 0}
    next_id = 1
    
    def get_id(vec):
        nonlocal next_id
        if vec in vec_to_id:
            return vec_to_id[vec]
        vec_to_id[vec] = next_id
        next_id += 1
        return vec_to_id[vec]

    def compute_state(seq):
        pvecs = [zero_vec]
        pset = set()
        for x in seq:
            v_x = vecs[x]
            last = pvecs[-1]
            v_new_list = [last[i] + v_x[i] for i in range(num_primes)]
            v_new = tuple(v_new_list)
            
            valid = True
            new_ids = []
            for i in range(len(pvecs)):
                diff_list = [v_new[j] - pvecs[i][j] for j in range(num_primes)]
                diff_tup = tuple(diff_list)
                diff_id = get_id(diff_tup)
                if diff_id in pset:
                    valid = False
                    break
                new_ids.append(diff_id)
            
            if not valid:
                return None
            
            pvecs.append(v_new)
            pset.update(new_ids)
        
        return (pvecs, pset)

    def try_improve(seq, time_limit):
        seq = sorted(seq)
        state = compute_state(seq)
        if state is None:
            return seq
        pvecs, pset = state
        
        candidates = [x for x in range(2, 129) if x not in seq]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for c in candidates:
                if time.time() >= time_limit:
                    break
                if c in seq:
                    continue
                
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < c:
                        lo = mid + 1
                    else:
                        hi = mid
                ins_idx = lo
                
                if ins_idx < len(seq) and seq[ins_idx] == c:
                    continue
                
                new_seq = seq[:ins_idx] + [c] + seq[ins_idx:]
                new_state = compute_state(new_seq)
                
                if new_state is not None:
                    seq = new_seq
                    pvecs, pset = new_state
                    improved = True
        
        return seq

    # Phase 1: Start with all primes
    best_seq = [p for p in primes if p <= 128]
    
    # Phase 2: Improve with greedy insertion
    current_best = try_improve(best_seq, deadline)
    
    # Phase 3: Random perturbations
    while time.time() < deadline - 5.0:
        composites_in_best = [x for x in current_best if x not in primes]
        if not composites_in_best:
            break
        
        for _ in range(5):
            if time.time() >= deadline - 5.0:
                break
            
        rem_idx = random.randint(0, len(composites_in_best) - 1)
        rem_val = composites_in_best[rem_idx]
        new_seq = [x for x in current_best if x != rem_val]
        
        improved_seq = try_improve(new_seq, min(deadline - 5.0, time.time() + 2.0))
        
        if len(improved_seq) > len(current_best):
            current_best = improved_seq
            break
            
    return current_best

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
