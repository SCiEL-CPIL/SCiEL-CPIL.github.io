import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    n = 128
    target_start = 128
    target_end = 256
    target_range = np.arange(target_start, target_end + 1)
    N = 129  # size of [0,128]
    
    def evaluate_fast(v):
        """
        v: numpy array of 0/1 of length 129.
        Returns (max_mult, num_uncovered).
        Uses convolution for speed.
        """
        conv = np.convolve(v, v, mode='full')
        # For target t, ordered count = conv[t]
        # Unordered count = (conv[t] + diag[t]) / 2 where diag[t]=1 if t even and t/2 in A
        ordered = conv[target_range]
        # diag: for even t, check if t//2 is in A
        even_mask = (target_range % 2 == 0)
        diag = np.zeros(len(target_range), dtype=np.int32)
        even_targets = target_range[even_mask]
        half_vals = even_targets // 2
        diag[even_mask] = v[half_vals]
        unordered = (ordered + diag) // 2
        num_uncovered = int(np.sum(unordered == 0))
        max_mult = int(np.max(unordered)) if num_uncovered < len(target_range) else 0
        return max_mult, num_uncovered
    
    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered, total_pairs)
        """
        if not A_set:
            return 1000, 100, 0
        v = np.zeros(N, dtype=np.int32)
        for x in A_set:
            if 0 <= x <= 128:
                v[x] = 1
            else:
                return 1000, 100, 0
        max_mult, num_uncovered = evaluate_fast(v)
        total_pairs = len(A_set) * (len(A_set) + 1) // 2
        return max_mult, num_uncovered, total_pairs
    
    def cost_from_eval(max_mult, num_uncovered):
        return num_uncovered * 10000 + max_mult

    def greedy_construct(v, max_adds=200):
        """
        Greedy: start from v (binary vector), add elements to cover all targets.
        Returns the resulting binary vector.
        """
        v = v.copy()
        for _ in range(max_adds):
            if time.time() > deadline - 3:
                break
            _, unc = evaluate_fast(v)
            if unc == 0:
                break
            # Find which targets are uncovered
            conv = np.convolve(v, v, mode='full')
            ordered = conv[target_range]
            even_mask = (target_range % 2 == 0)
            diag = np.zeros(len(target_range), dtype=np.int32)
            even_targets = target_range[even_mask]
            half_vals = even_targets // 2
            diag[even_mask] = v[half_vals]
            unordered = (ordered + diag) // 2
            uncovered_targets = target_range[unordered == 0]
            if len(uncovered_targets) == 0:
                break
            # For each candidate element a not in A, count how many uncovered targets it helps
            best_a = -1
            best_gain = 0
            for a in range(129):
                if v[a] == 1:
                    continue
                # Adding a: new pairs are (a, b) for b in A, plus (a,a)
                # For each uncovered target t, check if a + b = t for some b in A, or 2a = t
                gain = 0
                for t in uncovered_targets:
                    b = t - a
                    if 0 <= b <= 128:
                        if b == a:
                            # (a,a) gives 1 rep for t
                            gain += 1
                        elif v[b] == 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
            if best_a >= 0 and best_gain > 0:
                v[best_a] = 1
            else:
                break
        return v

    def local_search_polish(v, max_iters=5000):
        """
        Local search on binary vector v. Uses fast convolution evaluation.
        Tries add/remove/swap moves. Returns best binary vector found.
        """
        current_v = v.copy()
        cur_max, cur_unc = evaluate_fast(current_v)
        cur_cost = cost_from_eval(cur_max, cur_unc)
        best_v = current_v.copy()
        best_cost = cur_cost
        
        T = 3.0
        T_min = 0.05
        cooling = 0.995
        
        it = 0
        while it < max_iters and time.time() < deadline - 1:
            it += 1
            move_type = random.random()
            new_v = current_v.copy()
            
            if move_type < 0.35 and current_v.sum() > 2:
                # Remove
                ones = np.where(current_v == 1)[0]
                x = random.choice(ones)
                new_v[x] = 0
            elif move_type < 0.75:
                # Add
                zeros = np.where(current_v == 0)[0]
                if len(zeros) == 0:
                    continue
                x = random.choice(zeros)
                new_v[x] = 1
            else:
                # Swap
                ones = np.where(current_v == 1)[0]
                zeros = np.where(current_v == 0)[0]
                if len(ones) == 0 or len(zeros) == 0:
                    continue
                x = random.choice(ones)
                y = random.choice(zeros)
                new_v[x] = 0
                new_v[y] = 1
            
            if new_v.sum() == 0:
                continue
            
            new_max, new_unc = evaluate_fast(new_v)
            new_cost = cost_from_eval(new_max, new_unc)
            delta = new_cost - cur_cost
            
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                current_v = new_v
                cur_cost = new_cost
                cur_max, cur_unc = evaluate_fast(current_v)
                
                if cur_cost < best_cost:
                    best_v = current_v.copy()
                    best_cost = cur_cost
            
            T = max(T * cooling, T_min)
        
        return best_v

    best_overall_v = None
    best_overall_cost = float('inf')
    
    # Multiple restarts with different random initializations
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Initialize: random subset including 128
        init_v = np.zeros(N, dtype=np.int32)
        init_v[128] = 1
        num_init = random.randint(10, 35)
        candidates = list(range(128))
        random.shuffle(candidates)
        for c in candidates[:num_init]:
            init_v[c] = 1
        
        # Greedy to cover all targets
        greedy_v = greedy_construct(init_v)
        
        # Local search polish
        polished_v = local_search_polish(greedy_v, max_iters=2000)
        
        pm, pu = evaluate_fast(polished_v)
        pcost = cost_from_eval(pm, pu)
        
        if pcost < best_overall_cost:
            best_overall_cost = pcost
            best_overall_v = polished_v.copy()
        
        # Also try polishing the initial set directly
        polished_init = local_search_polish(init_v, max_iters=500)
        pim, piu = evaluate_fast(polished_init)
        picost = cost_from_eval(pim, piu)
        if picost < best_overall_cost:
            best_overall_cost = picost
            best_overall_v = polished_init.copy()
    
    # Fallback
    if best_overall_v is None:
        best_overall_v = np.zeros(N, dtype=np.int32)
        best_overall_v[64:129] = 1
    
    # Final polish: try removing redundant elements to reduce size (keeps coverage)
    if time.time() < deadline - 2:
        m, u = evaluate_fast(best_overall_v)
        if u == 0:
            for x in sorted(np.where(best_overall_v == 1)[0], reverse=True):
                if time.time() > deadline - 2:
                    break
                test_v = best_overall_v.copy()
                test_v[x] = 0
                if test_v.sum() == 0:
                    continue
                m_t, u_t = evaluate_fast(test_v)
                if u_t == 0 and m_t <= m:
                    best_overall_v = test_v
                    m, u = evaluate_fast(best_overall_v)
    
    # Convert to sorted list
    result = sorted(int(i) for i in np.where(best_overall_v == 1)[0])
    if not result:
        result = [128]
    
    return result
