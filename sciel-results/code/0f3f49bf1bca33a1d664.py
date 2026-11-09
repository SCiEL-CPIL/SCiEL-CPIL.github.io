import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255
    random.seed(42)

    def time_left():
        return deadline - time.time()

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def score_mask(mask):
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score_fast(mask, old_val, new_val):
        """Compute change in score when replacing old_val with new_val in mask.
        Only checks the ~40 d values that can actually change."""
        old_bit = 1 << (old_val - 1)
        new_bit = 1 << (new_val - 1)
        new_mask = mask ^ old_bit ^ new_bit
        
        # Collect affected d values efficiently
        affected = set()
        # old_val as first element: d in [1, (MAX_VAL - old_val)//2]
        max_d_old_first = (MAX_VAL - old_val) // 2
        for d in range(1, min(max_d_old_first, MAX_D) + 1):
            affected.add(d)
        # old_val as middle: d in [1, min(old_val-1, MAX_VAL-old_val)]
        max_d_old_mid = min(old_val - 1, MAX_VAL - old_val)
        for d in range(1, min(max_d_old_mid, MAX_D) + 1):
            affected.add(d)
        # old_val as last: d in [1, old_val//2]
        max_d_old_last = old_val // 2
        for d in range(1, min(max_d_old_last, MAX_D) + 1):
            affected.add(d)
        # new_val as first element
        max_d_new_first = (MAX_VAL - new_val) // 2
        for d in range(1, min(max_d_new_first, MAX_D) + 1):
            affected.add(d)
        # new_val as middle
        max_d_new_mid = min(new_val - 1, MAX_VAL - new_val)
        for d in range(1, min(max_d_new_mid, MAX_D) + 1):
            affected.add(d)
        # new_val as last
        max_d_new_last = new_val // 2
        for d in range(1, min(max_d_new_last, MAX_D) + 1):
            affected.add(d)
        
        delta = 0
        for d in affected:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta

    def gen_initial(lst_type='random'):
        """Generate an initial individual."""
        if lst_type == 'interval':
            return list(range(1, N + 1))
        elif lst_type == 'ap':
            step = random.randint(5, 15)
            lst = [1 + i * step for i in range(N)]
            return lst if max(lst) <= MAX_VAL else list(range(1, N + 1))
        elif lst_type == 'sumset':
            mult = random.choice([2, 3, 4, 5])
            sizeA = random.randint(3, 8)
            sizeB = random.randint(3, 8)
            A = set(random.sample(range(1, 60), sizeA))
            maxB = (MAX_VAL - 1) // mult
            B = set(random.sample(range(1, maxB + 1), sizeB))
            S = set()
            for a in A:
                for b in B:
                    v = a + mult * b
                    if 1 <= v <= MAX_VAL:
                        S.add(v)
            if len(S) >= N:
                return random.sample(sorted(S), N)
            return list(range(1, N + 1))
        elif lst_type == 'two_ap':
            """Union of two disjoint arithmetic progressions with cross-product structure."""
            # Try various splits and steps
            best_set = None
            best_sc = 0
            for n1 in range(8, 25):
                n2 = N - n1
                if n2 < 2:
                    continue
                for step1 in range(3, 20):
                    a1 = random.randint(1, 10)
                    ap1 = [a1 + i * step1 for i in range(n1)]
                    if max(ap1) > MAX_VAL:
                        continue
                    for step2 in range(3, 20):
                        a2 = random.randint(1, 10)
                        ap2 = [a2 + j * step2 for j in range(n2)]
                        if max(ap2) > MAX_VAL:
                            continue
                        S = set(ap1 + ap2)
                        if len(S) == N:
                            sc = score_mask(list_to_mask(list(S)))
                            if sc > best_sc:
                                best_sc = sc
                                best_set = list(S)
            if best_set is not None:
                return best_set
            return list(range(1, N + 1))
        elif lst_type == 'cantor_like':
            """Cantor-like set defined by base-4 digit patterns."""
            # Generate numbers whose base-4 representation uses only digits 0 and 1
            # This creates a sparse set with many progression differences
            candidates = []
            for i in range(4**6):  # up to 4096, but we filter
                s = i
                valid = True
                for _ in range(6):
                    if s % 4 > 1:
                        valid = False
                        break
                    s //= 4
                if valid:
                    v = i + 1
                    if 1 <= v <= MAX_VAL:
                        candidates.append(v)
            if len(candidates) >= N:
                return random.sample(candidates, N)
            return list(range(1, N + 1))
        else:  # random
            return random.sample(range(1, MAX_VAL + 1), N)

    def simulated_annealing(initial_lst, max_iter=5000, T0=5.0, cooling=0.9995):
        """Simulated annealing to optimize a single individual."""
        lst = initial_lst[:]
        m = list_to_mask(lst)
        s = score_mask(m)
        s_set = set(lst)
        
        best_lst = lst[:]
        best_m = m
        best_s = s
        
        T = T0
        for it in range(max_iter):
            if time_left() < 5:
                break
            
            i = random.randint(0, N - 1)
            old = lst[i]
            new = random.randint(1, MAX_VAL)
            attempts = 0
            while new in s_set and new != old and attempts < 50:
                new = random.randint(1, MAX_VAL)
                attempts += 1
            if new == old or new in s_set:
                continue
            
            delta = delta_score_fast(m, old, new)
            new_s = s + delta
            
            if new_s >= s or (delta < 0 and random.random() < math.exp(-delta / T)):
                s_set.discard(old)
                s_set.add(new)
                lst[i] = new
                m ^= (1 << (old - 1)) ^ (1 << (new - 1))
                s = new_s
                if s > best_s:
                    best_s = s
                    best_lst = lst[:]
                    best_m = m
            
            T *= cooling
        
        return (best_lst, best_m, best_s)

    def local_search(lst, m, s, max_iter=100):
        """Hill climbing on a single individual."""
        s_set = set(lst)
        for _ in range(max_iter):
            if time_left() < 5:
                break
            improved = False
            for i in range(N):
                old = lst[i]
                best_new = old
                best_s = s
                # Try a few random replacements
                for _ in range(10):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    delta = delta_score_fast(m, old, new)
                    new_s = s + delta
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << (old - 1)) ^ (1 << (best_new - 1))
                    s = best_s
                    improved = True
            if not improved:
                break
        return (lst, m, s)

    # Multi-start SA
    best_score = 0
    best_list = None
    best_mask = None
    
    initial_types = ['interval', 'ap', 'sumset', 'two_ap', 'cantor_like', 'random']
    num_starts = 24
    
    for start_idx in range(num_starts):
        if time_left() < 10:
            break
        
        lst_type = initial_types[start_idx % len(initial_types)]
        initial_lst = gen_initial(lst_type)
        
        # Run SA with varying parameters
        max_iter = 4000 if start_idx < 6 else 2500
        T0 = 5.0 if start_idx < 6 else 3.0
        cooling = 0.9995 if start_idx < 6 else 0.999
        
        result = simulated_annealing(initial_lst, max_iter, T0, cooling)
        
        # Local search on the result
        result = local_search(result[0], result[1], result[2], max_iter=50)
        
        if result[2] > best_score:
            best_score = result[2]
            best_list = result[0][:]
            best_mask = result[1]
            print(f"Start {start_idx} ({lst_type}): {best_score}", file=sys.stderr)
    
    # Final polish: exhaustive single-swap pass with precomputed candidates
    if time_left() > 5 and best_list is not None:
        lst = best_list[:]
        m = best_mask
        s = best_score
        s_set = set(lst)
        
        improved = True
        while improved and time_left() > 2:
            improved = False
            for i in range(N):
                old = lst[i]
                best_delta = 0
                best_new = None
                # Try all possible new values, but skip those in s_set
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    delta = delta_score_fast(m, old, new)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new
                if best_new is not None and best_delta > 0:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << (old - 1)) ^ (1 << (best_new - 1))
                    s += best_delta
                    improved = True
        
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list
# EVOLVE-BLOCK-END
