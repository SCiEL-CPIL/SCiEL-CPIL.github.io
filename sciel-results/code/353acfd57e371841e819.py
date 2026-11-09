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

    def delta_score(mask, old_val, new_val):
        """Compute change in score when replacing old_val with new_val in mask."""
        old_bit = 1 << (old_val - 1)
        new_bit = 1 << (new_val - 1)
        new_mask = mask ^ old_bit ^ new_bit
        
        # Only check d values where the AP status could change
        # A value v participates in a 3-AP with difference d if:
        #   v is first: v, v+d, v+2d all in mask
        #   v is middle: v-d, v, v+d all in mask
        #   v is last: v-2d, v-d, v all in mask
        affected_ds = set()
        
        for d in range(1, MAX_D + 1):
            # old_val as first, middle, or last
            if old_val + 2*d <= MAX_VAL or old_val - d >= 1 or old_val - 2*d >= 1:
                affected_ds.add(d)
            # new_val as first, middle, or last
            if new_val + 2*d <= MAX_VAL or new_val - d >= 1 or new_val - 2*d >= 1:
                affected_ds.add(d)
        
        delta = 0
        for d in affected_ds:
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
            
            delta = delta_score(m, old, new)
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
                    delta = delta_score(m, old, new)
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
    
    initial_types = ['interval', 'ap', 'sumset', 'random']
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time_left() < 10:
            break
        
        lst_type = initial_types[start_idx % len(initial_types)]
        initial_lst = gen_initial(lst_type)
        
        # Run SA with varying parameters
        max_iter = 3000 if start_idx < 5 else 2000
        T0 = 5.0 if start_idx < 5 else 3.0
        cooling = 0.9995 if start_idx < 5 else 0.999
        
        result = simulated_annealing(initial_lst, max_iter, T0, cooling)
        
        # Local search on the result
        result = local_search(result[0], result[1], result[2], max_iter=50)
        
        if result[2] > best_score:
            best_score = result[2]
            best_list = result[0][:]
            best_mask = result[1]
            print(f"Start {start_idx}: {best_score}", file=sys.stderr)
    
    # Final polish: exhaustive single-swap pass
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
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    delta = delta_score(m, old, new)
                    if delta > 0:
                        s_set.discard(old)
                        s_set.add(new)
                        lst[i] = new
                        m ^= (1 << (old - 1)) ^ (1 << (new - 1))
                        s += delta
                        improved = True
                        break
        
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list
# EVOLVE-BLOCK-END
