import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255

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

    def delta_score(mask, old_bit, new_bit):
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, MAX_D + 1):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta, new_mask

    def delta_score_fast(mask, old_bit, new_bit):
        """Faster delta: only check d's where the change could matter."""
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, MAX_D + 1):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta, new_mask

    # --- Spectral / Fourier-based construction ---
    # Key insight: the number of distinct d with a 3-AP is related to the 
    # autocorrelation of the set. We can use a continuous relaxation where
    # we assign weights to positions and maximize a smooth surrogate.
    
    def spectral_construct():
        """
        Use a continuous relaxation: assign a real weight w[x] in [0,1] to each
        position x. The objective is to maximize the number of d where the
        "soft" 3-AP count is above threshold. We use a smooth surrogate:
        For each d, soft_count(d) = sum_x w[x] * w[x+d] * w[x+2d]
        We want to maximize the number of d where soft_count(d) > 0.5.
        Use a smooth approximation: sum_d sigmoid(alpha * (soft_count(d) - 0.5))
        
        Then round to get a 32-element set.
        """
        alpha = 8.0
        # Initialize weights
        w = [0.0] * (MAX_VAL + 1)
        # Start with a structured initialization
        random.seed(123)
        
        # Try multiple initializations
        best_w = None
        best_obj = -1e9
        
        for init_trial in range(5):
            w = [0.0] * (MAX_VAL + 1)
            if init_trial == 0:
                # Uniform-ish
                for x in range(1, MAX_VAL + 1):
                    w[x] = 0.5
            elif init_trial == 1:
                # Base-4 Cantor-like
                for x in range(1, MAX_VAL + 1):
                    # Check if x has only digits 0,1 in base 4
                    y = x
                    ok = True
                    while y > 0:
                        if y % 4 > 1:
                            ok = False
                            break
                        y //= 4
                    w[x] = 1.0 if ok else 0.0
            elif init_trial == 2:
                # AP-like
                for x in range(1, MAX_VAL + 1):
                    w[x] = 1.0 if (x - 1) % 8 < 4 else 0.0
            elif init_trial == 3:
                # Random
                for x in range(1, MAX_VAL + 1):
                    w[x] = random.random()
            else:
                # Two-scale structure
                for x in range(1, MAX_VAL + 1):
                    w[x] = 1.0 if (x - 1) % 16 < 6 else 0.0
            
            # Gradient ascent on the smooth objective
            lr = 0.05
            for step in range(200):
                if time_left() < 30:
                    break
                
                # Compute objective and gradient
                obj = 0.0
                grad = [0.0] * (MAX_VAL + 1)
                
                for d in range(1, MAX_D + 1):
                    soft_count = 0.0
                    for a in range(1, MAX_VAL - 2 * d + 1):
                        prod = w[a] * w[a + d] * w[a + 2 * d]
                        soft_count += prod
                        # Gradient contribution
                        grad[a] += w[a + d] * w[a + 2 * d]
                        grad[a + d] += w[a] * w[a + 2 * d]
                        grad[a + 2 * d] += w[a] * w[a + d]
                    
                    # Smooth sigmoid surrogate
                    z = alpha * (soft_count - 0.5)
                    if z > 50:
                        sig = 1.0
                        dsig = 0.0
                    elif z < -50:
                        sig = 0.0
                        dsig = 0.0
                    else:
                        sig = 1.0 / (1.0 + math.exp(-z))
                        dsig = alpha * sig * (1.0 - sig)
                    
                    obj += sig
                    # Gradient of sigmoid w.r.t. soft_count
                    for a in range(1, MAX_VAL - 2 * d + 1):
                        grad[a] *= dsig
                        grad[a + d] *= dsig
                        grad[a + 2 * d] *= dsig
                
                # Update weights
                for x in range(1, MAX_VAL + 1):
                    w[x] += lr * grad[x]
                    w[x] = max(0.0, min(1.0, w[x]))
                
                if obj > best_obj:
                    best_obj = obj
                    best_w = w[:]
            
            if time_left() < 30:
                break
        
        if best_w is None:
            return None
        
        # Round: select top 32 positions by weight
        weighted = [(best_w[x], x) for x in range(1, MAX_VAL + 1)]
        weighted.sort(reverse=True)
        lst = [x for _, x in weighted[:N]]
        m = list_to_mask(lst)
        s = score_mask(m)
        return (lst, m, s)

    # --- Targeted AP-completing local search ---
    def ap_completing_search(individual, time_budget=5.0):
        """Aggressively try to complete 3-APs by replacing elements."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            
            # For each d, find all nearly-complete 3-APs
            for d in range(1, MAX_D + 1):
                if time.time() - start >= time_budget:
                    break
                
                # Find a where {a, a+d} in set but a+2d not
                for a in range(1, MAX_VAL - 2 * d + 1):
                    if time.time() - start >= time_budget:
                        break
                    if a in s_set and (a + d) in s_set and (a + 2 * d) not in s_set:
                        new_val = a + 2 * d
                        # Find best element to replace
                        best_idx = -1
                        best_delta = 0
                        for idx in range(N):
                            if lst[idx] == new_val:
                                continue
                            old_bit = 1 << (lst[idx] - 1)
                            new_bit = 1 << (new_val - 1)
                            delta, _ = delta_score(m, old_bit, new_bit)
                            if delta > best_delta:
                                best_delta = delta
                                best_idx = idx
                        if best_idx >= 0:
                            old = lst[best_idx]
                            s_set.discard(old)
                            s_set.add(new_val)
                            lst[best_idx] = new_val
                            m ^= (1 << (old - 1)) ^ (1 << (new_val - 1))
                            s += best_delta
                            improved = True
                            break  # restart d-loop
                    
                    elif a in s_set and (a + 2 * d) in s_set and (a + d) not in s_set:
                        new_val = a + d
                        best_idx = -1
                        best_delta = 0
                        for idx in range(N):
                            if lst[idx] == new_val:
                                continue
                            old_bit = 1 << (lst[idx] - 1)
                            new_bit = 1 << (new_val - 1)
                            delta, _ = delta_score(m, old_bit, new_bit)
                            if delta > best_delta:
                                best_delta = delta
                                best_idx = idx
                        if best_idx >= 0:
                            old = lst[best_idx]
                            s_set.discard(old)
                            s_set.add(new_val)
                            lst[best_idx] = new_val
                            m ^= (1 << (old - 1)) ^ (1 << (new_val - 1))
                            s += best_delta
                            improved = True
                            break
        
        return (lst, m, s)

    # --- Exhaustive single-swap ---
    def exhaustive_swap(individual, time_budget=15.0):
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for i in range(N):
                if time.time() - start >= time_budget:
                    break
                old = lst[i]
                old_bit = 1 << (old - 1)
                best_new = old
                best_delta = 0
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    new_bit = 1 << (new - 1)
                    delta, _ = delta_score(m, old_bit, new_bit)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= old_bit ^ (1 << (best_new - 1))
                    s += best_delta
                    improved = True
        
        return (lst, m, s)

    # --- Simulated annealing ---
    def simulated_annealing(individual, time_budget=15.0):
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        best_lst = lst[:]
        best_m = m
        best_s = s
        start = time.time()
        T = 8.0
        cooling = 0.985
        
        while time.time() - start < time_budget:
            i = random.randint(0, N - 1)
            old = lst[i]
            old_bit = 1 << (old - 1)
            new = random.randint(1, MAX_VAL)
            if new == old or new in s_set:
                continue
            new_bit = 1 << (new - 1)
            delta, new_m = delta_score(m, old_bit, new_bit)
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                s_set.discard(old)
                s_set.add(new)
                lst[i] = new
                m = new_m
                s += delta
                if s > best_s:
                    best_s = s
                    best_lst = lst[:]
                    best_m = m
            T *= cooling
        
        return (best_lst, best_m, best_s)

    # --- Structured seed generation ---
    def gen_seeds():
        random.seed(42)
        solutions = []
        seen = set()
        
        def add(lst):
            if len(lst) != N:
                return
            key = tuple(sorted(lst))
            if key in seen:
                return
            seen.add(key)
            m = list_to_mask(lst)
            s = score_mask(m)
            solutions.append((lst, m, s))
        
        # 1. Base-4 Cantor sets
        for digits in [[0, 1], [0, 2], [1, 3], [0, 1, 2], [1, 2, 3]]:
            candidates = []
            mp = 0
            while 4 ** mp <= MAX_VAL:
                mp += 1
            def gen(pos, cur):
                if pos == mp:
                    if 1 <= cur <= MAX_VAL:
                        candidates.append(cur)
                    return
                for d in digits:
                    gen(pos + 1, cur + d * (4 ** pos))
            gen(0, 0)
            candidates = sorted(set(candidates))
            if len(candidates) >= N:
                for _ in range(3):
                    add(random.sample(candidates, N))
        
        # 2. Arithmetic progressions
        for d in range(3, 18):
            if 1 + (N - 1) * d <= MAX_VAL:
                add([1 + i * d for i in range(N)])
        
        # 3. Residue classes
        for mod in range(3, 15):
            for start in range(1, mod + 1):
                lst = []
                for v in range(start, MAX_VAL + 1, mod):
                    lst.append(v)
                    if len(lst) == N:
                        break
                if len(lst) == N:
                    add(lst)
        
        # 4. Random
        for _ in range(50):
            add(random.sample(range(1, MAX_VAL + 1), N))
        
        # 5. Structured: AP with perturbations
        for d in range(4, 14):
            base = [1 + i * d for i in range(N)]
            for _ in range(10):
                lst = base[:]
                for _ in range(random.randint(2, 5)):
                    idx = random.randint(0, N - 1)
                    lst[idx] = random.randint(1, MAX_VAL)
                seen_set = set()
                unique = []
                for x in lst:
                    if x not in seen_set and 1 <= x <= MAX_VAL:
                        seen_set.add(x)
                        unique.append(x)
                while len(unique) < N:
                    x = random.randint(1, MAX_VAL)
                    if x not in seen_set:
                        seen_set.add(x)
                        unique.append(x)
                add(unique[:N])
        
        # 6. Union of two APs
        for d1 in range(3, 12):
            for d2 in range(3, 12):
                if d1 == d2:
                    continue
                for a1 in range(1, 15):
                    for a2 in range(1, 15):
                        s1 = set(a1 + i * d1 for i in range(16))
                        s2 = set(a2 + i * d2 for i in range(16))
                        comb = {x for x in s1 | s2 if 1 <= x <= MAX_VAL}
                        if len(comb) >= N:
                            add(random.sample(sorted(comb), N))
        
        # 7. Sumset constructions
        for mult in [2, 3, 4]:
            for _ in range(30):
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
                    add(random.sample(sorted(S), N))
        
        return solutions

    # === Main ===
    random.seed(42)
    
    # Step 1: Generate seeds
    population = gen_seeds()
    population.sort(key=lambda x: x[2], reverse=True)
    
    best_score = population[0][2]
    best_list = population[0][0][:]
    best_mask = population[0][1]
    print(f"Initial best: {best_score}", file=sys.stderr)
    
    # Step 2: Spectral construction
    if time_left() > 40:
        spec_result = spectral_construct()
        if spec_result and spec_result[2] > best_score:
            best_score = spec_result[2]
            best_list = spec_result[0][:]
            best_mask = spec_result[1]
            print(f"Spectral: {best_score}", file=sys.stderr)
        elif spec_result:
            population.append(spec_result)
            population.sort(key=lambda x: x[2], reverse=True)
    
    # Step 3: AP-completing search on top candidates
    top_k = min(15, len(population))
    for idx in range(top_k):
        if time_left() < 20:
            break
        t_budget = min(4.0, time_left() * 0.05)
        result = ap_completing_search(population[idx], time_budget=t_budget)
        if result[2] > best_score:
            best_score = result[2]
            best_list = result[0][:]
            best_mask = result[1]
            print(f"AP-complete improved: {best_score}", file=sys.stderr)
        population[idx] = result
    
    population.sort(key=lambda x: x[2], reverse=True)
    
    # Step 4: SA on top candidates
    for idx in range(min(8, len(population))):
        if time_left() < 25:
            break
        t_budget = min(12.0, time_left() * 0.1)
        sa_result = simulated_annealing(population[idx], time_budget=t_budget)
        if sa_result[2] > best_score:
            best_score = sa_result[2]
            best_list = sa_result[0][:]
            best_mask = sa_result[1]
            print(f"SA improved: {best_score}", file=sys.stderr)
        population[idx] = sa_result
    
    population.sort(key=lambda x: x[2], reverse=True)
    
    # Step 5: Exhaustive swap on best
    if time_left() > 10:
        t_budget = min(time_left() * 0.25, 20.0)
        es_result = exhaustive_swap((best_list, best_mask, best_score), time_budget=t_budget)
        if es_result[2] > best_score:
            best_score = es_result[2]
            best_list = es_result[0][:]
            best_mask = es_result[1]
            print(f"Exhaustive improved: {best_score}", file=sys.stderr)
    
    # Step 6: Final AP-completing polish
    if time_left() > 8:
        t_budget = min(time_left() * 0.3, 10.0)
        ap_result = ap_completing_search((best_list, best_mask, best_score), time_budget=t_budget)
        if ap_result[2] > best_score:
            best_score = ap_result[2]
            best_list = ap_result[0][:]
            best_mask = ap_result[1]
    
    # Step 7: Final exhaustive swap
    if time_left() > 5:
        t_budget = min(time_left() * 0.4, 12.0)
        es_result = exhaustive_swap((best_list, best_mask, best_score), time_budget=t_budget)
        if es_result[2] > best_score:
            best_score = es_result[2]
            best_list = es_result[0][:]
            best_mask = es_result[1]
    
    print(f"Final score: {best_score}", file=sys.stderr)
    
    # Verify
    result = mask_to_list(best_mask)
    assert len(result) == N, f"Expected {N} elements, got {len(result)}"
    assert len(set(result)) == N, "Duplicate elements"
    assert all(1 <= x <= MAX_VAL for x in result), "Out of range"
    
    return result

# EVOLVE-BLOCK-END
