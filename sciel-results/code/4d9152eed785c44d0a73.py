"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 85  # seconds, leave 5s margin
    
    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT
    
    N = 512
    
    # Precompute divisor counts
    div_count = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            div_count[m] += 1
    
    # LCM helper
    def lcm(a, b):
        g = math.gcd(a, b)
        return a // g * b
    
    def can_add(x, s):
        for y in s:
            if lcm(x, y) > N:
                return False
        return True
    
    def max_lcm_with(x, s):
        mx = 1
        for y in s:
            l = lcm(x, y)
            if l > N:
                return N + 1
            if l > mx:
                mx = l
        return mx
    
    def get_divisors(m):
        divs = set()
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.add(i)
                if i != m // i:
                    divs.add(m // i)
            i += 1
        return divs
    
    # Greedy construction from a seed set
    def greedy_build(seed_set):
        current = set(seed_set)
        current.add(1)
        candidates = [x for x in range(2, N + 1) if x not in current]
        
        added_any = True
        while added_any and not check_deadline():
            added_any = False
            best_c = None
            best_score = N + 1
            for x in candidates:
                if x in current:
                    continue
                s = max_lcm_with(x, current)
                if s < best_score:
                    best_score = s
                    best_c = x
            if best_c is not None and best_score <= N:
                current.add(best_c)
                candidates.remove(best_c)
                added_any = True
        return current
    
    # Local search: 1-for-1 swaps
    def local_search_1for1(current, max_iters=50):
        current = set(current)
        for _ in range(max_iters):
            if check_deadline():
                break
            improved = False
            elems = sorted(current)
            for el in elems:
                if el == 1:
                    continue
                if check_deadline():
                    break
                trial = set(current)
                trial.discard(el)
                # Greedily add
                new_cands = [x for x in range(2, N + 1) if x not in trial]
                added = True
                while added and not check_deadline():
                    added = False
                    best_c = None
                    best_s = N + 1
                    for x in new_cands:
                        if x in trial:
                            continue
                        s = max_lcm_with(x, trial)
                        if s < best_s:
                            best_s = s
                            best_c = x
                    if best_c is not None and best_s <= N:
                        trial.add(best_c)
                        new_cands.remove(best_c)
                        added = True
                if len(trial) > len(current):
                    current = trial
                    improved = True
                    break
            if not improved:
                break
        return current
    
    # Local search: 2-for-3 swaps
    def local_search_2for3(current, max_pairs=200):
        current = set(current)
        pairs_checked = 0
        improved = True
        while improved and not check_deadline() and pairs_checked < max_pairs:
            improved = False
            elems = sorted(current)
            for i in range(len(elems)):
                if check_deadline() or pairs_checked >= max_pairs:
                    break
                for j in range(i + 1, len(elems)):
                    if check_deadline() or pairs_checked >= max_pairs:
                        break
                    el1, el2 = elems[i], elems[j]
                    if el1 == 1 or el2 == 1:
                        continue
                    pairs_checked += 1
                    trial = set(current)
                    trial.discard(el1)
                    trial.discard(el2)
                    # Greedily add up to 3 new elements
                    new_cands = [x for x in range(2, N + 1) if x not in trial]
                    added_count = 0
                    added = True
                    while added and added_count < 3 and not check_deadline():
                        added = False
                        best_c = None
                        best_s = N + 1
                        for x in new_cands:
                            if x in trial:
                                continue
                            s = max_lcm_with(x, trial)
                            if s < best_s:
                                best_s = s
                                best_c = x
                        if best_c is not None and best_s <= N:
                            trial.add(best_c)
                            new_cands.remove(best_c)
                            added = True
                            added_count += 1
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        break
                if improved:
                    break
        return current
    
    # Strategy 1: Start with divisors of the number with most divisors
    best_m = max(range(1, N + 1), key=lambda x: div_count[x])
    seed1 = get_divisors(best_m)
    set1 = greedy_build(seed1)
    set1 = local_search_1for1(set1)
    set1 = local_search_2for3(set1)
    
    best_set = set(set1)
    best_size = len(best_set)
    
    # Strategy 2: Try other highly-divisor-count seeds
    nums_by_divs = sorted(range(1, N + 1), key=lambda x: -div_count[x])
    tried_seeds = {best_m}
    for m in nums_by_divs[:15]:
        if check_deadline():
            break
        if m in tried_seeds:
            continue
        tried_seeds.add(m)
        seed = get_divisors(m)
        s = greedy_build(seed)
        s = local_search_1for1(s, max_iters=30)
        if len(s) > best_size:
            best_set = set(s)
            best_size = len(s)
    
    # Strategy 3: Random restarts
    if not check_deadline():
        random.seed(42)
        for restart in range(8):
            if check_deadline():
                break
            m = random.randint(1, N)
            seed = get_divisors(m)
            # Add a few random compatible numbers
            for _ in range(3):
                x = random.randint(1, N)
                if can_add(x, seed):
                    seed.add(x)
            s = greedy_build(seed)
            s = local_search_1for1(s, max_iters=20)
            if len(s) > best_size:
                best_set = set(s)
                best_size = len(s)
    
    # Final: more aggressive local search on best
    if not check_deadline():
        best_set = local_search_1for1(best_set, max_iters=40)
        best_set = local_search_2for3(best_set, max_pairs=300)
    
    # Final verification
    final_list = sorted(best_set)
    valid = len(final_list) > 0
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if lcm(final_list[i], final_list[j]) > N:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return list(range(1, int(math.isqrt(N)) + 1))
    
    return final_list

# EVOLVE-BLOCK-END
