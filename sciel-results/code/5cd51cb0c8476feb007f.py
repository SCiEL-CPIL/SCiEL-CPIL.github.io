import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # Strategy 1: Start with Odds
        # Odds are sum-free because odd+odd=even.
        s1 = set(x for x in values if x % 2 != 0)
        c1 = set(x for x in values if x % 2 == 0)
        
        # Strategy 2: Start with Large Absolute Values
        # If |x| > 5000, then |x+y| > 10000 for x,y same sign?
        # Actually if x,y > 5000, x+y > 10000.
        # If x,y < -5000, x+y < -10000.
        # So {x : x > 5000} is sum-free. {x : x < -5000} is sum-free.
        # Union is sum-free?
        # a > 5000, b < -5000. a+b can be anything in (-10000, 10000).
        # So Union is NOT necessarily sum-free.
        # But {x : |x| > 5000} is not sum-free if we mix signs.
        # However, {x : x > 5000} is sum-free.
        # {x : x < -5000} is sum-free.
        # Let's try starting with Positives > 5000 and Negatives < -5000 separately?
        # Or just start with one side?
        # Let's try starting with all x such that x > 5000.
        s2_pos = set(x for x in values if x > 5000)
        s2_neg = set(x for x in values if x < -5000)
        
        # Strategy 3: Start with all x such that x % 3 != 0?
        # If a,b not div by 3, a+b can be div by 3.
        # 1+2=3. So no.
        
        # Let's stick to S1 (Odds) and S2 (Large Pos) and S3 (Large Neg)
        # Actually, S1 is usually best because it captures ~50% of elements.
        # S2 captures ~25% (half of the 50% large ones).
        
        def improve(s, c, v_set):
            """
            Local search: add elements from c to s if possible, or swap.
            s: current set (sum-free)
            c: candidates
            v_set: all values
            """
            s = set(s)
            c = set(c)
            
            # Precompute conflicts for initial state
            # conflict[e] = set of elements in s that conflict with e
            # Conflict if e/2 in s (and e/2 is int) OR (e-a in s for some a in s)
            
            def get_conflicts(e, s):
                conflicts = set()
                # Check e/2
                if e % 2 == 0:
                    half = e // 2
                    if half in s:
                        conflicts.add(half)
                # Check pairs a+b=e
                # Iterate over s
                for a in s:
                    b = e - a
                    if b in s:
                        conflicts.add(a)
                        conflicts.add(b)
                return conflicts

            # Add free elements
            improved = True
            while improved and time.time() < DEADLINE:
                improved = False
                # Sort candidates by conflict count (ascending)
                # We need to recompute conflicts for all c each time?
                # Or just pick one with 0 conflicts.
                
                # Find any e in c with 0 conflicts
                free_e = None
                # To make it faster, we can just iterate
                for e in c:
                    if time.time() > DEADLINE:
                        break
                    conf = get_conflicts(e, s)
                    if len(conf) == 0:
                        free_e = e
                        break
                
                if free_e is not None:
                    s.add(free_e)
                    c.discard(free_e)
                    improved = True
                    continue
                
                # If no free elements, try swaps (conflict count 1)
                # This is more expensive.
                # Pick e with min conflicts > 0
                best_e = None
                best_conf = None
                min_conf_size = float('inf')
                
                # Limit search for swaps to avoid slowing down
                # If s is large, this is O(|c| * |s|)
                for e in c:
                    if time.time() > DEADLINE:
                        break
                    conf = get_conflicts(e, s)
                    if len(conf) < min_conf_size:
                        min_conf_size = len(conf)
                        best_e = e
                        best_conf = conf
                        if len(conf) == 1:
                            break # Can't do better than 1
                
                if best_e is not None and min_conf_size == 1:
                    # Swap: remove the single conflict, add best_e
                    to_remove = list(best_conf)[0]
                    s.remove(to_remove)
                    s.add(best_e)
                    c.discard(best_e)
                    c.add(to_remove)
                    improved = True
                    continue
                
                # If no swaps with 1 conflict, try 2 conflicts?
                # Net gain -1. Unlikely to help unless it unlocks many free adds.
                # We'll stop here for now.
                
            return list(s)

        # Run strategies
        results = []
        
        # Strategy 1: Odds
        if time.time() < DEADLINE:
            res1 = improve(s1, c1, v_set)
            results.append(res1)
        
        # Strategy 2: Large Positives
        if time.time() < DEADLINE:
            c2 = set(x for x in values if x <= 5000)
            res2 = improve(s2_pos, c2, v_set)
            results.append(res2)
            
        # Strategy 3: Large Negatives
        if time.time() < DEADLINE:
            c3 = set(x for x in values if x >= -5000)
            res3 = improve(s2_neg, c3, v_set)
            results.append(res3)
            
        # Strategy 4: Random Greedy (Backup)
        if time.time() < DEADLINE:
            # Greedy by degree
            # Build conflict graph approx
            # Just do a few random restarts of greedy add
            best_greedy = []
            for _ in range(10):
                if time.time() > DEADLINE:
                    break
                perm = list(values)
                random.shuffle(perm)
                s_g = set()
                for x in perm:
                    # Check if x conflicts with s_g
                    ok = True
                    if x % 2 == 0 and (x//2) in s_g:
                        ok = False
                    if ok:
                        for a in s_g:
                            if x + a in s_g:
                                ok = False
                                break
                    if ok:
                        s_g.add(x)
                if len(s_g) > len(best_greedy):
                    best_greedy = list(s_g)
            results.append(best_greedy)

        # Pick the best valid result
        best = []
        for res in results:
            if is_sum_free(res) and len(res) > len(best):
                best = res
        
        # Final local search on best
        if time.time() < DEADLINE:
             # Re-run improve on best to see if we can squeeze more
             # We need c = v_set - set(best)
             c_final = v_set - set(best)
             best = improve(best, c_final, v_set)
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        res = solve_instance(vals)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to odds
            res = [x for x in vals if x % 2]
            
        results.append(res)
        
    return results
