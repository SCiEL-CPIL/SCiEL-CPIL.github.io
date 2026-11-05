"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def can_add(x, S):
        for a in S:
            if a + x in S:
                return False
        if 2 * x in S:
            return False
        return True
    
    def greedy(elements, order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    def solve_instance(elements, time_budget=35):
        n = len(elements)
        odds = [x for x in elements if x % 2 != 0]
        evens = [x for x in elements if x % 2 == 0]
        elem_set = set(elements)
        
        best = set(odds)
        best_size = len(best)
        
        # Conflict count
        conflict = {}
        for x in elements:
            c = 0
            for y in elements:
                if x + y in elem_set:
                    c += 1
            if 2 * x in elem_set:
                c += 1
            conflict[x] = c
        
        # Build diverse orderings
        orderings = []
        orderings.append(sorted(elements, key=abs))
        orderings.append(sorted(elements, key=abs, reverse=True))
        orderings.append(sorted(elements))
        orderings.append(sorted(elements, reverse=True))
        orderings.append(odds + evens)
        orderings.append(evens + odds)
        orderings.append(sorted(elements, key=lambda x: conflict[x], reverse=True))
        orderings.append(sorted(elements, key=lambda x: conflict[x]))
        
        rng = random.Random(12345)
        for _ in range(40):
            sh = elements[:]
            rng.shuffle(sh)
            orderings.append(sh)
        
        # Also try: sort by |x| but with random tie-breaking
        for _ in range(10):
            sh = sorted(elements, key=lambda x: (abs(x), rng.random()))
            orderings.append(sh)
        
        # Run all orderings
        for order in orderings:
            if time.time() > deadline - 5:
                break
            S = greedy(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Try odds-first with various even orderings
        for order in orderings:
            if time.time() > deadline - 5:
                break
            S = set(odds)
            for x in order:
                if x % 2 == 0 and x not in S and can_add(x, S):
                    S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Local improvement: try to swap one out for two in
        if time.time() < deadline - 8:
            S = set(best)
            S_list = sorted(S)
            candidates = [x for x in elements if x not in S]
            # Limit candidates for speed
            if len(candidates) > 50:
                rng2 = random.Random(999)
                rng2.shuffle(candidates)
                candidates = candidates[:50]
            
            improved = True
            iters = 0
            while improved and iters < 10:
                improved = False
                iters += 1
                S_list = sorted(S)
                for a in S_list:
                    if time.time() > deadline - 3:
                        break
                    S_trial = S - {a}
                    for i in range(len(candidates)):
                        b = candidates[i]
                        if b in S_trial:
                            continue
                        for j in range(i+1, len(candidates)):
                            c = candidates[j]
                            if c in S_trial:
                                continue
                            if b + c in S_trial:
                                continue
                            if 2*b in S_trial or 2*c in S_trial:
                                continue
                            ok = True
                            for d in S_trial:
                                if d + b in S_trial or d + c in S_trial:
                                    ok = False
                                    break
                            if ok:
                                S = S_trial | {b, c}
                                improved = True
                                break
                        if improved:
                            break
                    if improved or time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        return sorted(best)
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        result = solve_instance(elements)
        results.append(result)
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            break
    
    return results
# EVOLVE-BLOCK-END
