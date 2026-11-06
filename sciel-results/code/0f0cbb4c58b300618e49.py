# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(Sset, x, S):
        """Check if x can be added to S without violating sum-free property."""
        # Need: for all a in S, a + x not in S, x + a not in S, and x + x not in S
        # Also x + x not in S
        if x + x in Sset:
            return False
        for a in S:
            if a + x in Sset:
                return False
            if x + a in Sset:
                return False
        return True

    def greedy_build(elements, order_key, Sset=None):
        """Greedy: add elements in given order if they don't violate sum-free."""
        S = []
        Sset = set()
        for x in sorted(elements, key=order_key):
            if can_add(Sset, x, S):
                S.append(x)
                Sset.add(x)
        return S

    def local_search(S, elements, max_iters=5000, time_limit=2.0):
        """Local search: try to add excluded elements, or swap to improve size."""
        S = list(S)
        Sset = set(S)
        excluded = [x for x in elements if x not in Sset]
        
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding each excluded element
            for i, x in enumerate(excluded):
                if can_add(Sset, x, S):
                    S.append(x)
                    Sset.add(x)
                    excluded.pop(i)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove one from S, add one from excluded (net 0 or +1)
            # Actually for size improvement, we need net +1, which means just adding.
            # For quality improvement (allowing more additions later), try 1-for-1 swaps
            # that might enable further additions. But simpler: just try to add.
            # If no single addition works, try 1-for-1 swap that might help later.
            # For now, just try additions. If stuck, try random restarts.
        return S

    def multi_strategy_local_search(S, elements, time_limit=3.0):
        """More aggressive local search with random restarts and perturbation."""
        best = list(S)
        best_set = set(S)
        start = time.time()
        
        while time.time() - start < time_limit:
            S = list(best)
            Sset = set(S)
            excluded = [x for x in elements if x not in Sset]
            
            # Greedy add
            changed = True
            while changed:
                changed = False
                for i, x in enumerate(excluded):
                    if can_add(Sset, x, S):
                        S.append(x)
                        Sset.add(x)
                        excluded.pop(i)
                        changed = True
                        break
            
            if len(S) > len(best):
                best = list(S)
                best_set = set(S)
            
            # Perturbation: remove a random element, try to add others
            if S and excluded:
                S2 = list(S)
                S2set = set(S)
                # Remove 1-2 random elements
                k = min(2, len(S2))
                removed = random.sample(S2, k)
                for r in removed:
                    S2.remove(r)
                    S2set.discard(r)
                excluded2 = [x for x in elements if x not in S2set]
                
                # Greedy add on perturbed set
                changed = True
                while changed:
                    changed = False
                    for i, x in enumerate(excluded2):
                        if can_add(S2set, x, S2):
                            S2.append(x)
                            S2set.add(x)
                            excluded2.pop(i)
                            changed = True
                            break
                
                if len(S2) > len(best):
                    best = list(S2)
                    best_set = set(S2)
        
        return best

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > DEADLINE:
            # Emergency: return odd numbers as fallback
            elems = instance(size, seed)
            results.append([x for x in elems if x % 2])
            continue
        
        elements = instance(size, seed)
        
        # Strategy 1: Greedy by absolute value (large first)
        s1 = greedy_build(elements, key=lambda x: -abs(x))
        # Strategy 2: Greedy by value (large positive first)
        s2 = greedy_build(elements, key=lambda x: -x)
        # Strategy 3: Greedy by value (small negative first)
        s3 = greedy_build(elements, key=lambda x: x)
        # Strategy 4: Greedy by absolute value (small first)
        s4 = greedy_build(elements, key=lambda x: abs(x))
        # Strategy 5: Greedy by parity then value
        s5 = greedy_build(elements, key=lambda x: (x % 2 == 0, -abs(x)))
        # Strategy 6: Random order (multiple)
        rng = random.Random(seed + 999)
        s6 = list(elements)
        rng.shuffle(s6)
        s6 = greedy_build(s6, key=lambda x: 0)  # preserve order
        
        # Strategy 7: Odds only (baseline)
        s7 = [x for x in elements if x % 2]
        
        candidates = [s1, s2, s3, s4, s5, s6, s7]
        
        # Local search on each candidate
        best = max(candidates, key=len)
        
        # Apply local search to top candidates
        candidates_sorted = sorted(candidates, key=len, reverse=True)
        
        for cand in candidates_sorted[:4]:  # top 4
            if time.time() > DEADLINE - 5:
                break
            ls_result = multi_strategy_local_search(cand, elements, time_limit=3.0)
            if len(ls_result) > len(best):
                best = ls_result
        
        # Additional local search iterations if time permits
        if time.time() < DEADLINE - 10:
            # Try more random restarts
            for _ in range(5):
                if time.time() > DEADLINE - 5:
                    break
                random.shuffle(elements)
                rnd = greedy_build(elements, key=lambda x: 0)
                ls_result = multi_strategy_local_search(rnd, elements, time_limit=2.0)
                if len(ls_result) > len(best):
                    best = ls_result
        
        # Final verification
        if not is_sum_free(best):
            # Fallback to odds
            best = [x for x in elements if x % 2]
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
