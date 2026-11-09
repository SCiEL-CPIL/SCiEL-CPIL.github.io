"""Stronger sum-free subset search with deadline guard."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t_start = time.time()
    DEADLINE = 110.0  # safety margin under the 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(y, S, S0_set):
        """Check if y can be added to S without violating sum-free property."""
        if y in S:
            return False
        # Condition 3: 2y not in S
        if 2 * y in S:
            return False
        # Condition 1: no a, b in S with a+b=y
        for a in S:
            if (y - a) in S:
                return False
        # Condition 2: no a in S with a+y in S
        for a in S:
            if (a + y) in S:
                return False
        return True

    def local_search(S0, S_init, deadline, max_passes=100):
        """Local search starting from S_init. S0 is the full candidate set (list)."""
        S0_set = set(S0)
        S = set(S_init)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding every element not in S
            for y in S0:
                if y in S:
                    continue
                if can_add(y, S, S0_set):
                    S.add(y)
                    improved = True
            
            # Try removing elements to allow more additions
            for x in list(S):
                if time.time() > deadline:
                    break
                S2 = S - {x}
                gain = 0
                for y in S0:
                    if y in S2:
                        continue
                    if can_add(y, S2, S0_set):
                        # Only count if it wasn't already addable to S
                        if not can_add(y, S, S0_set):
                            gain += 1
                if gain > 1:
                    S = S2
                    improved = True
            
            if not improved:
                break
        return S

    def sum_free(S):
        S = set(S)
        for a in S:
            for b in S:
                if a + b in S:
                    return False
        return True

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S0 = instance(size, seed)
        S0_set = set(S0)
        
        # Multiple starting points
        odd = [x for x in S0 if x % 2]
        top = S0[len(S0)//2:]
        bottom = S0[:len(S0)//2]
        empty = []
        
        best = None
        best_len = -1
        
        for init in [odd, top, bottom, empty]:
            if time.time() > DEADLINE - 10:
                break
            cand = local_search(S0, init, DEADLINE - 5)
            if len(cand) > best_len:
                best_len = len(cand)
                best = cand
        
        if best is None:
            best = set(odd)
        results.append(sorted(best))
    
    return results
# EVOLVE-BLOCK-END
