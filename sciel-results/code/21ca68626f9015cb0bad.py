import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    n = 256
    
    def _try_add(S, S_set, cand):
        """Check if cand can be added to S (full three-case check)."""
        # Case 1: cand plays role of 'a' (smallest in triple)
        # Need: no b,c in S with b > cand, c >= b, cand | (b+c)
        larger = [x for x in S if x > cand]
        for i in range(len(larger)):
            b = larger[i]
            for j in range(i, len(larger)):
                c = larger[j]
                if (b + c) % cand == 0:
                    return False
        
        # Case 2: cand plays role of 'b' (middle in triple)
        # Need: no a,c in S with a < cand, c >= cand, a | (cand + c)
        smaller = [x for x in S if x < cand]
        geq = [x for x in S if x >= cand]
        for a in smaller:
            for c in geq:
                if (cand + c) % a == 0:
                    return False
        
        # Case 3: cand plays role of 'c' (largest in triple)
        # Need: no a,b in S with a < b <= cand, a | (b + cand)
        for a in S:
            if a >= cand:
                continue
            for b in S:
                if a < b <= cand and (b + cand) % a == 0:
                    return False
        return True

    def _local_search(S, S_set, deadline):
        """Try to add skipped elements in descending then ascending order."""
        skipped = set(range(1, n + 1)) - S_set
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cand in sorted(skipped, reverse=True):
                if time.time() > deadline:
                    break
                if _try_add(S, S_set, cand):
                    S.append(cand)
                    S_set.add(cand)
                    skipped.discard(cand)
                    improved = True
        
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for cand in sorted(skipped):
                    if time.time() > deadline:
                        break
                    if _try_add(S, S_set, cand):
                        S.append(cand)
                        S_set.add(cand)
                        skipped.discard(cand)
                        improved = True
        return S, S_set

    def _greedy(order, deadline):
        S = []
        S_set = set()
        for cand in order:
            if time.time() > deadline:
                break
            if _try_add(S, S_set, cand):
                S.append(cand)
                S_set.add(cand)
        return S, S_set

    # Initial run: Descending order (known good)
    best_S, best_S_set = _greedy(range(n, 0, -1), deadline)
    best_S, best_S_set = _local_search(best_S, best_S_set, deadline)
    best_size = len(best_S)
    
    # Multi-start with random orders
    if time.time() < deadline - 5: # Reserve 5s for final checks
        while time.time() < deadline - 5:
            # Random permutation
            order = list(range(1, n + 1))
            random.shuffle(order)
            
            # Optional: Bias towards larger numbers? 
            # Or just pure random.
            
            S, S_set = _greedy(order, deadline)
            S, S_set = _local_search(S, S_set, deadline)
            
            if len(S) > best_size:
                best_S = S
                best_S_set = S_set
                best_size = len(S)
                
    return best_S
