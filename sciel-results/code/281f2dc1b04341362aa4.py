"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _try_add(S, cand, n):
    """Check if cand can be added to S using vectorized NumPy checks."""
    S_arr = np.array(sorted(S), dtype=np.int64)
    if len(S_arr) == 0:
        return True
    
    # Case 1: cand plays role of 'a' (smallest in triple)
    # Need: no b,c in S with b > cand, c >= b, cand | (b+c)
    larger = S_arr[S_arr > cand]
    if len(larger) >= 2:
        # For each pair (b, c) with b <= c in larger, check if (b+c) % cand == 0
        # Vectorized: compute all b+c for b<=c pairs
        # This is O(m^2) where m = len(larger), but we can vectorize the mod check
        for i in range(len(larger)):
            b = larger[i]
            # c ranges from larger[i] to larger[-1]
            cs = larger[i:]
            if np.any((b + cs) % cand == 0):
                return False
    
    # Case 2: cand plays role of 'b' (middle in triple)
    # Need: no a,c in S with a < cand, c >= cand, a | (cand + c)
    smaller = S_arr[S_arr < cand]
    geq = S_arr[S_arr >= cand]
    if len(smaller) > 0 and len(geq) > 0:
        # For each a in smaller, check if any c in geq satisfies (cand + c) % a == 0
        for a in smaller:
            if np.any((cand + geq) % a == 0):
                return False
    
    # Case 3: cand plays role of 'c' (largest in triple)
    # Need: no a,b in S with a < b <= cand, a | (b + cand)
    for a in S_arr[S_arr < cand]:
        bs = S_arr[(S_arr > a) & (S_arr <= cand)]
        if len(bs) > 0 and np.any((bs + cand) % a == 0):
            return False
    
    return True

def _local_search_add(S, n, deadline):
    """Try to add skipped elements to S."""
    S = set(S)
    skipped = set(range(1, n+1)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in sorted(skipped, reverse=True):
            if time.time() > deadline:
                break
            if _try_add(S, cand, n):
                S.add(cand)
                skipped.discard(cand)
                improved = True
    return S

def _swap_local_search(S, n, deadline, max_swaps=50):
    """Try to improve S by swapping out elements for better ones."""
    S = set(S)
    skipped = set(range(1, n+1)) - S
    S_list = sorted(S)
    swaps_made = 0
    
    while swaps_made < max_swaps and time.time() < deadline:
        improved = False
        # First try pure additions
        for cand in sorted(skipped, reverse=True):
            if _try_add(S, cand, n):
                S.add(cand)
                skipped.discard(cand)
                improved = True
                swaps_made += 1
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove y from S, add x from skipped
        for y in S_list:
            if time.time() > deadline:
                return S
            S_temp = S - {y}
            for x in sorted(skipped, reverse=True):
                if x == y:
                    continue
                if _try_add(S_temp, x, n):
                    S = S_temp | {x}
                    skipped.discard(x)
                    S_list = sorted(S)
                    swaps_made += 1
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try 1-for-2 swaps: remove y from S, add x1 and x2 from skipped
            for y in S_list:
                if time.time() > deadline:
                    return S
                S_temp = S - {y}
                # Try to add two elements
                added = []
                for x in sorted(skipped, reverse=True):
                    if x == y:
                        continue
                    if _try_add(S_temp, x, n):
                        S_temp.add(x)
                        added.append(x)
                        if len(added) == 2:
                            break
                if len(added) >= 2:
                    S = S_temp
                    for x in added:
                        skipped.discard(x)
                    S_list = sorted(S)
                    swaps_made += 1
                    improved = True
                    break
                elif len(added) == 1:
                    # Only got 1, not worth it
                    pass
            
            if not improved:
                break
    
    return S

def _construct_greedy(n, order, deadline):
    """Greedy construction based on a specific order of candidates."""
    S = set()
    for cand in order:
        if time.time() > deadline:
            break
        if _try_add(S, cand, n):
            S.add(cand)
    return S

def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # Safety margin (90s limit, 5s buffer)
    n = 256
    
    best_S = set()
    
    # Strategy 1: Descending order
    S1 = _construct_greedy(n, range(n, 0, -1), total_deadline)
    S1 = _local_search_add(S1, n, total_deadline)
    if len(S1) > len(best_S):
        best_S = set(S1)
    
    # Strategy 2: Ascending order
    if time.time() < total_deadline:
        S2 = _construct_greedy(n, range(1, n+1), total_deadline)
        S2 = _local_search_add(S2, n, total_deadline)
        if len(S2) > len(best_S):
            best_S = set(S2)
    
    # Strategy 3: Randomized Greedy with multiple seeds
    rng = random.Random(42)
    for seed in range(8):
        if time.time() > total_deadline - 8:
            break
        rng = random.Random(seed * 12345 + 67890)
        order = list(range(1, n+1))
        rng.shuffle(order)
        S_rand = _construct_greedy(n, order, total_deadline)
        S_rand = _local_search_add(S_rand, n, total_deadline)
        if len(S_rand) > len(best_S):
            best_S = set(S_rand)
    
    # Strategy 4: Weighted random (bias toward larger numbers)
    if time.time() < total_deadline - 5:
        for seed in range(4):
            if time.time() > total_deadline - 5:
                break
            rng = random.Random(seed * 7777 + 12345)
            # Create weighted order: larger numbers have higher probability
            weights = [i**2 for i in range(1, n+1)]
            order = []
            remaining = list(range(1, n+1))
            while remaining:
                total_w = sum(weights[i-1] for i in remaining)
                r = rng.random() * total_w
                cum = 0
                for idx, val in enumerate(remaining):
                    cum += weights[val-1]
                    if r <= cum:
                        order.append(val)
                        remaining.pop(idx)
                        break
            S_w = _construct_greedy(n, order, total_deadline)
            S_w = _local_search_add(S_w, n, total_deadline)
            if len(S_w) > len(best_S):
                best_S = set(S_w)
    
    # Strategy 5: Swap-based local search on best found
    if time.time() < total_deadline - 10:
        S_polish = _swap_local_search(best_S, n, total_deadline - 3, max_swaps=100)
        if len(S_polish) > len(best_S):
            best_S = S_polish
    
    # Final: one more pass of additions
    if time.time() < total_deadline - 2:
        S_final = _local_search_add(best_S, n, total_deadline - 1)
        if len(S_final) > len(best_S):
            best_S = S_final
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
