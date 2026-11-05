"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _is_compatible_with_set(x, S):
    """Check if x can be added to set S (all pairs have lcm <= N)."""
    for y in S:
        if _lcm(x, y) > N:
            return False
    return True

def _greedy_add(S, candidates):
    """Try to add each candidate to S in order; return updated S."""
    S = set(S)
    for x in candidates:
        if x not in S and _is_compatible_with_set(x, S):
            S.add(x)
    return S

def _get_divisors(m):
    divs = []
    for i in range(1, int(m**0.5) + 1):
        if m % i == 0:
            divs.append(i)
            if i != m // i:
                divs.append(m // i)
    return [d for d in divs if d <= N]

def _local_search_advanced(S, start_time, deadline):
    """
    Advanced local search:
    1. Try to add any compatible element.
    2. Try removing 1 element and greedily adding as many as possible.
    3. Try removing 2 elements and greedily adding as many as possible.
    """
    S = set(S)
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        current_size = len(S)
        
        # 1. Try to add any single element
        for x in range(1, N + 1):
            if (time.time() - start_time) > deadline:
                return S
            if x not in S and _is_compatible_with_set(x, S):
                S.add(x)
                improved = True
                current_size += 1
                break
        
        if improved:
            continue

        # 2. Try Remove 1, Add Greedy
        S_list = sorted(S)
        for elem in S_list:
            if (time.time() - start_time) > deadline:
                return S
            S_without = S - {elem}
            # Greedily add compatible elements from 1..N
            S_new = set(S_without)
            for x in range(1, N + 1):
                if x in S_new:
                    continue
                if _is_compatible_with_set(x, S_new):
                    S_new.add(x)
            
            if len(S_new) > current_size:
                S = S_new
                improved = True
                break
        
        if improved:
            continue

        # 3. Try Remove 2, Add Greedy (Only if time permits and size is small enough)
        if current_size < 30 and (time.time() - start_time) < deadline - 5:
            # To avoid O(N^2) combinations of removals being too slow, 
            # we limit the number of pairs we try.
            # We try pairs involving small numbers first? Or just first 20 pairs.
            pairs_tried = 0
            for i in range(len(S_list)):
                if pairs_tried > 50 or (time.time() - start_time) > deadline:
                    break
                for j in range(i + 1, len(S_list)):
                    if pairs_tried > 50 or (time.time() - start_time) > deadline:
                        break
                    pairs_tried += 1
                    e1, e2 = S_list[i], S_list[j]
                    S_without = S - {e1, e2}
                    S_new = set(S_without)
                    for x in range(1, N + 1):
                        if x in S_new:
                            continue
                        if _is_compatible_with_set(x, S_new):
                            S_new.add(x)
                    
                    if len(S_new) > current_size:
                        S = S_new
                        improved = True
                        break
                if improved:
                    break

    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Divisors of highly composite numbers
    highly_composite = [360, 480, 504, 420, 336, 240, 180, 120, 720] # 720 > 512, so divisors <= 512
    # Filter divisors <= 512
    for m in highly_composite:
        if (time.time() - start_time) > deadline:
            break
        divs = _get_divisors(m)
        if not divs: continue
        S = set(divs)
        # Greedy add
        for x in range(1, N + 1):
            if (time.time() - start_time) > deadline:
                break
            if x not in S and _is_compatible_with_set(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        
        # Local search from this
        if (time.time() - start_time) < deadline:
            S_local = _local_search_advanced(S, start_time, deadline)
            if len(S_local) > best_size:
                best_size = len(S_local)
                best_S = S_local

    # Strategy 2: Randomized greedy with multiple restarts
    if (time.time() - start_time) < deadline:
        random.seed(42)
        for trial in range(20):
            if (time.time() - start_time) > deadline:
                break
            S_rand = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if (time.time() - start_time) > deadline:
                    break
                if x not in S_rand and _is_compatible_with_set(x, S_rand):
                    S_rand.add(x)
            
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_S = S_rand
            
            # Local search from random result
            if (time.time() - start_time) < deadline - 5:
                S_rand_local = _local_search_advanced(S_rand, start_time, deadline)
                if len(S_rand_local) > best_size:
                    best_size = len(S_rand_local)
                    best_S = S_rand_local

    # Final local search from best
    if (time.time() - start_time) < deadline:
        S_final = _local_search_advanced(best_S, start_time, deadline)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final

    if not best_S:
        best_S = {1}
    
    return list(best_S)
# EVOLVE-BLOCK-END
