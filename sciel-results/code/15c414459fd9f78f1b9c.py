import math
import time

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    # Prime factors of 255
    factors = [3, 5, 17]
    
    # Candidate pool: numbers in [1,255] that share a factor with 255
    candidates = set()
    for p in factors:
        for k in range(1, n // p + 1):
            candidates.add(k * p)
    candidates = sorted(candidates)
    
    # Precompute gcd relationships
    # For each candidate, compute which other candidates share a factor
    # We need a set S containing 255 where all pairs have gcd > 1
    
    best_set = list(range(3, n + 1, 3))  # multiples of 3, size 85
    best_size = len(best_set)
    
    # Try greedy addition starting from multiples of 3
    current = set(best_set)
    for x in candidates:
        if x in current:
            continue
        if time.time() > deadline:
            break
        # Check if x shares gcd > 1 with all elements in current
        compatible = True
        for y in current:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            current.add(x)
    
    if len(current) > best_size:
        best_set = list(current)
        best_size = len(best_set)
    
    # Try starting from multiples of 5
    current5 = set(range(5, n + 1, 5))
    for x in candidates:
        if x in current5:
            continue
        if time.time() > deadline:
            break
        compatible = True
        for y in current5:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            current5.add(x)
    
    if len(current5) > best_size:
        best_set = list(current5)
        best_size = len(best_set)
    
    # Try starting from multiples of 17
    current17 = set(range(17, n + 1, 17))
    for x in candidates:
        if x in current17:
            continue
        if time.time() > deadline:
            break
        compatible = True
        for y in current17:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            current17.add(x)
    
    if len(current17) > best_size:
        best_set = list(current17)
        best_size = len(best_set)
    
    # Try mixed approach: start with multiples of 3, try adding multiples of 5 
    # that also share factors with all multiples of 3
    # A multiple of 5 (not 3) shares gcd>1 with a multiple of 3 (not 5) only if
    # they share another prime factor. Let's check systematically.
    
    # More systematic: try all subsets of {3,5,17} as "base primes" and see
    # what maximum clique we can form
    
    # Actually, let's try a local search / greedy from different starting points
    # Try starting with just {255} and greedily add
    
    current_greedy = {255}
    # Sort candidates by how many connections they have (heuristic)
    # Number of candidates sharing a factor with x
    conn_count = {}
    for x in candidates:
        count = 0
        for y in candidates:
            if x != y and math.gcd(x, y) > 1:
                count += 1
        conn_count[x] = count
    
    sorted_candidates = sorted([x for x in candidates if x != 255], key=lambda x: conn_count[x], reverse=True)
    
    for x in sorted_candidates:
        if time.time() > deadline:
            break
        compatible = True
        for y in current_greedy:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            current_greedy.add(x)
    
    if len(current_greedy) > best_size:
        best_set = list(current_greedy)
        best_size = len(best_set)
    
    # Local search: try to swap elements to increase size
    # For each element in best_set, try removing it and adding two new ones
    best_set_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, x in enumerate(sorted(best_set_set)):
            if time.time() > deadline:
                break
            # Try removing x
            temp_set = best_set_set - {x}
            # Try adding any candidate not in temp_set
            for y in candidates:
                if y in temp_set:
                    continue
                # Check if y is compatible with all in temp_set
                compatible = True
                for z in temp_set:
                    if math.gcd(y, z) == 1:
                        compatible = False
                        break
                if compatible:
                    # Now check if we can add another element w
                    temp_set2 = temp_set | {y}
                    for w in candidates:
                        if w in temp_set2:
                            continue
                        compatible2 = True
                        for z in temp_set2:
                            if math.gcd(w, z) == 1:
                                compatible2 = False
                                break
                        if compatible2:
                            # We found a swap: remove x, add y and w
                            best_set_set = temp_set2 | {w}
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    
    if len(best_set_set) > best_size:
        best_set = list(best_set_set)
        best_size = len(best_set)
    
    return best_set
