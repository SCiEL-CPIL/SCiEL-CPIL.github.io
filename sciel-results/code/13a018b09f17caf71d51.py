def construct():
    import time
    from fractions import Fraction

    target = Fraction(5, 31)
    best_max = float('inf')
    best_denoms = []

    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s/90s kill

    # The first denominator d1 must satisfy 1/d1 < 5/31 => d1 > 6.2
    # So d1 starts at 7.
    # We can bound d1. If d1 is too large, the remaining fraction 5/31 - 1/d1 
    # will likely require larger denominators.
    # Let's iterate d1 from 7 to 200.

    for d1 in range(7, 200):
        if time.time() > deadline:
            break

        r = target - Fraction(1, d1)
        if r <= 0:
            continue

        used = {d1}
        denoms = [d1]

        # Greedy completion with lookahead to avoid huge denominators
        valid = True
        while r > 0:
            # Find smallest k not in used such that 1/k <= r
            # k >= ceil(1/r) = ceil(r.denominator / r.numerator)
            k_min = (r.denominator + r.numerator - 1) // r.numerator

            # Try a few candidates starting from k_min to see if we can
            # reduce the maximum denominator in the remaining expansion.
            # The pure greedy choice is k_min (or next available).
            # But sometimes choosing a slightly larger k leads to smaller
            # subsequent denominators.
            
            best_candidate_k = None
            best_candidate_max = float('inf')
            
            # Look ahead a small number of steps for each candidate
            # Limit search to avoid timeout
            for candidate_k in range(k_min, k_min + 20):
                if candidate_k in used:
                    continue
                if candidate_k > 10000000:
                    break
                
                # Simulate the rest of the greedy expansion from this point
                temp_r = r - Fraction(1, candidate_k)
                if temp_r < 0:
                    continue
                if temp_r == 0:
                    # Perfect, no more terms needed
                    temp_max = candidate_k
                    if temp_max < best_candidate_max:
                        best_candidate_max = temp_max
                        best_candidate_k = candidate_k
                    continue
                
                # Greedy complete the rest, but only check the max
                temp_used = used | {candidate_k}
                temp_max = candidate_k
                temp_valid = True
                temp_terms = 0
                tr = temp_r
                while tr > 0:
                    if temp_terms > 50:  # Safety: if expansion is too long, skip
                        temp_valid = False
                        break
                    k2_min = (tr.denominator + tr.numerator - 1) // tr.numerator
                    k2 = k2_min
                    while k2 in temp_used:
                        k2 += 1
                        if k2 > 10000000:
                            temp_valid = False
                            break
                    if not temp_valid:
                        break
                    temp_used.add(k2)
                    if k2 > temp_max:
                        temp_max = k2
                    tr -= Fraction(1, k2)
                    temp_terms += 1
                    if tr < 0:
                        temp_valid = False
                        break
                
                if temp_valid and temp_max < best_candidate_max:
                    best_candidate_max = temp_max
                    best_candidate_k = candidate_k
            
            if best_candidate_k is None:
                # Fallback to pure greedy
                k = k_min
                while k in used:
                    k += 1
                best_candidate_k = k
                best_candidate_max = k

            denoms.append(best_candidate_k)
            used.add(best_candidate_k)
            r -= Fraction(1, best_candidate_k)

        if denoms:
            current_max = max(denoms)
            if current_max < best_max:
                best_max = current_max
                best_denoms = denoms[:]
                # Early exit if we find a very good solution
                if best_max <= 50:
                    break

    if not best_denoms:
        # Fallback: pure greedy from d=8
        r = target - Fraction(1, 8)
        best_denoms = [8]
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            best_denoms.append(d)
            r -= Fraction(1, d)

    return best_denoms
