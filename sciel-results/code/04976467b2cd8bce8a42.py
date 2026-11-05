import time

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Step 1: Precompute Largest Prime Factor (LPF) for all numbers up to TARGET
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (TARGET + 1)
    
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            # i is a prime
            lpf[i] = i
            # Mark multiples of i. Since we iterate i in increasing order,
            # any prime that divides a multiple will be smaller than or equal to i.
            # Wait, if we update lpf[j] = i for all multiples j of i,
            # and we go from i=2 to N, then for a number j, the last prime 
            # that divides it and is <= j will be the largest prime factor.
            # Example: j=12. 
            # i=2: lpf[12]=2
            # i=3: lpf[12]=3
            # i=4: not prime (lpf[4]!=0)
            # i=5..11: no
            # i=12: not prime
            # Final lpf[12]=3. Correct.
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    # Ensure lpf[1] is defined as 1 per problem statement P(1)=1
    lpf[1] = 1

    # Step 2: Scan all pairs (a, b) with a+b=TARGET
    best_max_p = TARGET  # Worst case is a prime factor of TARGET or similar
    best_a = 1
    best_b = TARGET - 1
    
    # We only need to check a from 1 to TARGET//2
    # because (a, b) and (b, a) yield the same max(P(a), P(b))
    for a in range(1, TARGET // 2 + 1):
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            
        # Check time limit just in case, though this loop is very fast
        if time.time() - start_time > DEADLINE:
            break

    return [best_a, best_b]
