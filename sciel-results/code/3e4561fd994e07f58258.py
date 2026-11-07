"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5-second margin before 60s limit
    
    # Pre-compute small primes up to the maximum possible gap
    # Max gap in P is 20 (10009 to 10037 is 28? No, 10037-10009=28. Let's check max gap.)
    # 10009-10007=2
    # 10037-10009=28
    # 10039-10037=2
    # 10061-10039=22
    # 10067-10061=6
    # 10069-10067=2
    # 10079-10069=10
    # 10091-10079=12
    # 10093-10091=2
    # 10099-10093=6
    # 10103-10099=4
    # 10111-10103=8
    # 10133-10111=22
    # 10139-10133=6
    # 10141-10139=2
    # 10151-10141=10
    # 10159-10151=8
    # 10163-10159=4
    # 10169-10163=6
    # 10177-10169=8
    # 10181-10177=4
    # 10193-10181=12
    # 10211-10193=18
    # 10223-10211=12
    # 10243-10223=20
    # 10247-10243=4
    # 10253-10247=6
    # 10259-10253=6
    # 10267-10259=8
    # 10271-10267=4
    # 10273-10271=2
    # 10289-10273=16
    
    max_gap = 0
    for lo, hi in P['gaps']:
        if hi - lo > max_gap:
            max_gap = hi - lo
            
    # Sieve for primes up to max_gap
    # We need primes strictly less than the gap size g
    # So if g=28, we check divisibility by 2,3,5,7,11,13,17,19,23
    is_prime = [True] * (max_gap + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_gap**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_gap + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        g = hi - lo
        # Primes to check are those strictly less than g
        check_primes = [p for p in primes if p < g]
        
        found = None
        for m in range(lo + 1, hi):
            # Check if m is divisible by any prime in check_primes
            is_valid = True
            for p in check_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                found = m
                break
                
        result[i] = found
        
    return result
# EVOLVE-BLOCK-END
