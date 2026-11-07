import time
from math import gcd

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 50.0 # Leave 10s margin for 60s limit
    
    N = 256
    result = []
    
    # Greedy algorithm: Pick largest numbers first that are coprime to all selected
    # Iterate from N down to 1
    for n in range(N, 0, -1):
        # Check if n is coprime to all already selected numbers
        is_coprime = True
        for x in result:
            if gcd(n, x) != 1:
                is_coprime = False
                break
        
        if is_coprime:
            result.append(n)
            
    # Verify time
    if time.time() - start_time > timeout:
        # Should not happen, but just in case
        pass
        
    # Ensure types are correct
    return [int(x) for x in result]
