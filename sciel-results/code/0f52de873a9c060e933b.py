import math

def construct():
    # Range for n is [10000, 1000000]
    # We need largest prime factors for n and n+1
    # Max value is 1000001
    MAX_VAL = 1000001
    
    # Sieve to find largest prime factor for each number up to MAX_VAL
    lpf = [0] * (MAX_VAL + 1)
    
    # Initialize lpf for all numbers
    # We can do a standard sieve modification
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
                
    best_n = -1
    best_score = float('inf')
    
    # Iterate through valid n
    for n in range(10000, 1000000 + 1):
        # Largest prime factor of n is lpf[n]
        # Largest prime factor of n+1 is lpf[n+1]
        current_max = max(lpf[n], lpf[n+1])
        
        if current_max < best_score:
            best_score = current_max
            best_n = n
            
    return best_n
