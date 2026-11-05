import time
import random

def construct():
    # Start with a known good configuration: odd numbers 1..23
    # This yields 8 distinct prime factors: {2, 3, 5, 7, 11, 13, 17, 19}
    best_set = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    best_primes = set()
    
    # Calculate primes for the initial set
    def get_primes(s):
        primes = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                val = s[i] + s[j]
                # Factor val
                temp = val
                d = 2
                while d * d <= temp:
                    while temp % d == 0:
                        primes.add(d)
                        temp //= d
                    d += 1
                if temp > 1:
                    primes.add(temp)
        return primes

    best_primes = get_primes(best_set)
    best_count = len(best_primes)
    
    # Try a few random perturbations to see if we can beat 8
    # Given the tight time limit and small N, a simple local search might help
    # but the static set is already very good.
    # We will just return the static set to ensure reliability and speed.
    
    return best_set
