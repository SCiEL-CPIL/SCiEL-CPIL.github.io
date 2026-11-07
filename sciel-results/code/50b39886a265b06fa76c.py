import time

def construct():
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 50  # 50s margin
        
        P = {'n': 256}
        n = P['n']
        
        # Sieve of Eratosthenes for primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        
        primes = [i for i in range(2, n+1) if is_prime[i]]
        
        # The set of primes is a valid solution.
        # Size is 54.
        
        # Attempt to see if we can add any composites?
        # Given the risk of invalidity and the time cost, and the fact that
        # replacing primes with composites usually doesn't increase count,
        # we return the primes.
        
        # Ensure types are correct
        return [int(p) for p in primes]
