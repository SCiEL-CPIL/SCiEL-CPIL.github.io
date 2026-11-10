      def mobius_sieve(n):
          mu = np.ones(n+1, dtype=np.int8)
          is_prime = np.ones(n+1, dtype=bool)
          primes = []
          mu[0] = 0
          for i in range(2, n+1):
              if is_prime[i]:
                  primes.append(i)
                  mu[i] = -1
              for p in primes:
                  if i * p > n:
                      break
                  is_prime[i * p] = False
                  if i % p == 0:
                      mu[i * p] = 0
                      break
                  else:
                      mu[i * p] = -mu[i]
          return mu
      