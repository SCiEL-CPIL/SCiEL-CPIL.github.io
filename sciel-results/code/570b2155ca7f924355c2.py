   def construct():
       n = 256
       primes = _sieve_primes(n)
       primes_set = set(primes)
       
       # Calculate max power for each prime
       powers = {}
       for p in primes:
           p_power = p
           next_p = p * p
           while next_p <= n:
               p_power = next_p
               next_p *= p
           powers[p] = p_power
       
       # Initial set: powers of all primes
       current_set = set(powers.values())
       current_sum = sum(current_set)
       
       # We want to replace (p_power, q) with (p*q)
       # for small p.
       # p must be in {2, 3, 5, 7, 11, 13}
       # q must be a prime > p
       # p*q <= 256
       # q not used by another p
       
       # Greedy: sort small primes by potential gain?
       # Or just iterate.
       
       used_q = set()
       # We need to decide which q to assign to which p.
       # Since gain is linear in q, we should assign largest available q to p with highest (p-1)?
       # Gain = q(p-1) - P_p.
       # Slope is (p-1).
       # p=13 slope 12.
       # p=11 slope 10.
       # p=7 slope 6.
       # p=5 slope 4.
       # So 13 should get the largest q, then 11, then 7, then 5.
       
       small_primes = [13, 11, 7, 5] # Sorted by slope descending
       
       # Available q's: primes > p, p*q <= 256.
       # We need to pick distinct q's.
       
       # Let's try to assign greedily.
       assigned_q = {}
       for p in small_primes:
           p_power = powers[p]
           # Find largest q > p, q prime, p*q <= 256, q not in used_q
           best_q = None
           best_gain = 0
           for q in range(n // p, p, -1):
               if q in primes_set and q not in used_q:
                   prod = p * q
                   gain = prod - p_power - q
                   if gain > best_gain:
                       best_gain = gain
                       best_q = q
           
           if best_q is not None and best_gain > 0:
               assigned_q[p] = best_q
               used_q.add(best_q)
       
       # Apply assignments
       for p, q in assigned_q.items():
           current_set.remove(powers[p])
           current_set.remove(q)
           current_set.add(p * q)
           current_sum += p * q - powers[p] - q
       
       # Add 1
       current_set.add(1)
       current_sum += 1
       
       # Local search: Add any number coprime to all
       # (This might find more gains if we missed something)
       
       return list(current_set)
   