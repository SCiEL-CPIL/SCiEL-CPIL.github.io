import sys
sys.setrecursionlimit(10000)

def construct():
       primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
       # Generate k and sigma(k)
       # We want to count M = k * sigma(k)
       
       solutions = {}
       
       def gen(idx, current_k, current_sigma):
           if idx == len(primes):
               M = current_k * current_sigma
               if M <= 10**18:
                   if M not in solutions:
                       solutions[M] = []
                   solutions[M].append(current_k)
               return
           
           p = primes[idx]
           # Multiply by p^e
           k_pow = 1
           sigma_pow = 1
           # Sum 1 + p + ... + p^e
           # sigma(p^e) = (p^(e+1)-1)/(p-1)
           # We update current_k and current_sigma as we go
           
           # Start with e=0
           gen(idx + 1, current_k, current_sigma)
           
           # e=1
           k_pow *= p
           # sigma(p^1) = 1 + p
           # new_sigma = current_sigma * (1+p) ? No.
           # sigma(k * p^e) = sigma(k) * sigma(p^e) if gcd(k, p)=1.
           # Here k is composed of primes < p, so gcd is 1.
           
           # Let's just accumulate the factor for this prime.
           # Let S_p(e) = sigma(p^e).
           # We need to multiply current_sigma by S_p(e).
           
           # It's easier to just compute sigma(k) from scratch for each k?
           # No, we can pass the "partial sigma" for the primes used so far.
           
           # Let's restart the logic:
           # gen(idx, k, sigma_k)
           # where k = product of primes[0:idx] ** exponents
           # and sigma_k = product of sigma(p^e)
           
           # This works.
           
           e = 1
           while True:
               # Update for p^e
               # k_new = current_k * p^e
               # sigma_new = current_sigma * sigma(p^e)
               
               # Calculate p^e and sigma(p^e) iteratively
               # p^1 = p, sigma(p^1) = 1+p
               # p^2 = p*p, sigma(p^2) = 1+p+p^2
               
               # Let's keep track of pow_p and sigma_p
               # Initially pow_p = 1, sigma_p = 1
               # Step e:
               # pow_p *= p
               # sigma_p += pow_p
               
               # Wait, I need to do this inside the loop.
               
               break # Placeholder
           
           # Correct implementation:
           
           pow_p = 1
           sigma_p = 1
           while True:
               pow_p *= p
               if current_k * pow_p > 10**9: # Limit k
                   break
               sigma_p += pow_p
               
               new_k = current_k * pow_p
               new_sigma = current_sigma * sigma_p
               
               gen(idx + 1, new_k, new_sigma)
               
       gen(0, 1, 1)
       
       # Find best M
       best_M = 0
       best_k_list = []
       for M, k_list in solutions.items():
           if len(k_list) > len(best_k_list):
               best_M = M
               best_k_list = k_list
               
       return {'M': best_M, 'k': best_k_list}
