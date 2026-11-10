import math
import time
def construct():
       start_time = time.time()
       deadline = start_time + 80
       
       MAX_N = 1000000
       MIN_N = 10000
       limit = MAX_N + 1
       
       # SPF Sieve
       spf = list(range(limit + 1))
       for i in range(2, int(math.isqrt(limit)) + 1):
           if spf[i] == i:
               for j in range(i * i, limit + 1, i):
                   if spf[j] == j:
                       spf[j] = i
       
       def calc_pp(n):
           x = n
           val = 1
           while x > 1:
               p = spf[x]
               e = 0
               while x % p == 0:
                   x //= p
                   e += 1
               if e >= 2:
                   val *= p ** e
           return val
       
       best_n = MIN_N
       best_val = 0
       
       curr_pp = calc_pp(MIN_N)
       
       for n in range(MIN_N, MAX_N):
           next_pp = calc_pp(n + 1)
           val = curr_pp * next_pp
           if val > best_val:
               best_val = val
               best_n = n
           
           curr_pp = next_pp
           
           if time.time() > deadline - 5:
               break
               
       return best_n
