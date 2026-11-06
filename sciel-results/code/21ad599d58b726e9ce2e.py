   def find_solution(r, min_d, M):
       if r == 0:
           return []
       if min_d > M:
           return None
       
       # Lower bound for d
       # 1/d <= r => d >= 1/r
       low_d = max(min_d, (1 + r.denominator) // r.numerator) 
       # Wait, 1/r = d/n. ceil(d/n).
       # If r = n/d_num, 1/r = d_num/n.
       # ceil(d_num/n) = (d_num + n - 1) // n.
       
       if low_d > M:
           return None
       
       # Check if sum of max possible terms can reach r?
       # Sum from low_d to M.
       # If sum_{i=low_d}^M 1/i < r, return None.
       # This is a good pruning.
       # Approximation: integral.
       
       for d in range(low_d, M + 1):
           rem = r - Fraction(1, d)
           if rem < 0:
               continue
           res = find_solution(rem, d + 1, M)
           if res is not None:
               return [d] + res
       return None
   