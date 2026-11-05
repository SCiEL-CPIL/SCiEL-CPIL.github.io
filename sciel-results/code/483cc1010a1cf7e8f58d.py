import math

def construct():
       moduli = list(range(20, 201, 2))
       selected = []
       
       for m in moduli:
           found = False
           for r in range(m):
               valid = True
               for m2, r2 in selected:
                   g = math.gcd(m, m2)
                   if r % g == r2 % g:
                       valid = False
                       break
               if valid:
                   selected.append([m, r])
                   found = True
                   break
       
       return selected
