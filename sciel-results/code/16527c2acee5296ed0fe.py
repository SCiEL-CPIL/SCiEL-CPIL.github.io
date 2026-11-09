   def oe_merge(n):
       if n == 1: return [(0, 1)]
       comps = [(i, n+i) for i in range(n)]
       # Recursive calls on even and odd indexed wires
       # The even wires are 0, 2, 4... n-1? No, 0, 2, ..., 2*(n/2)-1?
       # The sub-problems are of size n (which is 2*(n/2)).
       # The indices in the sub-problem 0..n-1 map to global 0, 2, 4.. and 1, 3, 5..
       even_sub = oe_merge(n // 2)
       odd_sub = oe_merge(n // 2)
       for a, b in even_sub:
           comps.append((2*a, 2*b))
       for a, b in odd_sub:
           comps.append((2*a+1, 2*b+1))
       return comps
   