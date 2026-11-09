import itertools
import time

def construct():
       # Return Clebsch graph as fallback
       edges_clebsch = []
       n = 16
       for u in range(n):
           for v in range(u + 1, n):
               xor = u ^ v
               bits = bin(xor).count('1')
               if bits == 1 or bits == 4:
                   edges_clebsch.append([u, v])
       
       # Try to find a better graph
       # ... search logic ...
       
       return edges_clebsch
