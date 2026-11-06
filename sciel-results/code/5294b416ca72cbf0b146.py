   def construct():
       n = 20
       parts = [[i for i in range(5*j, 5*(j+1))] for j in range(4)]
       A, B, C, D = parts
       
       edges = []
       
       # AB: Complete
       for a in A:
           for b in B:
               idx_a = a % 5
               idx_b = b % 5
               edges.append([a, b, idx_a]) # Color 0-4. 
               # Wait, color must be unique per edge? No, just per cycle.
               # But if we use idx_a, all edges from a have same color.
               # Is that ok?
               # Type 1: AB, BC, CD, DA.
               # c_AB(a,b) = idx_a.
               # c_BC(b,c) = 5 + ...
               # Distinct ranges. OK.
               
       # CD: Complete
       for c in C:
           for d in D:
               idx_c = c % 5
               edges.append([c, d, 10 + idx_c])
               
       # BC: Remove diagonal
       for b in B:
           idx_b = b % 5
           for c in C:
               idx_c = c % 5
               if idx_b != idx_c:
                   edges.append([b, c, 5 + idx_b])
                   
       # DA: Remove diagonal
       for d in D:
           idx_d = d % 5
           for a in A:
               idx_a = a % 5
               if idx_d != idx_a:
                   edges.append([d, a, 15 + idx_d])
                   
       # AC: Complete
       for a in A:
           idx_a = a % 5
           # Unused color in DA for a is 15 + idx_a
           color_ac = 15 + idx_a
           for c in C:
               edges.append([a, c, color_ac])
               
       # BD: Complete
       for b in B:
           idx_b = b % 5
           # Unused color in BC for b is 5 + idx_b
           color_bd = 5 + idx_b
           for d in D:
               edges.append([b, d, color_bd])
               
       return edges
   