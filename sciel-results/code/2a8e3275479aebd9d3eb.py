   import time
   import random

   def construct():
       start_time = time.time()
       deadline = start_time + 55.0
       
       n = 10
       L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
       
       def find_orthogonal_square():
           L2 = [[None for _ in range(n)] for _ in range(n)]
           row_used = [set() for _ in range(n)]
           col_used = [set() for _ in range(n)]
           trans_used = [set() for _ in range(n)]
           
           def solve(r, c):
               if r == n:
                   return True
               
               nr, nc = r, c + 1
               if nc == n:
                   nr, nc = r + 1, 0
                   
               k = (r + c) % n
               
               # Heuristic: try values in random order
               vals = list(range(n))
               random.shuffle(vals)
               
               for v in vals:
                   if v in row_used[r] or v in col_used[c] or v in trans_used[k]:
                       continue
                   
                   L2[r][c] = v
                   row_used[r].add(v)
                   col_used[c].add(v)
                   trans_used[k].add(v)
                   
                   if solve(nr, nc):
                       return True
                   
                   L2[r][c] = None
                   row_used[r].remove(v)
                   col_used[c].remove(v)
                   trans_used[k].remove(v)
                   
               return False
               
           return solve(0, 0), L2

       found, L2 = find_orthogonal_square()
       
       if found:
           res_L1 = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
           res_L2 = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
           return [res_L1, res_L2]
       else:
           # Retry a few times if it fails (unlikely)
           for _ in range(10):
               if time.time() > deadline:
                   break
               found, L2 = find_orthogonal_square()
               if found:
                   res_L1 = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
                   res_L2 = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
                   return [res_L1, res_L2]
                   
           # Fallback
           res_L1 = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
           return [res_L1]
   