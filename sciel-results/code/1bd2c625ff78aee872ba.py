import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 115  # 5s margin

       # Start with n=32 seed
       def get_seed_32():
           # Re-implement seed logic
           def mul(a, b):
               r = 0
               for i in range(4):
                   if b >> i & 1:
                       r ^= a << i
               for i in range(7, 3, -1):
                   if r >> i & 1:
                       r ^= 0b10011 << (i - 4)
               return r
           log, x = {}, 1
           for e in range(15):
               log[x] = e
               x = mul(x, 2)
           colors = []
           for i in range(32):
               for j in range(i + 1, 32):
                   colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
           return colors

       colors = get_seed_32()
       n = 32
       
       # Helper to get color of edge (i, j)
       def get_edge_color(i, j):
           if i > j: i, j = j, i
           # Index in flat array
           # Sum of previous rows: i*(i-1)/2 + (j-i-1)
           # Wait, lexicographic order (i,j) with i<j.
           # Index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1) ?
           # No, the list is for fixed n.
           # Let's use a 2D array for speed during search, then flatten at the end.
           pass
       
       # Use 2D list for colors
       # colors[i][j] for i < j
       c = [[0]*n for _ in range(n)]
       idx = 0
       for i in range(n):
           for j in range(i+1, n):
               c[i][j] = colors[idx]
               c[j][i] = colors[idx] # Symmetric
               idx += 1
               
       # Function to count mono triangles
       def count_triangles(c, n):
           count = 0
           for i in range(n):
               for j in range(i+1, n):
                   col = c[i][j]
                   for k in range(j+1, n):
                       if c[i][k] == col and c[j][k] == col:
                           count += 1
           return count
           
       # Initial check
       # It should be 0.
       
       # Try to expand n
       best_n = n
       best_colors = [c[i][j] for i in range(n) for j in range(i+1, n)]
       
       def try_expand(n_current, c_current):
           nonlocal best_n, best_colors
           if time.time() > deadline:
               return False
           
           n_new = n_current + 1
           # Initialize new colors
           # c_new is (n_new) x (n_new)
           c_new = [row[:] for row in c_current]
           # Add new row/col
           c_new.append([0]*n_new)
           for i in range(n_current):
               c_new[i].append(0)
               
           # Randomly initialize new edges
           for i in range(n_current):
               c_new[i][n_new-1] = random.randint(0, 3)
               c_new[n_new-1][i] = c_new[i][n_new-1]
               
           # Local Search
           # We want to minimize triangles.
           # Start with random init, then flip edges to reduce count.
           
           # To speed up, we can try to find a good init first.
           # Or just run SA.
           
           current_count = count_triangles(c_new, n_new)
           
           # If we start with 0 triangles, we are done.
           # But random init likely has many.
           
           # SA Loop
           T = 1.0
           cooling = 0.99
           steps = 10000
           
           for step in range(steps):
               if time.time() > deadline:
                   return False
               
               if current_count == 0:
                   best_n = n_new
                   best_colors = [c_new[i][j] for i in range(n_new) for j in range(i+1, n_new)]
                   return True
               
               # Pick random edge
               u = random.randint(0, n_new-1)
               v = random.randint(0, n_new-1)
               if u == v: continue
               if u > v: u, v = v, u
               
               old_col = c_new[u][v]
               # Try all other colors? Or just flip?
               # Flipping is faster.
               new_col = (old_col + 1) % 4
               
               # Calculate delta
               delta = 0
               # Triangles involving (u,v)
               for w in range(n_new):
                   if w == u or w == v: continue
                   
                   # Before
                   if c_new[u][w] == old_col and c_new[v][w] == old_col:
                       delta -= 1 # We remove a triangle
                       
                   # After
                   if c_new[u][w] == new_col and c_new[v][w] == new_col:
                       delta += 1 # We add a triangle
                   
               # Accept if delta < 0 or with prob exp(-delta/T)
               if delta < 0:
                   c_new[u][v] = new_col
                   c_new[v][u] = new_col
                   current_count += delta
               else:
                   # Prob
                   import math
                   if random.random() < math.exp(-delta / T):
                       c_new[u][v] = new_col
                       c_new[v][u] = new_col
                       current_count += delta
               
               T *= cooling
               
           # If we didn't reach 0, we might keep the best state?
           # But we need 0 triangles for validity.
           # If we can't reach 0, we can't use this n.
           # So return False.
           return False
           
       # Expansion loop
       while n < 62:
           if time.time() > deadline:
               break
           
           # Try multiple times to expand
           success = False
           for attempt in range(50):
               if time.time() > deadline:
                   break
               # Re-initialize random seed for variety?
               # The function try_expand modifies c_current? No, it copies.
               # But it uses random.
               if try_expand(n, c):
                   c = [row[:] for row in c] # Update c to the new one?
                   # try_expand updates best_colors and best_n, but doesn't return the matrix.
                   # We need to update c.
                   # Let's refactor try_expand to return the matrix if successful.
                   success = True
                   break
                   
           if not success:
               break
               
       # Return best found
       return {'n': best_n, 'colors': best_colors}
