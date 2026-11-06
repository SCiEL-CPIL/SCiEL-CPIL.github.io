import time
import random
import numpy as np

def construct():
       start_time = time.time()
       deadline = 110.0 # Safety margin
       
       n = 50
       # Initialize colors
       colors = np.random.randint(0, 4, size=(n, n))
       # Symmetric
       colors = (colors + colors.T) // 2
       np.fill_diagonal(colors, 0)
       
       def count_mono_triangles(colors):
           count = 0
           for i in range(n):
               for j in range(i+1, n):
                   for k in range(j+1, n):
                       if colors[i,j] == colors[j,k] == colors[i,k]:
                           count += 1
           return count
       
       best_count = count_mono_triangles(colors)
       best_colors = colors.copy()
       
       if best_count == 0:
           return {'n': n, 'colors': [int(colors[i,j]) for i in range(n) for j in range(i+1, n)]}
       
       # Local Search
       while time.time() < deadline:
           # Pick random edge
           i = random.randint(0, n-1)
           j = random.randint(0, n-1)
           if i == j: continue
           if i > j: i, j = j, i
           
           current_color = colors[i,j]
           best_delta = float('inf')
           best_new_color = current_color
           
           for c in range(4):
               if c == current_color: continue
               
               # Calculate delta
               # Triangles involving (i,j): (i,j,k)
               # Delta = change in mono-triangles
               # Old: count of k where colors[i,k]==colors[j,k]==current_color
               # New: count of k where colors[i,k]==colors[j,k]==c
               
               # Optimization: Precompute?
               # For now, simple loop is O(n)
               delta = 0
               for k in range(n):
                   if k == i or k == j: continue
                   
                   old_c1 = colors[i,k]
                   old_c2 = colors[j,k]
                   old_c3 = current_color
                   
                   new_c1 = old_c1
                   new_c2 = old_c2
                   new_c3 = c
                   
                   old_mono = 1 if (old_c1 == old_c2 == old_c3) else 0
                   new_mono = 1 if (new_c1 == new_c2 == new_c3) else 0
                   
                   delta += new_mono - old_mono
               
               if delta < best_delta:
                   best_delta = delta
                   best_new_color = c
           
           # Accept if improvement, or with probability if worsening (Simulated Annealing)
           if best_delta < 0:
               colors[i,j] = best_new_color
               colors[j,i] = best_new_color
               # Recount? Or update incrementally?
               # Recounting is O(n^3) = 125k. 120s allows ~1000 recounts?
               # 1000 * 125k = 125M ops. Might be tight in Python.
               # Let's do incremental update.
               # We already calculated delta.
               best_count += best_delta
               if best_count < best_count: # Wait, best_count is global min?
                   # Update global best if we found a new minimum
                   pass
               
               if best_count == 0:
                   return {'n': n, 'colors': [int(colors[i,j]) for i in range(n) for j in range(i+1, n)]}
           else:
               # Random restart or accept worse?
               # Try random restart if stuck
               if random.random() < 0.01:
                   colors = np.random.randint(0, 4, size=(n, n))
                   colors = (colors + colors.T) // 2
                   np.fill_diagonal(colors, 0)
                   best_count = count_mono_triangles(colors)
                   if best_count == 0:
                       return {'n': n, 'colors': [int(colors[i,j]) for i in range(n) for j in range(i+1, n)]}
       
       # Return best found (might not be 0)
       # But task requires valid coloring. If not 0, it's invalid.
       # So we must return a valid one.
       # Fall back to n=32 seed if n=50 fails.
       
       # Re-implement seed for fallback
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
       colors32 = []
       for i in range(32):
           for j in range(i + 1, 32):
               colors32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
       return {'n': 32, 'colors': colors32}
