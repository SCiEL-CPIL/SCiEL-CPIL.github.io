import numpy as np
import time

N = 400

def construct():
       # Deadline guard
       start_time = time.time()
       deadline = start_time + 140  # 10s margin

       # Initialize with random
       x = np.random.choice([-1, 1], N)
       
       # Calculate initial cost
       # R = np.correlate(x, x, mode='full')
       # Sidelobes are R[N-1+1:N-1+1+N-1]? 
       # np.correlate(x, x, 'full') length 2N-1.
       # Lag 0 is at index N-1.
       # Sidelobes are indices 0..N-2 and N..2N-2.
       
       def get_cost(x):
           r = np.correlate(x, x, mode='full')
           # r[N-1] is lag 0
           # We want max abs of r excluding center
           # r[:N-1] and r[N:]
           left = r[:N-1]
           right = r[N:]
           return np.max(np.abs(np.concatenate([left, right])))

       current_cost = get_cost(x)
       best_x = x.copy()
       best_cost = current_cost

       # Greedy Bit Flipping
       # Try flipping each bit, if cost decreases, keep it.
       # Repeat until no improvement or time out.
       
       # To speed up, we can update R incrementally?
       # Flipping bit i changes R(k) by:
       # R_new(k) = R_old(k) - 2*x_i*(x_{i+k} + x_{i-k}) + ...?
       # Simpler: Just recalculate. O(N^2) is fast enough for 150s.
       # 150s / (160k ops) is huge.
       # Let's do full recalc.

       improved = True
       while improved and time.time() < deadline:
           improved = False
           for i in range(N):
               if time.time() > deadline:
                   break
               # Flip
               x[i] *= -1
               new_cost = get_cost(x)
               if new_cost < current_cost:
                   current_cost = new_cost
                   if new_cost < best_cost:
                       best_cost = new_cost
                       best_x = x.copy()
                   improved = True
               else:
                   # Revert
                   x[i] *= -1
                   
       return best_x
