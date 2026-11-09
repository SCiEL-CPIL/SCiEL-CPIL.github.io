import math
import time

def construct():
           denominators = list(range(2, 49))
           L = 1
           for d in denominators:
               L = math.lcm(L, d)
               
           # (weight, index)
           items = [(L // d, i) for i, d in enumerate(denominators)]
           # Sort by weight descending to maximize sum quickly
           items.sort(key=lambda x: x[0], reverse=True)
           
           best_sum = 0
           best_indices = []
           
           start_time = time.time()
           timeout = 50.0 # Safety margin for 60s limit
           
           def dfs(idx, current_sum, current_indices):
               nonlocal best_sum, best_indices
               
               if time.time() - start_time > timeout:
                   return
                   
               if current_sum > best_sum:
                   best_sum = current_sum
                   best_indices = current_indices[:]
                   
               if idx == len(items):
                   return
                   
               # Upper bound check
               # Sum of remaining weights
               # Precompute suffix sums of weights
               # suffix_sum[idx] = sum of weights from idx to end
               
               # If current_sum + suffix_sum[idx] <= best_sum:
               #     return
               
               # Option 1: Exclude item idx
               dfs(idx + 1, current_sum, current_indices)
               
               # Option 2: Include item idx
               w, i = items[idx]
               if current_sum + w < L:
                   current_indices.append(i)
                   dfs(idx + 1, current_sum + w, current_indices)
                   current_indices.pop()
                   
           # Precompute suffix sums
           suffix_sums = [0] * (len(items) + 1)
           for i in range(len(items) - 1, -1, -1):
               suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
               
           # Add suffix sum check to dfs
           # Modify dfs to use suffix_sums
           
           # Re-implement dfs with suffix sums
           def dfs_opt(idx, current_sum, current_indices):
               nonlocal best_sum, best_indices
               
               if time.time() - start_time > timeout:
                   return
                   
               if current_sum > best_sum:
                   best_sum = current_sum
                   best_indices = current_indices[:]
                   
               if idx == len(items):
                   return
                   
               # Prune if even taking all remaining items doesn't beat best
               if current_sum + suffix_sums[idx] <= best_sum:
                   return
                   
               # Exclude
               dfs_opt(idx + 1, current_sum, current_indices)
               
               # Include
               w, i = items[idx]
               if current_sum + w < L:
                   current_indices.append(i)
                   dfs_opt(idx + 1, current_sum + w, current_indices)
                   current_indices.pop()
                   
           dfs_opt(0, 0, [])
           
           return best_indices
