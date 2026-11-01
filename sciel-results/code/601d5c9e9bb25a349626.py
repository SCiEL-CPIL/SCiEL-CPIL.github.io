import itertools
import random
import time

def construct(n):
       if n == 1:
           return "1"
       
       # Generate permutations of 1..n-1
       symbols = list(range(1, n))
       perms = list(itertools.permutations(symbols))
       N = len(perms)
       
       # Precompute overlaps
       # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
       overlaps = [[0]*N for _ in range(N)]
       for i in range(N):
           pi = perms[i]
           for j in range(N):
               pj = perms[j]
               # Max possible overlap is n-2
               max_ov = n - 2
               ov = 0
               for k in range(max_ov, -1, -1):
                   if pi[-k:] == pj[:k]:
                       ov = k
                       break
               overlaps[i][j] = ov
       
       best_path = None
       best_len = float('inf')
       
       start_time = time.time()
       time_limit = 110 # Safety margin
       
       restarts = 0
       while time.time() - start_time < time_limit:
           restarts += 1
           start_node = random.randint(0, N-1)
           visited = [False] * N
           visited[start_node] = True
           current = start_node
           path = [start_node]
           current_overlap_sum = 0
           
           for _ in range(N - 1):
               max_ov = -1
               next_node = -1
               # Find neighbor with max overlap
               # To break ties randomly, we can collect all max or just pick first
               # Better: pick randomly from those with max overlap to diversify
               candidates = []
               for j in range(N):
                   if not visited[j]:
                       ov = overlaps[current][j]
                       if ov > max_ov:
                           max_ov = ov
                           candidates = [j]
                       elif ov == max_ov:
                           candidates.append(j)
               
               if not candidates:
                   break # Should not happen
               
               next_node = random.choice(candidates)
               visited[next_node] = True
               current_overlap_sum += max_ov
               current = next_node
               path.append(next_node)
               
           # Calculate length
           # Base length of first block: 2*(n-1) + 1
           # Each next block adds: 2*(n-1) + 1 - overlap
           block_len = 2 * (n - 1) + 1
           total_len = block_len + (N - 1) * block_len - current_overlap_sum
           
           if total_len < best_len:
               best_len = total_len
               best_path = path
               
       # Construct final string
       if best_path is None:
           # Fallback to simple construction if something went wrong
           # This shouldn't happen
           return "".join(map(str, [1]*n)) 
       
       word = []
       first_block = list(perms[best_path[0]]) + [n] + list(perms[best_path[0]])
       word.extend(first_block)
       
       for k in range(1, N):
           prev_p = perms[best_path[k-1]]
           curr_p = perms[best_path[k]]
           curr_block = list(curr_p) + [n] + list(curr_p)
           ov = overlaps[best_path[k-1]][best_path[k]]
           word.extend(curr_block[ov:])
       
       return "".join(map(str, word))
