import time


def construct():
       divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
       LCM = 30030
       n = 10
       
       # Binary search or linear search for W
       # Lower bound 9
       # Upper bound 1000 (safe)
       
       start_time = time.time()
       
       for W in range(9, 200):
           # Check all r
           for r in range(LCM):
               if time.time() - start_time > 50:
                   # Fallback?
                   pass
               
               # Build residues
               residues = [(r + j) % LCM for j in range(W + 1)]
               
               # Build adjacency for matching
               # Left: 0..n-1
               # Right: 0..W
               # Edge i -> j if residues[j] % divisors[i] == 0
               
               # Run matching
               match_right = [-1] * (W + 1)
               
               def dfs(i, visited):
                   for j in range(W + 1):
                       if visited[j]:
                           continue
                       if residues[j] % divisors[i] == 0:
                           visited[j] = True
                           if match_right[j] == -1 or dfs(match_right[j], visited):
                               match_right[j] = i
                               return True
                   return False
               
               # Try to match all
               success = True
               for i in range(n):
                   visited = [False] * (W + 1)
                   if not dfs(i, visited):
                       success = False
                       break
               
               if success:
                   # Construct result
                   # Find A
                   # We need x_i = k*LCM + r + j_i
                   # j_i is the index in residues that was matched to i
                   # match_right[j] = i  =>  j is assigned to i
                   # So for each i, find j such that match_right[j] == i
                   
                   assigned_j = [-1] * n
                   for j in range(W + 1):
                       if match_right[j] != -1:
                           assigned_j[match_right[j]] = j
                   
                   # Pick k such that values are <= 1000000
                   # Max value is k*LCM + r + max_j
                   # Min value is k*LCM + r + min_j
                   # We can just pick k=1 (A = LCM + r)
                   k = 1
                   result = []
                   for i in range(n):
                       j = assigned_j[i]
                       val = k * LCM + r + j
                       result.append(val)
                   
                   # Verify distinctness and divisibility
                   # They are distinct because j are distinct (matching)
                   # Divisibility: (k*LCM + r + j) % divisors[i]
                   # k*LCM is div by divisors[i]
                   # So we need (r+j) % divisors[i] == 0
                   # Which is residues[j] % divisors[i] == 0
                   # Which is guaranteed by the edge condition
                   
                   # Check bounds
                   if all(1 <= v <= 1000000 for v in result):
                       return result
       
       # Fallback (should not happen)
       return divisors[:]
