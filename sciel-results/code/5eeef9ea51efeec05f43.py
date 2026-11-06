import time
import itertools

def construct():
      start_time = time.time()
      deadline = start_time + 110  # 10s margin
      
      # Candidates: Paley graphs for primes q = 1 mod 4
      # q=17 is seed.
      # Try q=29, 37, 41, 53
      # Note: q must be prime for standard Paley, but quadratic residues work for any odd q?
      # Standard Paley is for prime powers q = 1 mod 4.
      
      def is_paley(q):
          # Check if q is prime and q % 4 == 1
          if q < 2: return False
          for i in range(2, int(q**0.5)+1):
              if q % i == 0: return False
          return q % 4 == 1

      def generate_paley(q):
          squares = set()
          for x in range(1, q):
              squares.add((x*x) % q)
          n = q
          edges = []
          for i in range(n):
              for j in range(i+1, n):
                  if (j-i) % q in squares:
                      edges.append([i, j])
          return n, edges

      def check_valid(n, edges):
          # Check K4
          # Build adjacency matrix or set
          adj = [set() for _ in range(n)]
          for u, v in edges:
              adj[u].add(v)
              adj[v].add(u)
          
          # Check K4: any 4 vertices form a clique
          # Optimize: check triangles first?
          # Just check all 4-subsets? \binom{40}{4} = 91390. Very fast.
          for comb in itertools.combinations(range(n), 4):
              u, v, w, x = comb
              if v in adj[u] and w in adj[u] and x in adj[u] and \
                 w in adj[v] and x in adj[v] and \
                 x in adj[w]:
                  return False, "K4 found"
                  
          # Check alpha < 6: no independent set of size 6
          # Check all 6-subsets
          # \binom{40}{6} = 3,838,380.
          # Python loop might take ~2-5 seconds.
          # We can break early if time is tight.
          
          # Optimization: Use bitsets for independence check
          adj_mask = [0] * n
          for u, v in edges:
              adj_mask[u] |= (1 << v)
              adj_mask[v] |= (1 << u)
              
          def is_independent(s):
              # s is a tuple of 6 vertices
              # Check all pairs
              for i in range(len(s)):
                  for j in range(i+1, len(s)):
                      if adj_mask[s[i]] & (1 << s[j]):
                          return False
              return True

          # To speed up, we can prune.
          # But 3.8M iterations in Python is okay within 10s.
          
          # Check deadline
          if time.time() > deadline - 2:
              return None, "timeout"
              
          for comb in itertools.combinations(range(n), 6):
              if is_independent(comb):
                  return False, "Ind6 found"
              
              if time.time() > deadline - 2:
                  return None, "timeout"
                  
          return True, "valid"

      best_n = 0
      best_edges = []
      
      # Try known good q values
      # 17 is known good.
      # 29 is likely good.
      # 37 is likely good or close.
      
      candidates = [17, 29, 37, 41, 53]
      
      for q in candidates:
          if not is_paley(q):
              continue
          
          n, edges = generate_paley(q)
          
          # Quick sanity check on size
          if n <= best_n:
              continue
              
          valid, reason = check_valid(n, edges)
          
          if valid:
              best_n = n
              best_edges = edges
          elif reason == "timeout":
              # Don't trust this one, but if it's larger than current best and we ran out of time,
              # we might want to return the previous best.
              pass
          else:
              # Invalid, try next smaller? No, we want max n.
              # If 37 is invalid, 29 might be valid.
              pass

      if best_n == 0:
          # Fallback to 17
          n, edges = generate_paley(17)
          best_n = n
          best_edges = edges

      return {'n': best_n, 'edges': best_edges}
