   def max_independent_set_size(n, adj_masks):
       best = 0
       # candidates is a bitmask of available vertices
       # current_size is the size of the independent set so far
       
       def dfs(candidates, current_size):
           nonlocal best
           if not candidates:
               if current_size > best:
                   best = current_size
               return
           
           # Upper bound: current_size + popcount(candidates)
           if current_size + candidates.bit_count() <= best:
               return
               
           # Pick a vertex
           # Heuristic: pick vertex with max degree in candidates?
           # Or just lowest bit?
           # Let's pick the vertex with the highest degree in the subgraph induced by candidates
           # This is expensive to compute each time.
           # Simple heuristic: pick the first available vertex.
           # Better: pick the vertex that is adjacent to the most other candidates?
           # Let's try picking the vertex with the maximum number of neighbors in candidates.
           
           # To speed up, we can just pick the lowest set bit.
           v_bit = candidates & -candidates
           v = (v_bit.bit_length() - 1)
           
           # Option 1: Include v
           # New candidates: candidates without v, and without neighbors of v
           new_cands = candidates & ~adj_masks[v] & ~v_bit
           dfs(new_cands, current_size + 1)
           
           # Option 2: Exclude v
           dfs(candidates ^ v_bit, current_size)

       dfs((1 << n) - 1, 0)
       return best
   