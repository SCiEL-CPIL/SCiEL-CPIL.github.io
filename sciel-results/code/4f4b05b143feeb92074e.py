   def heuristic_construct(q, n, radius):
       # Direct Sum Initialization
       if q == 3:
           # Use tetracode for first 4, all for rest
           tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
           init_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
       else:
           # Trivial: all words
           init_code = [list(w) for w in itertools.product(range(q), repeat=n)]
           
       # Local Search
       # Maintain cover counts
       N = q ** n
       # Map word to index
       def word_to_idx(w):
           idx = 0
           for d in range(n):
               idx = idx * q + w[d]
           return idx
           
       cover_count = np.zeros(N, dtype=int)
       current_code = init_code.copy()
       
       # Initialize counts
       for c in current_code:
           idx_c = word_to_idx(c)
           # Add 1 to all neighbors of c
           # Generate neighbors
           for delta in generate_neighbors(c, q, n, radius):
               idx_d = word_to_idx(delta)
               cover_count[idx_d] += 1
               
       best_code = current_code.copy()
       best_size = len(current_code)
       
       # Iterative Improvement
       for _ in range(100): # Max iterations
           improved = False
           # Try to remove each codeword
           for i, c in enumerate(current_code):
               # Check if c is redundant
               # Temporarily remove c
               idx_c = word_to_idx(c)
               # We need to know if any word becomes uncovered
               # This requires checking if any neighbor of c has cover_count == 1
               # And that neighbor is ONLY covered by c?
               # No, cover_count counts ALL codewords.
               # If we remove c, cover_count[neighbor] decreases by 1.
               # If it becomes 0, that neighbor is uncovered.
               
               # We can simulate removal
               is_redundant = True
               # Check all neighbors of c
               for delta in generate_neighbors(c, q, n, radius):
                   idx_d = word_to_idx(delta)
                   if cover_count[idx_d] == 1:
                       is_redundant = False
                       break
               
               if is_redundant:
                   # Remove c
                   for delta in generate_neighbors(c, q, n, radius):
                       idx_d = word_to_idx(delta)
                       cover_count[idx_d] -= 1
                   current_code.pop(i)
                   improved = True
                   if len(current_code) < best_size:
                       best_size = len(current_code)
                       best_code = current_code.copy()
                   break # Restart loop after modification
                   
           if not improved:
               break
               
       return best_code
   