import numpy as np
import itertools
import random
import math

def construct(q, n, radius):
       if q != 3 or n != 8 or radius != 1:
           # Fallback
           if n >= 4:
               tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
               return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
           else:
               return [list(x) for x in itertools.product(range(q), repeat=n)]

       # Specific optimization for q=3, n=8, radius=1
       tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
       initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
       
       total_words = 3**n
       
       # Precompute mapping from word to index
       # We need this to map neighbor words to indices for the counts array
       # But we can just use the integer representation of the word.
       
       def word_to_int(word):
           val = 0
           for i in range(n):
               val = val * 3 + word[i]
           return val

       def int_to_word(val):
           word = [0] * n
           for i in range(n-1, -1, -1):
               word[i] = val % 3
               val //= 3
           return word

       # Precompute neighbor indices for each initial codeword
       cw_neighbors = []
       for cw in initial_codewords:
           base_int = word_to_int(cw)
           neigh_ints = [base_int]
           for i in range(n):
               # Calculate step for position i
               # Value at i is (base_int // (3**i)) % 3
               # Changing it by +1 or -1
               power = 3**i
               current_val = (base_int // power) % 3
               if current_val > 0:
                   neigh_ints.append(base_int - power)
               if current_val < 2:
                   neigh_ints.append(base_int + power)
           cw_neighbors.append(neigh_ints)

       # Initialize
       counts = np.zeros(total_words, dtype=np.int32)
       for neigh_list in cw_neighbors:
           counts[neigh_list] += 1
           
       active = [True] * len(initial_codewords)
       
       best_size = len(initial_codewords)
       best_active = active.copy()
       
       # Simulated Annealing
       T = 5.0
       T_min = 0.01
       cooling = 0.995
       steps = 20000 # Adjust based on time
       
       # To speed up, we can run multiple short SAs or one long one.
       # Let's do one long one with restarts if stuck.
       
       # Optimization: Keep track of "critical" neighbors?
       # For now, just check min.
       
       import time
       start_time = time.time()
       
       while T > T_min and (time.time() - start_time) < 100: # Leave 20s buffer
           # Pick random active index
           active_indices = [i for i in range(len(active)) if active[i]]
           if not active_indices:
               break
           i = random.choice(active_indices)
           
           # Tentatively remove
           active[i] = False
           neigh = cw_neighbors[i]
           
           # Check coverage
           # We need to know if any count in neigh becomes 0
           # min_count = np.min(counts[neigh])
           # If min_count == 1, then after removal, it becomes 0.
           # Number of such neighbors is the penalty.
           
           # To be efficient, we can just check.
           # But updating counts is needed if we accept.
           
           # Let's check first without updating counts globally, 
           # or update and rollback.
           # Update and rollback is easier.
           
           counts[neigh] -= 1
           
           # Find number of zeros in counts[neigh]
           # zeros = np.sum(counts[neigh] == 0)
           # This is O(17) per step. 20000 * 17 is small.
           
           uncovered = np.sum(counts[neigh] == 0)
           
           if uncovered == 0:
               # Valid move, always accept
               current_size = sum(active)
               if current_size < best_size:
                   best_size = current_size
                   best_active = active.copy()
               # Do not cool? Or cool slightly?
               # Usually accept and cool.
           else:
               # Invalid move
               # Energy change:
               # We reduced size by 1, but introduced 'uncovered' errors.
               # Let's define Energy = Size + M * Uncovered
               # We want to minimize Energy.
               # Delta Size = -1
               # Delta Uncovered = uncovered (approx, assuming no overlap with other uncovered? 
               # Actually, if a word was already uncovered, removing c doesn't change its status.
               # But we only track counts. If count was 1, it becomes 0.
               # If count was >1, it stays >0.
               # So 'uncovered' is the number of words that BECAME uncovered.
               # Total uncovered in system might be higher if we are in an invalid state.
               # But we start valid.
               # So if we are in a valid state, uncovered is the new errors.
               
               # Accept with prob exp(-DeltaE / T)
               # DeltaE = -1 + M * uncovered
               # Choose M large enough to discourage invalid states.
               # M = 10?
               M = 10
               delta_e = -1 + M * uncovered
               if delta_e < 0:
                   accept = True
               else:
                   accept = random.random() < math.exp(-delta_e / T)
                   
               if not accept:
                   # Rollback
                   counts[neigh] += 1
                   active[i] = True
           
           T *= cooling
           
       # Restore best
       # Note: best_active might be from a valid state.
       # We need to reconstruct the code.
       
       final_codewords = [initial_codewords[i] for i in range(len(initial_codewords)) if best_active[i]]
       
       return final_codewords
