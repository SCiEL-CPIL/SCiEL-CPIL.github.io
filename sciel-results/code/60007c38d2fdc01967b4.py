import itertools
import random
import time
import math

def construct(v, k, t):
       points = list(range(v))
       all_t_subsets = list(itertools.combinations(points, t))
       subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
       total_subsets = len(all_t_subsets)
       
       # Precompute all possible k-blocks? 
       # C(19,8) = 75582. This is small enough to precompute their t-subsets?
       # 75582 * 56 = ~4.2M operations. Feasible.
       # But storing them might be memory heavy. 
       # Let's generate them on the fly or store indices.
       
       all_blocks = list(itertools.combinations(points, k))
       block_to_t_subsets = []
       for b in all_blocks:
           block_to_t_subsets.append(tuple(sorted(itertools.combinations(b, t))))
           
       # Map subset to blocks that cover it?
       # subset_to_blocks = defaultdict(list)
       # This is 4.2M entries. Might be slow/memory heavy.
       # Let's just use the coverage count array.
       
       def get_coverage(blocks):
           # Returns list of counts for each t-subset
           counts = [0] * total_subsets
           for b in blocks:
               for s in b: # b is a tuple of indices? No, b is a block (tuple of points)
                   # We need the t-subsets of b
                   pass
           # This is slow.
           
       # Better: Represent blocks by their index in all_blocks?
       # State: set of block indices.
       
       def compute_counts(block_indices):
           counts = [0] * total_subsets
           for bi in block_indices:
               subs = block_to_t_subsets[bi]
               for s in subs:
                   counts[subset_to_idx[s]] += 1
           return counts

       def get_uncovered(counts):
           return sum(1 for c in counts if c == 0)

       # SA Parameters
       start_time = time.time()
       time_limit = 115
       
       # Initial Solution: Greedy
       # Greedy: repeatedly pick block that covers most uncovered subsets.
       current_blocks = set()
       counts = [0] * total_subsets
       
       # To speed up greedy, we can precompute for each block, the list of subset indices it covers.
       block_subset_indices = []
       for subs in block_to_t_subsets:
           indices = [subset_to_idx[s] for s in subs]
           block_subset_indices.append(indices)

       # Greedy Construction
       uncovered = [1] * total_subsets # 1 if uncovered
       for _ in range(1000): # Max iterations, usually finishes sooner
           best_gain = 0
           best_block = -1
           # To avoid O(N_blocks * N_subsets) per step, we can just iterate all blocks
           # N_blocks = 75k. N_subsets = 11k. 75k * 11k = 800M ops. Too slow for 1000 steps.
           
           # Optimization: Only consider blocks that cover at least one uncovered subset.
           # Or, use a heuristic: Pick a random uncovered subset, then pick a random block covering it?
           # That's the current program's approach basically.
           
           # Let's use a faster Greedy:
           # 1. Find an uncovered subset.
           # 2. Among all blocks covering it, pick the one that covers the most *other* uncovered subsets.
           
           # To do this efficiently:
           # We need to know which blocks cover a given subset.
           # Let's build subset_to_blocks.
           
       # Building subset_to_blocks:
       subset_to_blocks = [[] for _ in range(total_subsets)]
       for bi, indices in enumerate(block_subset_indices):
           for idx in indices:
               subset_to_blocks[idx].append(bi)
               
       # Now Greedy is fast:
       # 1. Pick random uncovered subset index `u_idx`.
       # 2. Get candidates `subset_to_blocks[u_idx]`.
       # 3. For each candidate, calculate gain (number of uncovered subsets it covers).
       # 4. Pick best.
       
       current_blocks = set()
       counts = [0] * total_subsets
       
       while True:
           # Find uncovered
           uncovered_indices = [i for i, c in enumerate(counts) if c == 0]
           if not uncovered_indices:
               break
           
           u_idx = random.choice(uncovered_indices)
           candidates = subset_to_blocks[u_idx]
           
           best_gain = -1
           best_block = -1
           # Sample candidates if list is large?
           # Max candidates for a t-subset: How many k-blocks contain a specific t-subset?
           # C(v-t, k-t) = C(19-5, 8-5) = C(14, 3) = 364.
           # 364 is small. We can check all.
           
           for bi in candidates:
               gain = 0
               for idx in block_subset_indices[bi]:
                   if counts[idx] == 0:
                       gain += 1
               if gain > best_gain:
                   best_gain = gain
                   best_block = bi
                   
           if best_block == -1:
               break
               
           current_blocks.add(best_block)
           for idx in block_subset_indices[best_block]:
               counts[idx] += 1
               
       # Now we have a valid solution.
       # Apply Simulated Annealing
       
       def get_cost(blocks):
           if not blocks:
               return 10000
           counts = [0] * total_subsets
           for bi in blocks:
               for idx in block_subset_indices[bi]:
                   counts[idx] += 1
           uncovered = sum(1 for c in counts if c == 0)
           if uncovered > 0:
               return len(blocks) + uncovered * 100 # Heavy penalty for invalid
           return len(blocks)
           
       # SA Loop
       best_blocks = set(current_blocks)
       best_cost = get_cost(best_blocks)
       current_blocks = set(best_blocks)
       current_cost = best_cost
       
       T = 10.0
       cooling = 0.99
       
       while time.time() - start_time < time_limit and T > 0.1:
           # Generate neighbor
           new_blocks = set(current_blocks)
           move_type = random.random()
           
           if move_type < 0.3:
               # Add random block
               new_bi = random.randint(0, len(all_blocks)-1)
               new_blocks.add(new_bi)
           elif move_type < 0.6:
               # Remove random block
               if new_blocks:
                   bi_to_remove = random.choice(list(new_blocks))
                   new_blocks.remove(bi_to_remove)
           else:
               # Swap
               if new_blocks:
                   old_bi = random.choice(list(new_blocks))
                   new_bi = random.randint(0, len(all_blocks)-1)
                   new_blocks.remove(old_bi)
                   new_blocks.add(new_bi)
                   
           new_cost = get_cost(new_blocks)
           delta = new_cost - current_cost
           
           if delta < 0 or random.random() < math.exp(-delta / T):
               current_blocks = new_blocks
               current_cost = new_cost
               if current_cost < best_cost:
                   best_cost = current_cost
                   best_blocks = set(current_blocks)
                   
           T *= cooling
           
       # Convert best_blocks (indices) to list of lists of points
       result = []
       for bi in best_blocks:
           result.append(list(all_blocks[bi]))
           
       # Final cleanup: Remove redundant blocks if any (just in case SA left some)
       # This is the same local search as before, but applied to the SA result.
       # Actually, SA minimizes count, so it should be tight.
       
       return result
