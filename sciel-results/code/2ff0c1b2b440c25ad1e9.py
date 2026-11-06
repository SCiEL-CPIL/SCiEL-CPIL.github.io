def construct():
      import time
      start_time = time.time()
      
      # 1. Greedy Initialization
      vals = []
      prods = set()
      
      def try_add(x):
          nonlocal vals, prods
          # Calculate new tails
          tails = []
          prod = x
          for a in reversed(vals):
              tails.append(prod)
              prod *= a
          tails.append(x) # Wait, the loop logic in seed:
          # Seed:
          # tails, prod = [x], x
          # for a in reversed(values):
          #    prod *= a
          #    tails.append(prod)
          # This generates x, x*last, x*last*second_last, etc.
          
          new_tails = [x]
          p = x
          for a in reversed(vals):
              p *= a
              new_tails.append(p)
              
          if not prods.intersection(new_tails):
              vals.append(x)
              prods.update(new_tails)
              return True
          return False

      for x in range(2, 129):
          try_add(x)
          
      best_len = len(vals)
      
      # 2. Local Search
      # We want to maximize length.
      # We'll try to add elements. If we can't, we'll try swaps to enable additions.
      
      def get_prods(v):
          ps = set()
          for i in range(len(v)):
              p = 1
              for j in range(i, len(v)):
                  p *= v[j]
                  ps.add(p)
          return ps

      def is_valid(v):
          ps = get_prods(v)
          return len(ps) == len(v)*(len(v)+1)//2

      current_vals = list(vals)
      
      while time.time() - start_time < 50: # Safety margin
          improved = False
          
          # Try to add any missing number
          added = False
          for x in range(2, 129):
              if x not in current_vals:
                  # Check if x can be added
                  # Since order is sorted, x will be inserted at correct position
                  # But try_add logic assumes appending?
                  # No, the product set is independent of order for uniqueness check?
                  # Wait. Subarray products depend on order.
                  # But the set of subarray products for a SET of numbers is NOT the same for different orders.
                  # However, the problem states "strictly increasing integers".
                  # So the order IS fixed by the values.
                  # So if we have a set S, the sequence is sorted(S).
                  # The products are determined by the sorted order.
                  
                  # So we must check validity of sorted(current_vals + [x])
                  candidate = sorted(current_vals + [x])
                  if is_valid(candidate):
                      current_vals = candidate
                      improved = True
                      added = True
                      break
          
          if added:
              continue
          
          # Try swaps
          # To avoid O(N*M) checks being too slow, we can be smart.
          # But N is small.
          swapped = False
          # We want to try swaps that might allow future additions.
          # Heuristic: swap a "large" element with a "smaller" missing one?
          # Or just iterate all pairs.
          for i, a in enumerate(current_vals):
              for b in range(2, 129):
                  if b in current_vals:
                      continue
                  # Try swapping a with b
                  # Ensure b != a
                  candidate = current_vals[:]
                  candidate[i] = b
                  candidate.sort()
                  if is_valid(candidate):
                      # Accept swap?
                      # We accept it if it doesn't reduce length (it doesn't) 
                      # and we hope it opens up space.
                      # To prevent cycling, we can require that the "sum" of elements changes?
                      # Or just track visited sets.
                      # Given time limit, we can just accept and restart the add-loop.
                      current_vals = candidate
                      swapped = True
                      break
              if swapped:
                  break
                  
          if not swapped:
              # No swaps found? Or all swaps invalid?
              # If no valid swaps at all, we are stuck.
              # But there are usually valid swaps (e.g. swapping primes).
              # If we cycle, we need a mechanism to stop.
              # Let's add a randomization or a move counter.
              # If we've been swapping without adding, we might cycle.
              # Let's limit the number of consecutive swaps without an addition.
              pass

      return current_vals
