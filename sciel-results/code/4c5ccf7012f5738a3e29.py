   def construct():
       start_time = time.time()
       deadline = start_time + 85.0
       
       best_set = [31, 32, 34]
       
       def check_addition(current_sums, x):
           # Returns True if x can be added
           # current_sums is a set of ints
           if x in current_sums:
               return False
           
           new_sums_list = [x]
           for s in current_sums:
               new_sums_list.append(s + x)
               
           new_sums = set(new_sums_list)
           if len(new_sums) != len(new_sums_list):
               return False
           
           # Check intersection with current
           if new_sums & current_sums:
               return False
           
           # Check divisibility
           # We need to check pairs within new_sums, and between new_sums and current_sums
           # Optimization: if n1 < n2, check n2 % n1 == 0
           
           # Convert to lists for sorting/indexing if needed, but sets are fine for iteration
           # To speed up, we can just iterate.
           # M is small.
           
           # Check new vs current
           for n in new_sums:
               for c in current_sums:
                   if n < c:
                       if c % n == 0:
                           return False
                   else:
                       if n % c == 0:
                           return False
           
           # Check new vs new
           new_list = list(new_sums)
           new_list.sort()
           for i in range(len(new_list)):
               n1 = new_list[i]
               for j in range(i+1, len(new_list)):
                   n2 = new_list[j]
                   if n2 % n1 == 0:
                       return False
           
           return True
       
       def get_sums(elements):
           sums = set()
           for mask in range(1, 1 << len(elements)):
               s = 0
               for i in range(len(elements)):
                   if mask & (1 << i):
                       s += elements[i]
               sums.add(s)
           return sums
       
       # Initial state
       current_elements = [31, 32, 34]
       current_sums = get_sums(current_elements)
       
       # Try to expand
       # We'll use a random search strategy
       # 1. Try to add a random element from 1..256 not in current_elements
       # 2. If success, keep it.
       # 3. If fail, try swapping.
       
       # To avoid getting stuck in local minima, we can restart with different random seeds/initial sets
       
       for attempt in range(1000):
           if time.time() > deadline:
               break
           
           # Randomly perturb current_elements
           # 50% chance: try to add
           # 50% chance: try to swap one out, one in
           
           if random.random() < 0.5:
               # Try add
               candidates = [x for x in range(1, 257) if x not in current_elements]
               random.shuffle(candidates)
               for x in candidates[:10]: # Try 10 random candidates
                   if check_addition(current_sums, x):
                       current_elements.append(x)
                       current_sums = get_sums(current_elements) # Recalculate for simplicity, or update incrementally
                       # Update current_sums incrementally for speed?
                       # get_sums is O(2^N * N). For N=10, 10k ops. Fine.
                       break
           else:
               # Try swap
               if not current_elements:
                   continue
               idx = random.randint(0, len(current_elements)-1)
               old_elem = current_elements[idx]
               new_elements = current_elements[:idx] + current_elements[idx+1:]
               new_sums = get_sums(new_elements)
               
               candidates = [x for x in range(1, 257) if x not in new_elements]
               random.shuffle(candidates)
               for x in candidates[:10]:
                   if check_addition(new_sums, x):
                       current_elements = new_elements + [x]
                       current_sums = get_sums(current_elements)
                       break
                       
           # Track best
           if len(current_elements) > len(best_set):
               best_set = list(current_elements)
               
       return best_set
   