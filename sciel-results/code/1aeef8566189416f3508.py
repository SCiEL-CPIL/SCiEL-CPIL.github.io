def construct():
                  import time
                  import random
                  start_time = time.time()
                  deadline = start_time + 85.0 # 5s margin
                  
                  best_set = set()
                  best_size = 0
                  
                  def is_valid_addition(S, x):
                      # Check if adding x to S creates a 4-AP
                      # x is in [1, 512]
                      for d in range(1, 171):
                          # Case 1: x is last term
                          if x - 3*d >= 1:
                              if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                                  return False
                          # Case 2: x is 3rd term
                          if x - 2*d >= 1 and x + d <= 512:
                              if (x - 2*d in S) and (x - d in S) and (x + d in S):
                                  return False
                          # Case 3: x is 2nd term
                          if x - d >= 1 and x + 2*d <= 512:
                              if (x - d in S) and (x + d in S) and (x + 2*d in S):
                                  return False
                          # Case 4: x is 1st term
                          if x + 3*d <= 512:
                              if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                                  return False
                      return True

                  def greedy_construct(seed):
                      random.seed(seed)
                      candidates = list(range(1, 513))
                      random.shuffle(candidates)
                      S = set()
                      for x in candidates:
                          if is_valid_addition(S, x):
                              S.add(x)
                      return S

                  # Phase 1: Multiple Greedy
                  for i in range(200):
                      if time.time() > deadline: break
                      S = greedy_construct(i)
                      if len(S) > best_size:
                          best_size = len(S)
                          best_set = S.copy()

                  # Phase 2: Local Search on best_set
                  # We will try to improve best_set
                  current_set = best_set
                  current_size = best_size
                  
                  # We need a faster way to find conflicts for swapping
                  # For local search, we will try:
                  # 1. Add x (if valid)
                  # 2. Remove y, Add x (if valid)
                  # 3. Remove y, Add x1, Add x2 (if valid) -> Net +1
                  
                  # To make this efficient, we can just iterate and check.
                  # Since N=512, iterating all x is cheap.
                  
                  improved = True
                  while improved and time.time() < deadline:
                      improved = False
                      # Try to add
                      for x in range(1, 513):
                          if x not in current_set:
                              if is_valid_addition(current_set, x):
                                  current_set.add(x)
                                  current_size += 1
                                  improved = True
                                  break # Restart search with new state
                      
                      if improved:
                          continue
                          
                      # Try 1-1 swap (to escape local max)
                      # Pick random y to remove, then try to add
                      # Or systematic:
                      # For each y in current_set:
                      #   S_temp = current_set - {y}
                      #   For each x not in current_set:
                      #      if is_valid_addition(S_temp, x):
                      #         current_set = S_temp union {x}
                      #         improved = True
                      #         break
                      
                      # This is O(|S| * |S^c| * CostCheck)
                      # |S| ~ 200, |S^c| ~ 300. 60k checks.
                      # CostCheck ~ 700 ops.
                      # Total ~ 40M ops. Might be too slow in Python if done many times.
                      
                      # Optimization:
                      # Only try swaps if we are stuck.
                      # Use randomization to avoid getting stuck in a specific cycle.
                      
                      if time.time() > deadline - 5.0:
                          break
                          
                      # Randomized Local Search
                      # Try a few random swaps
                      for _ in range(10):
                          if time.time() > deadline: break
                          y = random.choice(list(current_set))
                          S_temp = current_set.copy()
                          S_temp.remove(y)
                          # Try to add 1 or 2 elements
                          added = 0
                          candidates_to_add = [x for x in range(1, 513) if x not in current_set]
                          random.shuffle(candidates_to_add)
                          
                          for x in candidates_to_add:
                              if added >= 2: break
                              if is_valid_addition(S_temp, x):
                                  S_temp.add(x)
                                  added += 1
                          
                          if len(S_temp) > current_size:
                              current_set = S_temp
                              current_size = len(S_temp)
                              improved = True
                              break
                          elif len(S_temp) == current_size:
                              # Accept swap to explore
                              # Only if it doesn't lead to immediate dead end?
                              # Just accept to randomize
                              if random.random() < 0.5:
                                  current_set = S_temp
                              # Don't mark as improved (size didn't change)
                              # But we break out of the inner loop?
                              # No, we want to try more swaps if possible.
                              # But we already modified current_set? No, S_temp is local.
                              # If we accept, we update current_set.
                              # If we don't improve size, we might loop forever.
                              # Let's only accept size-increasing moves for "improved" flag.
                              # But we need to explore.
                              # Let's accept equal size moves with low probability and continue searching.
                              pass
                          
                      if current_size > best_size:
                          best_size = current_size
                          best_set = current_set.copy()
                          
                  return sorted(best_set)
