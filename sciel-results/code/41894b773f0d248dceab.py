def construct():
      import time
      start_time = time.time()
      deadline = start_time + 50 # 50s limit
      
      # Start with a large set
      S = set(range(129, 257))
      
      def get_violation(S_list):
          n = len(S_list)
          # Sort for determinism
          S_list = sorted(S_list)
          for i in range(n):
              a = S_list[i]
              for j in range(n):
                  if i == j: continue
                  b = S_list[j]
                  for k in range(n):
                      if i == k or j == k: continue
                      c = S_list[k]
                      if (b * c) % a == 0:
                          return (a, b, c)
          return None

      # Ensure initial set is valid
      while time.time() < deadline:
          S_list = list(S)
          viol = get_violation(S_list)
          if viol is None:
              break
          a, b, c = viol
          # Remove a
          S.remove(a)
          
      # If we ran out of time, return what we have (might be invalid, so check)
      # But we must return a VALID set.
      # So we need a fallback.
      
      # Try to add more numbers from [2, 128]
      candidates = list(range(2, 129))
      # Sort candidates? Maybe larger first? Or smaller?
      # Smaller numbers are more likely to divide products, so they are "riskier" to add.
      # Larger numbers are safer.
      candidates.sort(reverse=True)
      
      for x in candidates:
          if time.time() > deadline:
              break
          if x in S:
              continue
          new_S = S | {x}
          new_list = list(new_S)
          if get_violation(new_list) is None:
              S = new_S
              
      # Final check and cleanup if necessary (should be valid)
      S_list = list(S)
      # If the loop broke due to time, S might be invalid if we didn't finish checking?
      # No, we only add if valid. And we removed until valid.
      # But if we exited the first while loop due to time, S might be invalid.
      # So we must ensure validity.
      
      # Robust cleanup:
      # If S is invalid, we must fix it.
      # Given the time, just return the primes if in doubt?
      # Or do a quick check.
      
      if get_violation(list(S)) is not None:
          # Fallback to primes
          return [a for a in range(2, 257) if all(a%d for d in range(2, int(a**0.5)+1))]
          
      return list(S)
