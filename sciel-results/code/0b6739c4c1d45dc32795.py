   def dfs():
       if time.time() > deadline:
           return None
       if not missing:
           return list(current_set)
       
       last = current_set[-1]
       # Try adding x
       # We can limit x?
       # If x is too large, it might not help cover small diffs?
       # But it can cover large diffs.
       
       # Heuristic: Sort candidates by how many missing diffs they cover
       candidates = []
       for x in range(last + 1, M + 1):
           # Check conflict
           conflict = False
           new_diffs = []
           for a in current_set:
               d = x - a
               if d in used_diffs:
                   conflict = True
                   break
               new_diffs.append(d)
           if conflict:
               continue
           
           # Count how many missing diffs are covered
           covered = 0
           for d in new_diffs:
               if d in missing:
                   covered += 1
           
           # Only add if it covers at least 1 missing diff?
           # If it covers 0, it might still be needed for future?
           # E.g., if we need diff 30, and we have {0, 1}, adding 2 covers 2, 1.
           # Adding 30 covers 30, 29.
           # If we skip 30, we might not be able to cover 30 later?
           # If we add 31, diff 31, 30. 30 is covered.
           # So covering 0 is possible.
           # However, adding elements that cover 0 missing diffs increases the set size and might cause conflicts.
           # Let's prefer those that cover > 0.
           
           candidates.append((covered, x, new_diffs))
       
       # Sort by covered desc, then x asc
       candidates.sort(key=lambda t: (-t[0], t[1]))
       
       for covered, x, new_diffs in candidates:
           # If covered == 0 and we have many missing, maybe skip?
           # Let's just try all.
           
           current_set.append(x)
           used_diffs.update(new_diffs)
           missing_update = missing - set(new_diffs)
           
           res = dfs()
           if res:
               return res
           
           current_set.pop()
           used_diffs.difference_update(new_diffs)
       
       return None
   