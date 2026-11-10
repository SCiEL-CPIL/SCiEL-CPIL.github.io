import time
import sys

def construct():
       # 1. Greedy Solution
       values, sums = [], set()
       for x in range(1, 257):
           tails, total = [x], x
           for a in reversed(values):
               total += a
               tails.append(total)
           if not sums.intersection(tails):
               values.append(x)
               sums.update(tails)
       
       greedy_len = len(values)
       best_seq = values
       
       # 2. IDDFS to try for greedy_len + 1
       # If that works, maybe try +2?
       # Keep it simple: try +1. If fast, try +2.
       
       def can_build_length(target_len, start_idx, current_vals, current_sums, time_limit):
           # current_vals: list
           # current_sums: set
           # start_idx: index in range(1, 257) to start checking from?
           # No, we need to pick the next value > current_vals[-1]
           
           if time.time() > time_limit:
               return None
               
           if len(current_vals) == target_len:
               return list(current_vals)
               
           last_val = current_vals[-1] if current_vals else 0
           
           # Pruning: max possible elements we can add
           # We have 256 - last_val numbers left.
           # We need target_len - len(current_vals) more.
           if 256 - last_val < target_len - len(current_vals):
               return None
               
           # Try candidates
           for x in range(last_val + 1, 257):
               # Check collision
               tails, total = [x], x
               valid = True
               for a in reversed(current_vals):
                   total += a
                   tails.append(total)
                   if total in current_sums:
                       valid = False
                       break
               if valid:
                   # Also check if any tail is in current_sums
                   # The loop above checks `total` which is the tail sum.
                   # But we need to check ALL tails against current_sums.
                   # Wait, the loop `for a in reversed...` generates tails in order.
                   # We break on first collision.
                   # If no collision in loop, we still need to check if any `tails` are in `current_sums`?
                   # No, `total` IS the tail sum. So checking `total in current_sums` covers it.
                   
                   # Also, we must ensure `tails` don't collide with *each other*?
                   # Tails are strictly increasing?
                   # x, x+last, x+last+prev...
                   # Since values are strictly increasing, the tails are strictly increasing.
                   # So no internal collisions.
                   
                   if not any(t in current_sums for t in tails):
                       # Recurse
                       new_vals = current_vals + [x]
                       new_sums = current_sums.union(tails)
                       res = can_build_length(target_len, x, new_vals, new_sums, time_limit)
                       if res:
                           return res
           
           return None

       time_limit = time.time() + 50 # 50 seconds
       
       # Try to extend by 1
       target = greedy_len + 1
       # We need to reset the search.
       # The DFS above starts from empty?
       # No, we want to find *any* sequence of length `target`.
       # So we start with empty list.
       
       # But starting from empty is slow.
       # We can use the greedy solution as a guide?
       # Or just start from scratch with strong pruning.
       
       # Let's start from scratch.
       res = can_build_length(target, 0, [], set(), time_limit)
       
       if res:
           return res
       
       return best_seq
