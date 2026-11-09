def construct():
       p = 1009
       inv = [0] * p
       for i in range(1, p):
           inv[i] = pow(i, p - 2, p)
           
       max_len = 0
       best_start = 1
       
       for s in range(2, p - 1): # s from 2 to 1007
           # If max possible length from s is <= max_len, we might skip?
           # Max possible e is 1007. Length is 1007 - s + 1.
           if 1007 - s + 1 <= max_len:
               break
               
           valid = True
           for e in range(s, 1008): # e from s to 1007
               # Check if adding e breaks the interval [s, e]
               # 1. e itself must not be self-inverse (1 or 1008).
               #    e >= 2 and e <= 1007, so e != 1 and e != 1008. OK.
               # 2. inv[e] must not be in [s, e].
               #    If inv[e] >= s and inv[e] <= e, then it's in the interval.
               if s <= inv[e] <= e:
                   valid = False
                   break
               
               # If we reach here, [s, e] is valid so far?
               # Wait. If we break out of inner loop, we found a break.
               # If we don't break, the current [s, e] is valid.
               # We want the MAX e.
               # So we should record the length for each valid e.
               
               current_len = e - s + 1
               if current_len > max_len:
                   max_len = current_len
                   best_start = s
                   
       return [best_start, max_len]
