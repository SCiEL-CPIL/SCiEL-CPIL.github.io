   def construct():
       # Candidate set: 144 to 192, but remove 147
       # Size: 49 - 1 = 48
       S = [x for x in range(144, 193) if x != 147]
       
       # Verify
       def is_valid(S):
           S_set = set(S)
           for a in S:
               others = [x for x in S if x != a]
               for x, y, z in combinations(others, 3):
                   if (x + y + z) % a == 0:
                       return False
           return True
       
       if is_valid(S):
           return S
       else:
           # Fallback to greedy (seed)
           selected = []
           for x in range(192, 0, -1):
               candidate = selected + [x]
               if all(sum(t) % a for a in candidate for t in combinations([v for v in candidate if v != a], 3)):
                   selected.append(x)
           return selected
   