   best_t = -1
   best_count = -1
   for t in range(N):
       c = 0
       for k in triple_to_Ks[t]:
           if uncovered[k]:
               c += 1
       if c > best_count:
           best_count = c
           best_t = t
           if c == 0: break
   