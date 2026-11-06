import random
import time

def construct():
       best_set = []
       start_time = time.time()
       
       def run_greedy(candidates):
           A = []
           sum_set = set()
           
           for x in candidates:
               if time.time() - start_time > 55:
                   return A
               
               k = len(A)
               s_new = set()
               s_new.add(3 * x)
               for y in A:
                   s_new.add(2 * x + y)
               for i in range(k):
                   y = A[i]
                   for j in range(i, k):
                       z = A[j]
                       s_new.add(x + y + z)
               
               num_expected = (k + 1) * (k + 2) // 2
               
               if len(s_new) != num_expected:
                   continue
                   
               if s_new & sum_set:
                   continue
               
               A.append(x)
               sum_set.update(s_new)
               
           return A

       # Try Smallest First
       A1 = run_greedy(range(1, 4097))
       if len(A1) > len(best_set):
           best_set = A1
           
       # Try Largest First
       A2 = run_greedy(range(4096, 0, -1))
       if len(A2) > len(best_set):
           best_set = A2

       # Try Random
       for trial in range(500):
           if time.time() - start_time > 55:
               break
           
           candidates = list(range(1, 4097))
           random.shuffle(candidates)
           A = run_greedy(candidates)
           
           if len(A) > len(best_set):
               best_set = A
               
       return best_set
