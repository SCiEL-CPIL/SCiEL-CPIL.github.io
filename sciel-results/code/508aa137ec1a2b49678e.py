   def bnb(available, included, current_weight):
       # 1. Greedy LB
       greedy_w, greedy_set = greedy(available, included)
       if current_weight + greedy_w > best_weight:
           best_weight = current_weight + greedy_w
           best_set = list(included) + greedy_set
       
       # 2. UB
       # Sum of weights in available
       # Optimization: Maintain sum_available?
       sum_avail = sum(weights[v] for v in available) # O(N)
       if current_weight + sum_avail <= best_weight:
           return
           
       # 3. Branch
       if not available:
           return
       
       # Choose v
       # Heuristic: Highest weight?
       v = max(available, key=lambda x: weights[x])
       
       # Option 1: Include v
       if can_add(v, included):
           new_included = included | {v}
           # Determine new_available
           # Remove v
           # Remove u if {u, s, v} is AP for any s in included
           forbidden = set()
           for s in included:
               u1 = 2*v - s
               if 1 <= u1 <= N: forbidden.add(u1)
               u2 = 2*s - v
               if 1 <= u2 <= N: forbidden.add(u2)
           
           new_available = set(available)
           new_available.remove(v)
           new_available -= forbidden
           
           bnb(new_available, new_included, current_weight + weights[v])
           
       # Option 2: Exclude v
       new_available = set(available)
       new_available.remove(v)
       bnb(new_available, included, current_weight)
   