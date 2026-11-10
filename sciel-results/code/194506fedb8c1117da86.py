import random
import time

def construct():
       d = 7
       n_vertices = 2 ** d
       all_edges = []
       edge_to_idx = {}
       
       for x in range(n_vertices):
           for i in range(d):
               y = x ^ (1 << i)
               if x < y:
                   e = (x, y)
                   edge_to_idx[e] = len(all_edges)
                   all_edges.append(e)
                   
       num_edges = len(all_edges)
       
       # Generate squares
       squares = []
       edge_to_squares = [[] for _ in range(num_edges)]
       
       for i in range(d):
           for j in range(i + 1, d):
               for x in range(n_vertices):
                   if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                       v0 = x
                       v1 = x ^ (1 << i)
                       v2 = x ^ (1 << j)
                       v3 = x ^ (1 << i) ^ (1 << j)
                       
                       e0 = (min(v0, v1), max(v0, v1))
                       e1 = (min(v1, v3), max(v1, v3))
                       e2 = (min(v3, v2), max(v3, v2))
                       e3 = (min(v2, v0), max(v2, v0))
                       
                       idxs = [edge_to_idx[e] for e in [e0, e1, e2, e3]]
                       sq_idx = len(squares)
                       squares.append(idxs)
                       for idx in idxs:
                           edge_to_squares[idx].append(sq_idx)
                           
       num_squares = len(squares)
       
       best_size = 0
       best_set = set()
       
       start_time = time.time()
       time_limit = 110 # Leave margin for overhead
       
       # Run multiple random restarts
       for run in range(1000): # Many fast runs
           if time.time() - start_time > time_limit:
               break
           
           # State
           # removed: set of edge indices removed
           removed = set()
           # sq_count[s]: number of edges present in square s (initially 4)
           sq_count = [4] * num_squares
           # edge_threat[e]: number of C4s (squares with count 4) containing e
           edge_threat = [0] * num_edges
           
           # Initialize threats
           for s in range(num_squares):
               for e in squares[s]:
                   edge_threat[e] += 1
                   
           # 1. Greedy Removal
           # We maintain a list of "active" C4s? Or just scan?
           # Scanning 392 squares is fast.
           # But we need to know which squares are C4s.
           # A square is a C4 if sq_count[s] == 4.
           
           while True:
               # Find C4s
               c4s = [s for s in range(num_squares) if sq_count[s] == 4]
               if not c4s:
                   break
               
               # For each C4, find the edge to remove
               # Heuristic: Remove edge with max edge_threat
               # Tie-break: Random
               
               # We can pick a random C4, then pick the best edge in it?
               # Or pick the best edge among ALL C4s?
               # "Best edge among all C4s" is stronger.
               
               max_threat = -1
               candidates = []
               
               # Iterate through all C4s
               for s in c4s:
                   for e in squares[s]:
                       if e in removed:
                           continue
                       # e is in a C4, so threat[e] >= 1
                       if edge_threat[e] > max_threat:
                           max_threat = edge_threat[e]
                           candidates = [e]
                       elif edge_threat[e] == max_threat:
                           candidates.append(e)
                           
               if not candidates:
                   break
                   
               e_to_remove = random.choice(candidates)
               removed.add(e_to_remove)
               
               # Update state
               for s in edge_to_squares[e_to_remove]:
                   if sq_count[s] == 4:
                       # It was a C4, now it's not
                       sq_count[s] -= 1
                       # Decrease threat of other edges in s
                       for e2 in squares[s]:
                           if e2 != e_to_remove:
                               edge_threat[e2] -= 1
                           # Note: edge_threat[e_to_remove] is not updated, but e_to_remove is removed so it doesn't matter
       
           # Current set is valid
           current_size = num_edges - len(removed)
           
           # 2. Recovery Phase
           # Try to add back removed edges
           # Iterate over removed edges in random order
           removed_list = list(removed)
           random.shuffle(removed_list)
           
           for e in removed_list:
               # Check if e can be added
               can_add = True
               for s in edge_to_squares[e]:
                   if sq_count[s] == 3:
                       # Adding e would make it 4 -> C4
                       can_add = False
                       break
                       
               if can_add:
                   removed.discard(e)
                   # Update state
                   for s in edge_to_squares[e]:
                       sq_count[s] += 1
                       # If sq_count becomes 4, it's a C4 again!
                       # But we just checked it was 3, so it becomes 4.
                       # This creates a C4.
                       # So we must update threats?
                       # Wait, if we add e and sq_count becomes 4, we have a C4.
                       # This violates the constraint.
                       # So we CANNOT add e if it creates a C4.
                       # My check `if sq_count[s] == 3` implies adding e makes it 4.
                       # So `can_add` should be False if ANY square has count 3.
                       # So the logic above is correct: if can_add is True, no square has count 3.
                       # So adding e will not create any C4.
                       # So we just increment sq_count.
                       # And we don't need to update threats because no new C4 is formed.
       
           # Check if we improved
           final_size = num_edges - len(removed)
           if final_size > best_size:
               best_size = final_size
               best_set = set(range(num_edges)) - removed
               
       # Return edges
       return [[all_edges[e][0], all_edges[e][1]] for e in best_set]
