import numpy as np
import time

_EVAL_START = time.time()
DEADLINE = _EVAL_START + 140

def _time_remaining():
    return DEADLINE - time.time()

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
       if _time_remaining() < 1.0:
           if len(unvisited_nodes) == 0:
               return 0
           feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
           if len(feasible) == 0:
               return 0
           return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

       if len(unvisited_nodes) == 0:
           return 0

       unvisited_arr = np.array(unvisited_nodes)
       feasible_mask = demands[unvisited_arr] <= rest_capacity
       feasible = unvisited_arr[feasible_mask]

       if len(feasible) == 0:
           return 0

       if len(feasible) == 1:
           return int(feasible[0])

       # Distances
       d_cur = distance_matrix[current_node, feasible]
       d_dep = distance_matrix[feasible, depot]
       
       # Pure Savings Heuristic
       # Maximize d(depot, j) - d(current, j)
       # Equivalent to minimizing d(current, j) - d(depot, j)
       scores = d_cur - d_dep
       
       best_idx = np.argmin(scores)
       return int(feasible[best_idx])
