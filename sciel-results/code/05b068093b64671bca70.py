import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
       n_unvisited = len(unvisited_nodes)
       
       if n_unvisited == 0:
           return destination_node
       
       if n_unvisited == 1:
           return int(unvisited_nodes[0])
       
       if n_unvisited == 2:
           c1 = unvisited_nodes[0]
           c2 = unvisited_nodes[1]
           cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
           cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
           return int(c1) if cost1 <= cost2 else int(c2)
       
       # Total cities is 50
       N = 50
       # Bias increases as we get closer to the end
       # At start, bias is low (NN). At end, bias is high (close to destination).
       # Let's use a stronger bias.
       # If n_unvisited is small, we should definitely prioritize closing the loop.
       # If n_unvisited is large, we should prioritize local proximity.
       
       # Let's try a fixed bias first to see if it's better than NN.
       # If NN is 1.0, and 2-step was 0.9, maybe 1-step with bias is better?
       
       # Let's use a bias that is 0 at start and 1 at end.
       # But maybe 1 is too strong?
       # Let's try lambda = 0.5 * (1 - n_unvisited / N)
       
       lam = 0.5 * (1.0 - n_unvisited / 50.0)
       
       best_cost = float('inf')
       best_next = unvisited_nodes[0]
       
       for c in unvisited_nodes:
           d_curr = distance_matrix[current_node, c]
           d_dest = distance_matrix[c, destination_node]
           cost = d_curr + lam * d_dest
           if cost < best_cost:
               best_cost = cost
               best_next = c
               
       return int(best_next)
