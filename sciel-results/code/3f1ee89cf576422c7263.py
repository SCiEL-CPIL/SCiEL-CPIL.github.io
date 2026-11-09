import numpy as np
import time
import itertools

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s hard kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction using a 2-step lookahead heuristic.
    
    For small sets of unvisited nodes, uses exact permutation search.
    For larger sets, uses a vectorized 2-step greedy lookahead to minimize the path cost
    from the current node through the chosen node, its best successor, and the destination.
    """
    # Time guard - if we're running out of time, fall back to greedy nearest neighbor
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Exact search for small N (up to 6 nodes, 720 permutations)
    if n_unvisited <= 6:
        best_cost = float('inf')
        best_first = -1
        # Generate permutations of unvisited nodes
        for p in itertools.permutations(unvisited_nodes):
            # Calculate path cost: current -> p[0] -> ... -> p[-1] -> destination
            c = distance_matrix[current_node, p[0]]
            for i in range(n_unvisited - 1):
                c += distance_matrix[p[i], p[i+1]]
            c += distance_matrix[p[-1], destination_node]
            
            if c < best_cost:
                best_cost = c
                best_first = p[0]
        return int(best_first)
    
    # Vectorized 2-Step Lookahead for larger N
    U = unvisited_nodes
    
    # Precompute distance slices
    D_cU = distance_matrix[current_node, U]       # (N,)
    D_UU = distance_matrix[np.ix_(U, U)]           # (N, N)
    D_Ud = distance_matrix[U, destination_node]    # (N,)
    
    # Step 1: Calculate cost from each node k to destination via its best successor m
    # Cost_k = min_{m != k} (d(k, m) + d(m, dest))
    # We create a matrix where entry (k, m) is d(k, m) + d(m, dest)
    Temp1 = D_UU + D_Ud[np.newaxis, :]
    # Mask diagonal (k cannot be its own successor in this lookahead step)
    np.fill_diagonal(Temp1, np.inf)
    Cost_k = np.min(Temp1, axis=1)
    
    # Step 2: Calculate cost from each node j to destination via its best successor k
    # Cost_j = min_{k != j} (d(j, k) + Cost_k[k])
    Temp2 = D_UU + Cost_k[np.newaxis, :]
    # Mask diagonal
    np.fill_diagonal(Temp2, np.inf)
    Cost_j = np.min(Temp2, axis=1)
    
    # Total score for choosing j as the next node
    # Score(j) = d(current, j) + Cost_j
    Scores = D_cU + Cost_j
    
    best_idx = np.argmin(Scores)
    return int(U[best_idx])
