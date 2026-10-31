"""Traveling salesman constructive heuristic: Dynamic Lookahead with Repulsion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track visited nodes for repulsion
_visited_stack = []
_start_time = None
_DEADLINE = 145.0  # Safety margin before 150s kill

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)

def select_next_node(current_node, destination_node, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    global _visited_stack, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    # Base cases
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we are running out of time, fall back to simple NN
    if _get_time_remaining() < 5.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # Strategy: Lookahead Nearest Neighbor with Repulsion
    # 1. Calculate immediate distance from current node to each candidate
    # 2. For each candidate, estimate the cost of the *next* step (min distance from candidate to remaining unvisited)
    # 3. Add a repulsion term based on how recently a node was visited (or its neighbors? 
    #    Since we don't have coordinates, we can't easily do spatial repulsion. 
    #    However, we can use the "visited stack" to penalize nodes that are "close" in the sequence 
    #    if we had a way to measure that. Without coordinates, pure distance is the main metric.
    #    
    #    Alternative Repulsion: 
    #    In constructive heuristics for TSP, "repulsion" often refers to avoiding edges that would 
    #    close a small loop or return to the start too early. 
    #    Here, we can penalize the destination_node if it is the only one left? No, that's handled.
    #    
    #    Let's stick to a robust 2-step lookahead.
    #    Score(candidate) = d(current, candidate) + alpha * min_{v in unvisited\{candidate}} d(candidate, v)
    #    This is equivalent to choosing the node that leads to the cheapest 2-edge path.
    
    # Optimization: 
    # We need to be fast. 
    # Looping through unvisited_nodes in Python is slow for large N, but N=50 is small.
    # However, select_next_node is called N times. 
    # Total ops: Sum_{k=1 to N} (N-k) * (N-k) approx N^3/3. 
    # For N=50, 50^3/3 ~ 40k ops. Very fast.
    
    best_node = unvisited_nodes[0]
    best_score = np.inf
    
    # Get indices for vectorization if possible, but simple loop is safer for clarity and small N
    # Let's use a simple loop but try to vectorize the "min next distance" part if we can.
    
    # Actually, let's just do the standard loop. It's O(N^2) per call, N calls -> O(N^3) total.
    # 50^3 = 125,000 operations. In Python, this is < 0.1s.
    
    # We can slightly improve by using numpy for the distance lookups.
    # dists_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate i in unvisited:
    #   next_dists = distance_matrix[i, unvisited_nodes]
    #   mask out i itself? No, we want min distance to *other* unvisited.
    #   If only 1 other unvisited, min is that one.
    
    # Let's implement a hybrid:
    # If N is small, pure Python loop with numpy scalar access might be slower than vectorized.
    # But N=50 is small. Let's try to vectorize the scoring.
    
    # 1. Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each candidate, we need the min distance to other unvisited nodes.
    # This is tricky to vectorize directly without a matrix of all pairwise distances among unvisited.
    # Let's extract the submatrix of unvisited nodes.
    # This is O(N^2) memory and copy per step. 50x50 is small.
    
    # Extract submatrix
    # unvisited_nodes is an array of indices.
    # sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # This creates a (k x k) matrix.
    
    # For each row i in sub_dm, we want the min value, excluding the diagonal (self).
    # If k=1, min is 0 (or inf, doesn't matter as n_unvisited==1 is handled).
    # If k>1, min(sub_dm[i, :]) but sub_dm[i, i] is 0. We need min of off-diagonals.
    
    # Trick: Set diagonal to inf, then take min.
    k = n_unvisited
    if k > 1:
        sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Create a copy to modify diagonal
        # We don't want to allocate a new array every time if possible, but k is small.
        np.fill_diagonal(sub_dm, np.inf)
        # min_next = np.min(sub_dm, axis=1)
        
        # Score = d_curr + alpha * min_next
        # What alpha? 
        # If alpha=1, it's pure 2-step lookahead.
        # If alpha=0, it's pure NN.
        # Let's use alpha=1.0.
        
        min_next = np.min(sub_dm, axis=1)
        scores = d_curr + min_next
        
        # Find argmin
        best_idx = np.argmin(scores)
        best_node = unvisited_nodes[best_idx]
        
    else:
        best_node = unvisited_nodes[0]
        
    # Update visited stack
    _visited_stack.append(best_node)
    
    return int(best_node)
# EVOLVE-BLOCK-END
