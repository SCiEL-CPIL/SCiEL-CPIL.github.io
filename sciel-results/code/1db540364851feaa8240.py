"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    # We need the indices of neighbors for each vertex to update gains efficiently
    adj = np.array(adjacency, dtype=bool)
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    
    # Initial random assignment
    labels = np.random.randint(0, 2, n)
    
    # Calculate initial cut size
    # Cut size is sum of edges between 0 and 1
    # We can compute this efficiently:
    # For each edge (u, v), if labels[u] != labels[v], it contributes 1.
    # Since the matrix is symmetric, we can just sum over all i,j where i<j or just use matrix ops.
    # Efficient way: 
    # Let S0 be set of 0s, S1 be set of 1s.
    # Cut = sum_{u in S0, v in S1} A[u,v]
    # This is equivalent to 0.5 * (sum_{u in S0} deg(u) - sum_{u in S0} sum_{v in S0} A[u,v]) ... no, simpler:
    # Cut = sum_{u} sum_{v > u} A[u,v] * (labels[u] != labels[v])
    
    # Even simpler for initialization:
    # cut = 0
    # for u in range(n):
    #    for v in neighbors[u]:
    #       if v > u and labels[u] != labels[v]:
    #           cut += 1
    
    # Vectorized initialization of cut size:
    # labels is 0/1. 
    # diff = labels[:, None] ^ labels[None, :] gives 1 where they differ.
    # cut = np.sum(adj & diff) / 2 # because each edge is counted twice (u,v) and (v,u)
    # But adj is bool, so adj & diff is bool.
    diff = labels[:, None] ^ labels[None, :]
    current_cut = np.sum(adj & diff) // 2
    
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Simulated Annealing Parameters
    # Heuristic for T0: average gain per flip is roughly related to degree.
    # Let's start with a high temperature to allow many moves.
    T0 = 1.0 * n
    T_end = 1e-5
    # Number of steps: We have ~120s. 
    # Each step involves O(deg(v)) work. Average deg is ~10-20 for random 100 node graphs? 
    # Actually, "100-node graphs" usually implies some sparsity or specific generator.
    # Let's assume we can do ~100,000 to 500,000 steps.
    # Let's aim for 200,000 steps.
    max_steps = 200000
    
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute gains for all vertices
    # gain[v] = change in cut if we flip v
    # If v flips, edges connecting v to same-side neighbors become crossing (gain +1 each)
    # Edges connecting v to opposite-side neighbors become non-crossing (gain -1 each)
    # Let same_side = sum_{u in N(v), labels[u] == labels[v]} 1
    # Let opp_side = sum_{u in N(v), labels[u] != labels[v]} 1
    # gain[v] = same_side - opp_side = same_side - (deg(v) - same_side) = 2*same_side - deg(v)
    
    gains = np.zeros(n, dtype=np.int32)
    for v in range(n):
        same_side = 0
        for u in neighbors[v]:
            if labels[u] == labels[v]:
                same_side += 1
        gains[v] = 2 * same_side - len(neighbors[v])
        
    # Cooling schedule: Geometric
    # T(t) = T0 * (T_end/T0)^(t/max_steps)
    # log(T) = log(T0) + t * (log(T_end) - log(T0)) / max_steps
    log_T0 = np.log(T0)
    log_T_end = np.log(T_end)
    log_delta = (log_T_end - log_T0) / max_steps
    
    current_T = T0
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Update temperature
        # We can update T every step or every few steps. Every step is fine.
        current_T = np.exp(log_T0 + step * log_delta)
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Calculate delta E (change in cut size)
        # If we flip v, the cut size changes by gains[v]
        delta = gains[v]
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # P = exp(delta / T)
            # If T is very small, exp(negative_large) is 0
            if current_T < 1e-10:
                accept = False
            else:
                # Avoid overflow/underflow issues
                ratio = np.exp(delta / current_T)
                accept = (np.random.rand() < ratio)
        
        if accept:
            # Flip v
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
            # Update gains for neighbors of v
            # When v flips, its side changes.
            # For any neighbor u of v:
            # The relationship between u and v changes from "same" to "opp" or vice versa.
            # This affects the gain of u.
            # Specifically, gain[u] = 2 * same_side(u) - deg(u)
            # If v was same_side(u), it becomes opp_side(u). same_side(u) decreases by 1.
            # So gain[u] decreases by 2.
            # If v was opp_side(u), it becomes same_side(u). same_side(u) increases by 1.
            # So gain[u] increases by 2.
            
            # Before flip:
            # labels[u] vs labels[v]
            # We just flipped v.
            # So we need to check the *new* state or *old* state?
            # Let's derive:
            # Old state: labels_old[v].
            # New state: labels_new[v] = 1 - labels_old[v].
            # For neighbor u:
            # If labels_old[u] == labels_old[v], then after flip labels_new[u] != labels_new[v].
            # So u lost a same-side neighbor. gain[u] -= 2.
            # If labels_old[u] != labels_old[v], then after flip labels_new[u] == labels_new[v].
            # So u gained a same-side neighbor. gain[u] += 2.
            
            # We have labels array which is now *updated* for v.
            # So for neighbor u, if labels[u] == labels[v] (new state), they are same.
            # This means they were opposite in old state. So gain[u] should increase by 2.
            # If labels[u] != labels[v] (new state), they are opposite.
            # This means they were same in old state. So gain[u] should decrease by 2.
            
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    # Now same, were opposite -> gain increases
                    gains[u] += 2
                else:
                    # Now opposite, were same -> gain decreases
                    gains[u] -= 2
            
            # The gain of v itself also changes!
            # gain[v] was delta.
            # After flip, the "same side" count for v becomes the "opposite side" count from before.
            # Let S_old = same_side_old, O_old = opp_side_old.
            # gain_old = 2*S_old - (S_old + O_old) = S_old - O_old = delta.
            # New same side S_new = O_old.
            # New opp side O_new = S_old.
            # gain_new = 2*S_new - (S_new + O_new) = 2*O_old - (O_old + S_old) = O_old - S_old = -delta.
            # So gains[v] = -gains[v]
            gains[v] = -gains[v]

    return best_labels
# EVOLVE-BLOCK-END
