"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard to ensure we finish well before the 150s limit
    if _get_time_remaining() < 1.0:
        # Fallback to simple NN if time is critical
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Clarke-Wright Savings Heuristic Logic
    # We want to minimize the total tour length.
    # In a star topology centered at destination_node, the cost is sum(2 * d(dest, node)).
    # When we merge two nodes u and v into a path segment u-v, we "save" distance:
    # Savings(u, v) = d(dest, u) + d(dest, v) - d(u, v)
    # 
    # However, we are constructing a path starting from current_node.
    # The standard savings heuristic builds a tree. Here we approximate by choosing
    # the next node that maximizes the savings relative to the "ideal" star, 
    # but constrained by the current path structure.
    
    # A simpler and effective adaptation for path construction:
    # Choose the next node 'v' from unvisited that minimizes:
    # d(current, v) + d(v, destination) 
    # BUT, we also want to avoid closing small loops or getting stuck.
    
    # Let's use a "Savings" inspired score.
    # If we pick v, we add edge (current, v). 
    # The "potential" savings of connecting v to destination later vs connecting current to destination now?
    
    # Actually, a robust heuristic for TSP path construction that beats NN is 
    # "Savings" or "Cheapest Insertion" approximations.
    # Since we must pick one node, let's look at the "regret" or "opportunity cost".
    
    # Let's try a weighted combination of:
    # 1. Distance from current (immediate cost)
    # 2. Distance from destination (future cost if we go there)
    # 3. A penalty for nodes that are "expensive" to connect to their neighbors (high local density?)
    
    # Let's stick to a pure Savings-based ranking among candidates.
    # For each candidate v in unvisited:
    # Calculate the "savings" if we were to connect current->v->...->dest 
    # compared to current->...->v->...->dest? No, that's complex.
    
    # Let's use the standard Clarke-Wright savings value for the pair (current, v) 
    # assuming the "hub" is the destination.
    # Savings(current, v) = d(current, dest) + d(v, dest) - d(current, v)
    # We want to pick the v that maximizes this savings?
    # Wait, if Savings is high, it means d(current, v) is small relative to the sum of distances to dest.
    # This is equivalent to minimizing d(current, v) - d(current, dest) - d(v, dest).
    # Since d(current, dest) is constant for all v, this is minimizing d(current, v) - d(v, dest).
    
    # Let's test this metric: Score(v) = d(current, v) - d(v, dest)
    # If d(v, dest) is large, the score is smaller (better).
    # This encourages moving towards nodes that are far from the destination, 
    # effectively pushing the "frontier" of the tour away from the destination 
    # to leave the direct connection to the destination for the very end, 
    # which is often longer but allows the middle of the tour to be more compact?
    
    # Actually, the standard Savings heuristic merges pairs (u, v) if 
    # d(u, hub) + d(v, hub) - d(u, v) is large.
    # Here, we are extending a path.
    # If we pick v, we are effectively "merging" the segment ending at current with the node v.
    # The cost added is d(current, v).
    # The "baseline" cost if we didn't merge them in this specific way is hard to define in a single step.
    
    # Let's try a different perspective: **Regret-2**.
    # For each candidate v, calculate the cost of choosing it: c1 = d(current, v).
    # Calculate the cost of the second best option: c2 = min(d(current, u) for u != v).
    # Regret = c1 - c2.
    # This is usually for assignment.
    
    # Let's go back to the **Savings** idea but applied to the *entire* remaining set.
    # A very strong heuristic for TSP is the **Nearest Neighbor with Lookahead (2-step)**.
    # But the prompt asks for a DIFFERENT strategy.
    
    # How about **Farthest Point** or **Most Distant** from current?
    # No, that makes long edges.
    
    # Let's try **Savings-based Selection**:
    # Score(v) = d(current, v) + d(v, destination)
    # This is the "insertion" cost if v were the last node.
    # Minimizing this tends to pick nodes that are close to current OR close to destination.
    # This is similar to the "Cheapest Insertion" heuristic.
    
    # Let's combine: 
    # Primary: d(current, v)
    # Secondary: d(v, destination)
    # We want to minimize d(current, v) + lambda * d(v, destination)
    # If lambda is small, it's NN. If lambda is large, it's "closest to destination".
    # For TSP, a good path usually keeps the "open ends" (current and destination) close to each other eventually.
    
    # Let's try **Maximizing Savings**:
    # Savings(v) = d(current, destination) + d(v, destination) - d(current, v)
    # We want to MAXIMIZE this.
    # Maximize (const + d(v, dest) - d(current, v))
    # Equivalent to Minimize (d(current, v) - d(v, dest))
    
    # Let's implement this specific metric. It is fundamentally different from pure NN 
    # because it penalizes nodes that are far from current but close to destination less heavily, 
    # and encourages nodes that are far from destination (to save the long "return" leg for later? 
    # No, if v is far from dest, d(v,dest) is large, so -d(v,dest) is very negative, 
    # making the score low (bad) if d(current,v) is small?
    
    # Let's re-evaluate:
    # Savings = d(cur, dest) + d(v, dest) - d(cur, v)
    # If we pick v, we "use up" the opportunity to save this distance.
    # We should pick the v with the HIGHEST savings.
    # High savings means d(cur, v) is small relative to d(cur, dest) + d(v, dest).
    # This is basically: d(cur, v) - d(v, dest) is small.
    
    # Let's test this logic on a simple line: 0-1-2-3. Dest=3. Current=0.
    # Unvisited: 1, 2.
    # d(0,3)=3, d(1,3)=2, d(2,3)=1.
    # d(0,1)=1, d(0,2)=2.
    # For v=1: Score = 1 - 2 = -1. Savings = 3+2-1=4.
    # For v=2: Score = 2 - 1 = 1. Savings = 3+1-2=2.
    # We pick v=1 (Highest Savings / Lowest Score).
    # This results in 0-1-2-3. Correct.
    
    # What if points are scattered?
    # This heuristic effectively tries to keep the path "tight" relative to the destination.
    
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    distances_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Score = d(current, v) - d(destination, v)
    # We want to MINIMIZE this score (which is equivalent to MAXIMIZING savings)
    scores = distances_from_current - distances_from_dest
    
    # Add a small tie-breaker to prefer closer nodes if scores are identical (numerical stability)
    # Or just take argmin.
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
