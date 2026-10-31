"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_DEADLINE_BUFFER = 10.0  # seconds safety margin before 150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_BUFFER
    
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # If only two left, pick the one that creates the shorter edge from current
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        # If we pick 0: path is ...->current->0->1->dest
        # Cost contribution: dist(current,0) + dist(0,1) + dist(1,dest)
        cost_pick_0 = d0 + d_cross + d_dest_1
        
        # If we pick 1: path is ...->current->1->0->dest
        # Cost contribution: dist(current,1) + dist(1,0) + dist(0,dest)
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])

    # --- Core Strategy: Global 2-Opt Improvement ---
    # The previous strategies relied on greedy construction heuristics (nearest neighbor, hull peeling).
    # These are fast but get stuck in local minima.
    # We have a generous 150s budget. We will:
    # 1. Construct an initial tour using Nearest Neighbor (fast).
    # 2. Iteratively apply 2-Opt swaps to improve the tour until no improvement is found or time is up.
    # 3. Return the next node in the optimized tour.
    
    # 1. Build initial tour using Nearest Neighbor
    # Note: 'current_node' is the start of the remaining path. 'destination_node' is the end (where we return to).
    # The unvisited nodes must be visited in some order between current and destination.
    
    # We treat this as a path problem: Start at current, visit all unvisited, end at destination.
    # However, TSP is a cycle. The "current_node" is the last visited node in the partial tour.
    # The "destination_node" is the node we must return to (the start of the full tour).
    
    # Let's construct a list of nodes: [current] + unvisited_order + [destination]
    # We want to minimize the sum of distances along this path.
    
    # Initial Order: Nearest Neighbor from current
    remaining = list(unvisited_nodes)
    order = []
    curr = current_node
    
    while remaining:
        # Find nearest in remaining
        # Optimization: Use numpy for speed
        dists = distance_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        order.append(next_node)
        remaining.pop(min_idx)
        curr = next_node
    
    # Initial tour segment: current -> order[0] -> ... -> order[-1] -> destination
    # Full node sequence for this segment: [current] + order + [destination]
    # But we only need to return the *next* node, which is order[0] after optimization.
    
    # To apply 2-Opt, we need the full sequence of nodes involved in this segment.
    # Let seq be the list of nodes in the order we visit them, including start and end.
    seq = np.array([current_node] + order + [destination_node])
    
    # 2. 2-Opt Improvement
    # A 2-Opt move removes two edges (i, i+1) and (j, j+1) and replaces them with (i, j) and (i+1, j+1).
    # This reverses the segment between i+1 and j.
    # We only care about the internal nodes of `seq` (indices 1 to len(seq)-2).
    # The start (current) and end (destination) are fixed anchors for this subproblem?
    # Actually, in the full TSP tour, current and destination are part of the cycle.
    # But here, we are constructing the tour step-by-step.
    # The evaluator calls this function 50 times.
    # Each call, we re-optimize the *entire remaining path* from current to destination.
    # This is effectively solving a small TSP path problem (from current to dest) at each step.
    # Since n_unvisited starts at 49 and decreases, the cost of 2-Opt is O(n^2) per step.
    # Total cost: Sum_{k=1}^{50} O(k^2) ~ O(50^3) = 125,000 operations per 2-opt pass.
    # With multiple passes, this is very fast (< 1s). We have 150s.
    
    # We can afford multiple 2-Opt passes.
    
    # Precompute distance matrix for the sequence to speed up lookups?
    # No, distance_matrix is already provided.
    
    # Helper to calculate cost of the path defined by seq
    def path_cost(s):
        return np.sum(distance_matrix[s[:-1], s[1:]])
    
    current_cost = path_cost(seq)
    
    # 2-Opt Loop
    # We only swap internal edges.
    # Edges are (0,1), (1,2), ..., (n-2, n-1).
    # Nodes are 0..n-1.
    # We cannot change the position of node 0 (current) or node n-1 (destination).
    # So we consider i in [0, n-3] and j in [i+1, n-2].
    # If we swap, we reverse seq[i+1 : j+1].
    
    n_seq = len(seq)
    improved = True
    passes = 0
    max_passes = 100 # Safety limit, though usually converges faster
    
    while improved and passes < max_passes and time.time() < _DEADLINE:
        improved = False
        passes += 1
        for i in range(n_seq - 2):
            # Early exit if deadline is close
            if time.time() > _DEADLINE - 2.0:
                break
            for j in range(i + 1, n_seq - 1):
                # Calculate delta cost
                # Edges removed: (i, i+1) and (j, j+1)
                # Edges added: (i, j) and (i+1, j+1)
                # Note: If i == j-1, edges are (i, i+1) and (i+1, i+2). Added: (i, i+2) and (i+1, i+1)=0? No.
                # Standard 2-Opt:
                # Remove (u, v) and (x, y). Add (u, x) and (v, y).
                # Here u=seq[i], v=seq[i+1], x=seq[j], y=seq[j+1].
                # If j == i+1, we are swapping adjacent nodes? No, that's not 2-opt, that's just reversing 2 nodes.
                # Usually j > i.
                
                u = seq[i]
                v = seq[i+1]
                x = seq[j]
                y = seq[j+1]
                
                # If u==x or v==y etc, handle carefully, but indices are distinct in the loop.
                # Exception: if j == i+1, then v == x.
                # Remove (u, v) and (v, y). Add (u, v) and (v, v)? No.
                # If j == i+1:
                # Remove (u, v) and (v, y).
                # Add (u, v) and (v, y)? No, reversal of [v] is [v].
                # The segment [i+1 : j+1] is just [v]. Reversing it doesn't change the sequence.
                # So j must be > i+1 for a meaningful 2-opt that changes the order?
                # Actually, standard 2-opt allows j = i+1?
                # If j = i+1, edges removed: (i, i+1), (i+1, i+2).
                # Edges added: (i, i+1), (i+1, i+1) -> 0? No.
                # The formula is:
                # Cost change = d(u, x) + d(v, y) - d(u, v) - d(x, y)
                # If j = i+1: x = v, y = seq[i+2].
                # Change = d(u, v) + d(v, y) - d(u, v) - d(v, y) = 0.
                # So j > i+1 is required for non-zero change?
                # Wait, if we reverse a segment of length 1, nothing changes.
                # So we can start j from i+2.
                
                if j == i + 1:
                    continue
                
                d_ux = distance_matrix[u, x]
                d_vy = distance_matrix[v, y]
                d_uv = distance_matrix[u, v]
                d_xy = distance_matrix[x, y]
                
                delta = d_ux + d_vy - d_uv - d_xy
                
                if delta < -1e-10: # Improvement
                    # Reverse segment i+1 to j (inclusive)
                    seq[i+1:j+1] = seq[i+1:j+1][::-1]
                    current_cost += delta
                    improved = True
                    # After a swap, the structure changes. 
                    # We can restart the inner loop or just continue.
                    # Continuing is safer but might miss improvements created by the reversal.
                    # Restarting the outer loop is more thorough but slower.
                    # Given the small size (<=50), restarting is fine.
                    break
            if improved:
                # Restart outer loop to find new improvements
                break
            if time.time() > _DEADLINE - 2.0:
                break
        
    # 3. Return the next node
    # The optimized sequence is seq.
    # The next node to visit is seq[1].
    return int(seq[1])

# EVOLVE-BLOCK-END
