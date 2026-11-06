"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: total wall-clock budget for all calls combined.
# We track elapsed time across calls to ensure we finish well before the 120s/150s limit.
_DEADLINE = 110.0  # safety margin
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Picks the next city in a TSP tour.
    
    Strategy:
    - If there are few unvisited nodes (<= 12), use Held-Karp DP for exact optimal suffix.
    - Otherwise, use a lookahead heuristic: evaluate each candidate by the cost of going to it
      plus the estimated cost of completing the tour from there (using nearest neighbor on remaining).
      This is a 2-step lookahead that is fast and better than pure nearest neighbor.
    
    We also use a time guard: if we're running low on time, fall back to nearest neighbor.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, use simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use exact Held-Karp
    if n_unvisited <= 12:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger sets, use lookahead heuristic
    return _lookahead_select(current_node, destination_node, unvisited_nodes, distance_matrix)


def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP to find the optimal next node.
    
    We want to choose the next node 'j' from unvisited_nodes such that:
    dist(current, j) + optimal_tour(j, unvisited_nodes \\ {j}, destination_node)
    is minimized.
    
    For n<=12, Held-Karp is 2^12 * 12^2 ≈ 600k operations, which is fast.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited nodes to indices 0..n-1
    node_list = unvisited_nodes.astype(int)
    
    # DP over subsets of unvisited nodes, ending at a specific node in the subset
    # dp[mask][i] = min cost to start at node_list[i], visit all nodes in mask, and end at node_list[j]
    # Actually, we want: start at current, go to some j, then visit remaining, then go to destination.
    
    # Let's define: dp[mask][j] = min cost to start at j, visit all nodes in mask (which includes j),
    # and end at some node... no, standard Held-Karp:
    # dp[mask][j] = min over i in mask\{j} of dp[mask\{j}][i] + dist(node_list[i], node_list[j])
    # Base: dp[1<<j][j] = 0
    
    # But we need to account for the start at current_node and end at destination_node.
    # Total cost = dist(current, first) + [cost to visit all unvisited starting from first] + dist(last, destination)
    
    # So for each possible first node j, we need:
    # dist(current, j) + min over all permutations of remaining nodes of [path cost from j through remaining] + dist(last, destination)
    
    # Let's compute dp[mask][j] = min cost to start at j, visit all nodes in mask (mask includes j),
    # and end at j? No, that's not right.
    
    # Standard Held-Karp: dp[mask][j] = min cost to start at a fixed start node, visit all nodes in mask, and end at j.
    # Here, the "start" is current_node, but we don't visit current_node in the DP.
    
    # Let me redefine: Let the unvisited nodes be indexed 0..n-1.
    # dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j),
    # and end at node_list[k] for some k... 
    
    # Actually, let's use: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask, and end at node_list[j]? No.
    
    # Let's use the standard formulation:
    # dp[mask][j] = minimum cost to start at node_list[j], visit all nodes in mask (which must include j),
    # and end at some node. But we need to track the ending node.
    
    # Correct formulation:
    # dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j), and end at node_list[j]? 
    # No, that's TSP cycle.
    
    # For path: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask, and end at node_list[i] for some i in mask.
    # But we need to know where we end to add the distance to destination.
    
    # Let's use: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j), and end at node_list[j].
    # That's not right either.
    
    # Standard Held-Karp for TSP path:
    # dp[mask][j] = min over i in mask, i != j: dp[mask ^ (1<<j)][i] + dist(i, j)
    # where mask includes j, and we're ending at j.
    # Base: dp[1<<j][j] = 0
    # Then the total cost for a path starting at start_node, visiting all nodes in unvisited, ending at j, then going to dest:
    # dist(start_node, j) + dp[all_mask][j] ... no, dp[all_mask][j] is the cost from j through all others.
    
    # I think the cleanest is:
    # Let f(mask, j) = min cost to start at j, visit all nodes in mask (mask includes j), and end at some node k in mask.
    # But we need to track k.
    
    # Alternative: g(mask, j) = min cost to start at j, visit all nodes in mask, and end at j? No.
    
    # Let's just do: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j), and end at node_list[j].
    # This is a cycle, not what we want.
    
    # OK, let me use the standard approach:
    # dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j), and end at node_list[i] for the LAST node i.
    # But we don't know i until we finish.
    
    # Actually, the standard Held-Karp computes:
    # dp[mask][j] = min cost to start at a FIXED start node s, visit all nodes in mask, and end at j.
    # Here, s = current_node, but current_node is not in unvisited.
    
    # So: dp[mask][j] = min cost to start at current_node, visit all nodes in mask, and end at node_list[j].
    # Base: dp[1<<j][j] = dist(current_node, node_list[j])
    # Recurrence: dp[mask][j] = min over i in mask, i != j: dp[mask ^ (1<<j)][i] + dist(node_list[i], node_list[j])
    
    # Then the total tour cost for a given first node j is:
    # dp[all_mask][j] + dist(node_list[j], destination_node)
    # Wait, no. dp[all_mask][j] already includes the cost from current_node through all nodes ending at j.
    # So total = dp[all_mask][j] + dist(node_list[j], destination_node)
    
    # We want to minimize this over j. The NEXT node is the first node in the optimal path.
    # To find the first node, we can look at dp[1<<j][j] = dist(current, j), but that doesn't account for the rest.
    
    # Actually, to find the optimal NEXT node, we need to find which j minimizes:
    # dist(current, j) + [min cost to visit all other unvisited nodes starting from j and ending at some k] + dist(k, destination)
    
    # Let h(mask, j) = min cost to start at j, visit all nodes in mask (mask includes j), and end at some node.
    # But we need to track the ending node to add dist to destination.
    
    # Let's compute: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j), and end at node_list[i] where i is the last node.
    # But we don't track i.
    
    # I'll use: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask, and end at node_list[j]. No.
    
    # Let me just compute the full Held-Karp properly:
    # dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask (mask includes j), and end at node_list[j] is wrong.
    
    # Correct: dp[mask][j] = min cost to start at node_list[j], visit all nodes in mask, and end at node_list[k] for some k.
    # But we need to know k.
    
    # I'll use the standard formulation where dp[mask][j] ends at j:
    # dp[mask][j] = min over i in mask, i != j: dp[mask ^ (1<<j)][i] + dist(node_list[i], node_list[j])
    # Base: dp[1<<j][j] = 0
    # Then the cost to start at current, go to j, visit all in mask, end at j, then go to dest:
    # dist(current, j) + dp[all_mask][j] + dist(node_list[j], dest) ... but dp[all_mask][j] is the cost from j through all others ending at j? No.
    
    # I think I'm overcomplicating. Let me just compute the best next node by trying each candidate and using a fast estimate.
    
    # For n <= 12, we can afford to compute the exact optimal suffix for each candidate.
    # For each candidate j, compute the optimal path from j through all other unvisited nodes to destination.
    
    best_cost = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        j = node_list[j_idx]
        remaining = np.delete(node_list, j_idx)
        if len(remaining) == 0:
            cost = distance_matrix[current_node, j] + distance_matrix[j, destination_node]
        else:
            # Compute optimal path from j through all remaining to destination
            # Use Held-Karp on remaining
            m = len(remaining)
            # dp[mask][k] = min cost to start at remaining[k], visit all in mask, end at remaining[l]
            # Actually, let's use: dp[mask][k] = min cost to start at j, visit all in mask, end at remaining[k]
            # Base: dp[1<<k][k] = dist(j, remaining[k])
            # Recurrence: dp[mask][k] = min over l in mask, l != k: dp[mask ^ (1<<k)][l] + dist(remaining[l], remaining[k])
            
            if m == 1:
                cost = distance_matrix[current_node, j] + distance_matrix[j, remaining[0]] + distance_matrix[remaining[0], destination_node]
            else:
                # Held-Karp for path from j through remaining to destination
                dp = np.full((1 << m, m), np.inf)
                for k in range(m):
                    dp[1 << k, k] = distance_matrix[j, remaining[k]]
                
                for mask in range(1, 1 << m):
                    for k in range(m):
                        if not (mask & (1 << k)):
                            continue
                        if mask == (1 << k):
                            continue
                        prev_mask = mask ^ (1 << k)
                        for l in range(m):
                            if prev_mask & (1 << l):
                                new_cost = dp[prev_mask, l] + distance_matrix[remaining[l], remaining[k]]
                                if new_cost < dp[mask, k]:
                                    dp[mask, k] = new_cost
                
                full_mask = (1 << m) - 1
                min_suffix = np.min(dp[full_mask, :])
                cost = distance_matrix[current_node, j] + min_suffix + min(
                    distance_matrix[remaining[k], destination_node] for k in range(m)
                    if dp[full_mask, k] == min_suffix
                )
        
        if cost < best_cost:
            best_cost = cost
            best_next = j
    
    return int(best_next)


def _lookahead_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Lookahead heuristic: for each candidate, estimate the cost of going to it
    plus a fast estimate of the remaining tour cost.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # For each candidate, compute:
    # dist(current, candidate) + estimate_remaining(candidate, unvisited \ {candidate}, destination)
    
    # Fast estimate: nearest neighbor tour cost from candidate through remaining to destination
    # But that's O(n^2) per candidate, so O(n^3) total. For n=50, that's 125k, which is fine.
    
    best_cost = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        j = unvisited_nodes[j_idx]
        remaining = np.delete(unvisited_nodes, j_idx)
        
        # Estimate: nearest neighbor path from j through remaining to destination
        current_pos = j
        est_cost = distance_matrix[current_node, j]
        
        for _ in range(len(remaining)):
            if len(remaining) == 0:
                break
            next_node = remaining[np.argmin(distance_matrix[current_pos, remaining])]
            est_cost += distance_matrix[current_pos, next_node]
            current_pos = next_node
            remaining = np.delete(remaining, np.where(remaining == next_node)[0][0])
        
        if len(remaining) > 0:
            est_cost += distance_matrix[current_pos, destination_node]
        
        if est_cost < best_cost:
            best_cost = est_cost
            best_next = j
    
    return int(best_next)
# EVOLVE-BLOCK-END
