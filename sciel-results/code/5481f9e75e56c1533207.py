"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for caching
_cache = {}
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin before 120s limit


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _time_exceeded():
    return (time.time() - _start_time) > _TIME_LIMIT


def _held_karp_top_k(start, candidates, unvisited, dist, k):
    """
    Use Held-Karp DP on the top-k candidates to find the best next node.
    
    For each candidate c in candidates:
      - We go from start to c, then visit all remaining unvisited nodes (except c),
        then return to destination.
    Actually, the problem is: we're building a tour step by step.
    current_node -> next_node -> ... -> destination_node (close the tour)
    
    So if we pick candidate c as next:
      cost = dist[start, c] + (cost to go from c, visit all unvisited\\{c}, end at destination)
    
    The "cost to go from c, visit all unvisited\\{c}, end at destination" is a TSP path
    from c to destination visiting all nodes in unvisited\\{c}.
    
    This is still expensive. Let's use a simpler approach:
    For each candidate c, estimate the remaining tour cost using a greedy heuristic
    from c visiting all unvisited\\{c} and ending at destination.
    
    Actually, let's use a proper DP for small subsets. If |unvisited| is small enough,
    we can do exact DP. Otherwise, use approximation.
    """
    n_unvisited = len(unvisited)
    if n_unvisited <= 1:
        # Only one node left, go there and back to destination
        if n_unvisited == 1:
            c = unvisited[0]
            return dist[start, c] + dist[c, destination_node]
        else:
            return dist[start, destination_node]
    
    # If too many unvisited nodes, fall back to greedy scoring
    if n_unvisited > 15:
        # Use a greedy estimate for each candidate
        best_score = float('inf')
        best_c = candidates[0]
        for c in candidates:
            # Greedy: from c, visit all unvisited\\{c} greedily, then go to destination
            remaining = [u for u in unvisited if u != c]
            cost = dist[start, c]
            curr = c
            rem_set = set(remaining)
            while rem_set:
                # Find nearest unvisited
                min_d = float('inf')
                next_node = None
                for u in rem_set:
                    d = dist[curr, u]
                    if d < min_d:
                        min_d = d
                        next_node = u
                cost += min_d
                rem_set.remove(next_node)
                curr = next_node
            cost += dist[curr, destination_node]
            if cost < best_score:
                best_score = cost
                best_c = c
        return best_c
    
    # For smaller sets, use Held-Karp DP
    # Map unvisited nodes to indices
    # We want: min over c in candidates of [dist[start,c] + tsp_path(c, unvisited\{c}, destination)]
    
    # tsp_path(start_node, visit_set, end_node) = min cost path from start_node through all nodes in visit_set to end_node
    
    # This is a standard TSP path problem. Let's use DP.
    # State: (current_node, visited_subset)
    # But this is exponential in |unvisited|.
    
    # For |unvisited| <= 15, 2^15 = 32768 states, each with up to 15 transitions.
    # That's ~500k operations per candidate, times up to 12 candidates = 6M. Might be slow.
    
    # Let's limit to |unvisited| <= 12 for exact DP
    if n_unvisited > 12:
        # Fall back to greedy
        best_score = float('inf')
        best_c = candidates[0]
        for c in candidates:
            remaining = [u for u in unvisited if u != c]
            cost = dist[start, c]
            curr = c
            rem_set = set(remaining)
            while rem_set:
                min_d = float('inf')
                next_node = None
                for u in rem_set:
                    d = dist[curr, u]
                    if d < min_d:
                        min_d = d
                        next_node = u
                cost += min_d
                rem_set.remove(next_node)
                curr = next_node
            cost += dist[curr, destination_node]
            if cost < best_score:
                best_score = cost
                best_c = c
        return best_c
    
    # Exact DP for |unvisited| <= 12
    # We'll compute for each candidate c:
    #   dist[start, c] + min_path(c, unvisited\{c}, destination)
    
    # min_path(start, S, end) where S is a set of nodes to visit
    # DP over subsets of S
    
    # Let's create a unified DP:
    # For each candidate c, we need the cost of a path from c through all nodes in (unvisited \ {c}) to destination.
    
    # Instead of doing it separately for each c, let's do it once.
    # Define: f(node, mask) = min cost to start at node, visit all nodes in mask, end at destination.
    # mask is a bitmask over unvisited nodes.
    
    # But we need to start at different nodes (each candidate c).
    
    # Let's just do it for each candidate. With |unvisited| <= 12, and up to 12 candidates,
    # each DP has 2^(n-1) * n states. n-1 <= 11, so 2^11 * 11 = 22528 states per candidate.
    # 12 candidates * 22528 = 270k states. Each state has up to 11 transitions. ~3M ops. OK.
    
    # Map unvisited to indices 0..n-1
    unvisited_list = list(unvisited)
    node_to_idx = {node: i for i, node in enumerate(unvisited_list)}
    n = len(unvisited_list)
    
    # DP: dp[mask][i] = min cost to start at unvisited_list[i], visit all nodes in mask, end at destination
    # mask is a bitmask over the n nodes
    # We want: for each candidate c (which is at index idx_c in unvisited_list),
    #   cost = dist[start, c] + dp[full_mask_without_c][idx_c]
    # where full_mask_without_c = (1 << n) - 1 - (1 << idx_c)
    
    # Actually, dp[mask][i] means: starting at node i, we need to visit all nodes in mask (excluding i if i is in mask? No, let's define clearly)
    
    # Let's define: dp[mask][i] = min cost to start at node unvisited_list[i], visit all nodes in mask, and end at destination.
    # The set mask should include i (the starting node) or not? Let's say mask is the set of nodes still to visit, including the current position.
    # Actually, standard formulation:
    # dp[mask][i] = min cost to have visited the set of nodes indicated by mask, currently at node i, and need to end at destination.
    # But we want a path from start to destination through all nodes.
    
    # Let me redefine:
    # For a fixed starting node s (which is one of the candidates), we want the min cost path from s through all nodes in V = unvisited\{s} to destination.
    # This is a TSP path from s to destination visiting all nodes in V.
    
    # DP: dp[mask][j] = min cost to start at s, visit all nodes in mask (subset of V), currently at node j (j in mask or j=s)
    # This is getting complicated. Let me just use a simple approach for each candidate.
    
    best_score = float('inf')
    best_c = candidates[0]
    
    for c in candidates:
        idx_c = node_to_idx[c]
        # Nodes to visit: all unvisited except c
        visit_nodes = [unvisited_list[i] for i in range(n) if i != idx_c]
        m = len(visit_nodes)
        
        if m == 0:
            cost = dist[start, c] + dist[c, destination_node]
        else:
            # DP over subsets of visit_nodes
            # dp[mask][j] = min cost to start at c, visit all nodes in mask (subset of visit_nodes), currently at visit_nodes[j]
            # Then add dist[visit_nodes[j], destination]
            # Or if mask is empty, cost is dist[c, destination]
            
            # Initialize
            full_mask = (1 << m) - 1
            dp = [[float('inf')] * m for _ in range(1 << m)]
            
            # Base case: start at c, go to each visit node
            for j in range(m):
                dp[1 << j][j] = dist[c, visit_nodes[j]]
            
            # Fill DP
            for mask in range(1, 1 << m):
                for j in range(m):
                    if dp[mask][j] == float('inf'):
                        continue
                    if not (mask & (1 << j)):
                        continue
                    # Try to go to next node k not in mask
                    for k in range(m):
                        if mask & (1 << k):
                            continue
                        new_mask = mask | (1 << k)
                        new_cost = dp[mask][j] + dist[visit_nodes[j], visit_nodes[k]]
                        if new_cost < dp[new_mask][k]:
                            dp[new_mask][k] = new_cost
            
            # Find min over j of dp[full_mask][j] + dist[visit_nodes[j], destination]
            min_remaining = float('inf')
            for j in range(m):
                if dp[full_mask][j] < float('inf'):
                    total = dp[full_mask][j] + dist[visit_nodes[j], destination_node]
                    if total < min_remaining:
                        min_remaining = total
            
            cost = dist[start, c] + min_remaining
        
        if cost < best_score:
            best_score = cost
            best_c = c
    
    return best_c


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour construction.
    
    Uses a multi-step lookahead with exact Held-Karp DP on top candidates
    when the remaining tour is small enough.
    """
    if len(unvisited_nodes) == 0:
        return -1  # Should not happen
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    dist = distance_matrix
    current = current_node
    dest = destination_node
    unvisited = unvisited_nodes
    
    # Check time limit
    if _time_exceeded():
        # Fall back to nearest neighbor
        return int(unvisited[np.argmin(dist[current, unvisited])])
    
    # Score all unvisited nodes using a combination of:
    # 1. Immediate distance from current
    # 2. Distance to destination (to ensure we can close the tour)
    # 3. Average distance to remaining unvisited nodes (for connectivity)
    
    n_unvisited = len(unvisited)
    
    # Immediate distance
    imm_dist = dist[current, unvisited]
    
    # Distance to destination
    dest_dist = dist[unvisited, dest]
    
    # For each candidate, estimate the remaining tour cost
    # Use a simple heuristic: after going to candidate, greedily visit remaining
    
    # First, get top-k candidates by immediate distance
    k = min(12, n_unvisited)
    top_k_indices = np.argpartition(imm_dist, k)[:k]
    top_k_candidates = unvisited[top_k_indices]
    
    # Sort top-k by immediate distance
    top_k_sorted = top_k_candidates[np.argsort(imm_dist[top_k_indices])]
    
    # If few unvisited, use exact DP
    if n_unvisited <= 12:
        best_c = _held_karp_top_k(current, top_k_sorted, unvisited, dist, k)
        return int(best_c)
    
    # For larger instances, use a greedy estimate for top-k candidates
    best_score = float('inf')
    best_c = top_k_sorted[0]
    
    for c in top_k_sorted:
        # Estimate: dist[current, c] + greedy tour from c through remaining to dest
        remaining = [u for u in unvisited if u != c]
        cost = dist[current, c]
        curr = c
        rem_set = set(remaining)
        while rem_set:
            # Find nearest unvisited
            min_d = float('inf')
            next_node = None
            for u in rem_set:
                d = dist[curr, u]
                if d < min_d:
                    min_d = d
                    next_node = u
            cost += min_d
            rem_set.remove(next_node)
            curr = next_node
        cost += dist[curr, dest]
        
        if cost < best_score:
            best_score = cost
            best_c = c
    
    return int(best_c)
# EVOLVE-BLOCK-END
