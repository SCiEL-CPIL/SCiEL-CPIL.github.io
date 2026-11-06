# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE = 115.0  # Leave 5s margin for 120s limit

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _reset_state():
    global _start_time
    _start_time = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Reset if this looks like the first call of a new evaluation
    # We don't know for sure, but if elapsed is huge, reset
    if _start_time is not None and _get_elapsed() > 110:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Check deadline
    elapsed = _get_elapsed()
    if elapsed > _DEADLINE:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # Strategy:
    # 1. If n_unvisited is small (<= 12), use Held-Karp DP for exact optimal completion.
    #    This guarantees the best possible suffix.
    # 2. If n_unvisited is larger, use a sophisticated heuristic:
    #    - 2-step lookahead: d(cur, u) + min_{v != u} d(u, v)
    #    - Enhanced with a "return to destination" penalty if destination is close to the end
    #    - Actually, for the prefix, we just want to minimize d(cur, u) + min_neighbor(u).
    #    - Let's try 3-step lookahead for better quality:
    #      Score(u) = d(cur, u) + min_v (d(u, v) + min_w (d(v, w)))
    #      This is O(N^3) which for N=50 is 125,000 ops, very fast.

    if n_unvisited <= 12:
        # Held-Karp DP
        # Map unvisited nodes to indices 0..n-1
        nodes = unvisited_nodes
        n = len(nodes)
        
        # dp[mask][last] = min cost to visit set mask (bitmask) ending at node 'last' (index in nodes)
        # Start from current_node, which is not in nodes.
        # So we need to add the first step: d(current, nodes[i])
        
        # We want to find the min cost path: current -> v1 -> v2 -> ... -> vn -> destination
        # Let's define dp[mask][i] as min cost to reach nodes[i] having visited exactly the set of nodes in mask (where i is in mask).
        # Initial state: mask has only bit i set. cost = d(current, nodes[i]).
        
        size = 1 << n
        dp = np.full((size, n), np.inf)
        
        # Initialize
        for i in range(n):
            dp[1 << i, i] = distance_matrix[current_node, nodes[i]]
        
        # Fill DP
        # Iterate over all masks
        # For each mask, for each i in mask, for each j not in mask:
        # dp[mask | (1<<j)][j] = min(dp[mask | (1<<j)][j], dp[mask][i] + d(nodes[i], nodes[j]))
        
        # This is O(2^n * n^2). For n=12, 4096 * 144 ~ 590k ops. Fast.
        
        for mask in range(size):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                cost = dp[mask, i]
                if cost == np.inf:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = cost + distance_matrix[nodes[i], nodes[j]]
                    if new_cost < dp[new_mask, j]:
                        dp[new_mask, j] = new_cost
        
        # Find the best completion
        full_mask = size - 1
        best_cost = np.inf
        best_next = int(nodes[0])
        
        for i in range(n):
            if dp[full_mask, i] < np.inf:
                total_cost = dp[full_mask, i] + distance_matrix[nodes[i], destination_node]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_next = int(nodes[i])
                    
        # We need to return the FIRST node in the optimal path, not the last.
        # The DP above gives the cost of the path ending at i.
        # To find the first node, we need to backtrack or reconstruct.
        # Alternatively, we can just pick the first node that leads to the optimal total cost.
        # How to do that?
        # Let optimal_total = best_cost.
        # We want v1 such that d(current, v1) + min_cost_path(v1, U\{v1}, dest) = optimal_total.
        # min_cost_path(v1, U\{v1}, dest) is the cost of the optimal path starting at v1, visiting all other unvisited, ending at dest.
        # This is equivalent to:
        #   min_{perm of U\{v1}} sum(d(perm[k], perm[k+1])) + d(perm[-1], dest)
        #   where perm[0] is the node after v1.
        #   Actually, the DP state dp[mask][i] is the cost from current to i visiting mask.
        #   So the cost from v1 to dest visiting U\{v1} is:
        #   min_{j in U\{v1}} ( d(v1, j) + ... )
        #   This is not directly stored.
        
        # Simpler approach:
        # We know the optimal total cost.
        # Let's iterate over possible first nodes v1.
        # Cost if we pick v1 first:
        # d(current, v1) + min_cost_to_complete(v1, U\{v1}, dest)
        # min_cost_to_complete(v1, U\{v1}, dest) is the min cost path starting at v1, visiting all nodes in U\{v1}, ending at dest.
        # This can be computed by a smaller DP or by reusing the previous DP?
        # The previous DP starts from 'current'.
        # Let's just compute it for each candidate v1.
        # Since n <= 12, and we are in the small case, we can afford to do this.
        # But wait, we already did the full DP.
        
        # Let's reconstruct the path.
        # We know the best last node is the one that gave best_cost.
        # Let's call it last_idx.
        # Then we backtrack to find the previous node.
        # prev_idx = argmin_{j in mask\{last}} (dp[mask\{last}][j] + d(nodes[j], nodes[last]))
        # Continue until mask has only 1 bit. That bit is the first node.
        
        # Let's implement backtracking.
        mask = full_mask
        last_idx = -1
        for i in range(n):
            if dp[full_mask, i] + distance_matrix[nodes[i], destination_node] == best_cost:
                last_idx = i
                break
        
        if last_idx == -1:
            # Fallback
            return int(nodes[np.argmin(distance_matrix[current_node, nodes])])
            
        path = [last_idx]
        current_mask = full_mask
        curr_idx = last_idx
        
        while (1 << curr_idx) != current_mask:
            prev_mask = current_mask ^ (1 << curr_idx)
            if prev_mask == 0:
                break
            min_cost = np.inf
            prev_idx = -1
            for j in range(n):
                if prev_mask & (1 << j):
                    c = dp[prev_mask, j] + distance_matrix[nodes[j], nodes[curr_idx]]
                    if c < min_cost:
                        min_cost = c
                        prev_idx = j
            if prev_idx == -1:
                break
            path.append(prev_idx)
            curr_idx = prev_idx
            current_mask = prev_mask
            
        # path is in reverse order (last to first)
        # The first node is path[-1]
        first_idx = path[-1]
        return int(nodes[first_idx])

    else:
        # Heuristic for larger n
        # 3-step lookahead
        # Score(u) = d(cur, u) + min_{v != u} (d(u, v) + min_{w != u, v} d(v, w))
        
        dist_current = distance_matrix[current_node, unvisited_nodes]
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        
        # min1[v] = min_{w != v} d(v, w)
        np.fill_diagonal(sub_dist, np.inf)
        min1 = np.min(sub_dist, axis=1)
        
        # For each u, we need min_{v != u} (d(u, v) + min_{w != u, v} d(v, w))
        # Let's compute min2[u] = min_{v != u} (d(u, v) + min_{w != u, v} d(v, w))
        
        # This is a bit tricky to vectorize fully.
        # For each u:
        #   For each v != u:
        #     cost = d(u, v) + min_{w != u, v} d(v, w)
        #     Note that min_{w != u, v} d(v, w) is NOT simply min1[v] because w cannot be u.
        #     If argmin1[v] != u, then min_{w != u, v} d(v, w) = min1[v].
        #     If argmin1[v] == u, then min_{w != u, v} d(v, w) = second_min1[v].
        
        # Let's compute min1 and argmin1, and second_min1.
        # sub_dist has inf on diagonal.
        # min1 = min along axis 1.
        # argmin1 = argmin along axis 1.
        
        # To get second min, we can sort or use a trick.
        # For N=50, we can just loop over u.
        
        min2 = np.zeros(n_unvisited)
        
        for u in range(n_unvisited):
            best_val = np.inf
            for v in range(n_unvisited):
                if u == v:
                    continue
                d_uv = sub_dist[u, v]
                # Find min d(v, w) for w != u, v
                # This is min of sub_dist[v, :] excluding index v (inf) and index u.
                # Let's just compute it.
                # sub_dist[v, :] has inf at v.
                # We need min of sub_dist[v, :] where index != u and index != v.
                # Since sub_dist[v, v] is inf, we just need to exclude u.
                # So, if argmin1[v] != u, then min is min1[v].
                # If argmin1[v] == u, then we need the second smallest in sub_dist[v, :].
                
                # Let's precompute argmin1 and second_min1 for all v.
                # This can be done outside the loop.
                
            # Let's precompute outside.
            pass

        # Precompute min1, argmin1, second_min1
        min1 = np.min(sub_dist, axis=1)
        argmin1 = np.argmin(sub_dist, axis=1)
        
        # second_min1[v] = min_{w != v, w != argmin1[v]} d(v, w)
        # We can compute this by sorting each row, or by:
        # second_min1[v] = min( min(sub_dist[v, :][sub_dist[v, :] < min1[v] + eps]), ... )
        # Simpler: for each v, find the two smallest values.
        # Since N=50, we can just do:
        second_min1 = np.zeros(n_unvisited)
        for v in range(n_unvisited):
            # Find two smallest in sub_dist[v, :]
            # sub_dist[v, v] is inf, so it won't be picked unless all are inf (not the case)
            # We can use np.partition or just sort
            # np.partition(sub_dist[v], 2)[1] gives the second smallest.
            # But we need to be careful if there are ties.
            # Actually, we just need the value.
            # Let's use np.argpartition
            # kth=1 gives the second smallest (0-indexed, so k=1 is the 2nd element)
            # Wait, np.argpartition(a, k) returns indices such that the element at index k is in its sorted position, and all before are <=, all after are >=.
            # So np.partition(sub_dist[v], 1)[1] is the second smallest value.
            second_min1[v] = np.partition(sub_dist[v], 1)[1]
        
        # Now compute min2[u]
        for u in range(n_unvisited):
            best_val = np.inf
            for v in range(n_unvisited):
                if u == v:
                    continue
                d_uv = sub_dist[u, v]
                if argmin1[v] == u:
                    min_vw = second_min1[v]
                else:
                    min_vw = min1[v]
                val = d_uv + min_vw
                if val < best_val:
                    best_val = val
            min2[u] = best_val
        
        scores = dist_current + min2
        best_local_idx = np.argmin(scores)
        return int(unvisited_nodes[best_local_idx])

# EVOLVE-BLOCK-END
