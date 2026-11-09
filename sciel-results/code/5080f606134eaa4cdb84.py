"""Program for hexagon-free graphs on 40 vertices with many edges."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 40
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    random.seed(42)
    
    # Bitset representation of adjacency
    # adj[u] is an integer where bit v is 1 if (u,v) is an edge
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_exists(u, v):
        return (adj[u] >> v) & 1
        
    def count_paths_len_5(u, v):
        """
        Count the number of simple paths of length 5 from u to v in the current graph.
        Note: This assumes the edge (u,v) is NOT present in the graph for the path to be 'new' 
        or 'removed'. Actually, a C6 containing (u,v) corresponds to a path of length 5 
        from u to v in G \ {(u,v)}.
        """
        count = 0
        # We want paths u -> n1 -> n2 -> n3 -> n4 -> v
        # We can do DFS from u, depth 5, target v.
        # To speed up, we can meet in the middle or just DFS.
        # DFS with memoization on (node, depth, visited_mask) is too memory heavy for visited_mask.
        # Simple DFS is fine for N=40, depth=5.
        
        def dfs(current, depth, visited):
            nonlocal count
            if depth == 0:
                if current == v:
                    count += 1
                return
            
            # If depth == 1, we must go to v
            if depth == 1:
                if (adj[current] >> v) & 1 and not (visited >> v) & 1:
                    count += 1
                return

            # Get neighbors of current, excluding visited and v (unless depth==1 handled above)
            neighbors = adj[current] & ~visited
            
            # We shouldn't visit v before the last step
            if depth > 1:
                neighbors &= ~(1 << v)
                
            while neighbors:
                lsb = neighbors & -neighbors
                next_v = lsb.bit_length() - 1
                neighbors ^= lsb
                dfs(next_v, depth - 1, visited | lsb)
                
        dfs(u, 5, 1 << u)
        return count

    def is_c6_free():
        """Check if the current graph is C6-free."""
        # For each edge (u,v), check if there is a path of length 5 between u and v in G \ {(u,v)}.
        # If yes, a C6 exists.
        for u in range(n):
            mask_u = adj[u]
            # Iterate neighbors v of u
            v_mask = mask_u
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                
                if u > v: # Check each edge only once? 
                    # Actually, path count is symmetric. 
                    # We just need to know if ANY edge creates a C6.
                    pass
                
                # Temporarily remove (u,v) to count paths
                # But we can just count paths in G. If a path of length 5 exists in G, 
                # does it imply a C6? 
                # A C6 is u-v-n1-n2-n3-n4-u.
                # This contains edge (u,v) and path u-n1-n2-n3-n4-v of length 5.
                # This path does NOT use edge (u,v).
                # So, if we count paths of length 5 from u to v in G \ {(u,v)}, and it's > 0, 
                # then (u,v) is part of a C6.
                
                # Optimization: If degree is low, maybe skip?
                # Let's just do the check.
                
                # To count paths in G \ {(u,v)}, we need to ignore the edge (u,v) in the DFS.
                # We can modify the DFS to ignore a specific edge.
                
                # Instead of calling the general count_paths_len_5 which uses global adj,
                # we write a specialized one that ignores (u,v).
                
                def dfs_ignore_uf(current, depth, visited):
                    nonlocal count
                    if depth == 0:
                        if current == v:
                            count += 1
                        return
                    
                    if depth == 1:
                        if (adj[current] >> v) & 1 and not (visited >> v) & 1:
                            # Check if the step is (u,v) or (v,u)? 
                            # We are going to v. The previous node is current.
                            # The edge is (current, v).
                            # We want to ignore edge (u,v).
                            # If current == u, we are trying to take edge (u,v). Ignore it.
                            if current != u:
                                count += 1
                        return

                    neighbors = adj[current] & ~visited
                    
                    # Ignore v if depth > 1
                    if depth > 1:
                        neighbors &= ~(1 << v)
                        
                    # Ignore u if current is v? No, we don't visit v until depth 1.
                    # But we might visit u? 
                    # The path starts at u. u is in visited. So u is never a neighbor.
                    
                    while neighbors:
                        lsb = neighbors & -neighbors
                        next_v = lsb.bit_length() - 1
                        neighbors ^= lsb
                        dfs_ignore_uf(next_v, depth - 1, visited | lsb)

                count = 0
                # Start from u, depth 5, visited {u}
                # We want to reach v in 5 steps.
                # The first step is from u.
                # If we go u->v, that's depth 1. But we need depth 5.
                # So we must NOT go to v at depth 1? 
                # Wait, the path is u -> ... -> v. Length 5.
                # Steps: 1, 2, 3, 4, 5.
                # At step 5, we arrive at v.
                # So at depth 1 (after 1 step), we are at some node x.
                # ...
                # At depth 4, we are at node y.
                # At depth 1 (remaining), we go to v.
                
                # The DFS logic above:
                # dfs(current, depth, visited)
                # If depth == 1, check if v is neighbor.
                # This counts paths of length 'depth' from 'current' to 'v'.
                # We call dfs(u, 5, 1<<u).
                # This counts paths of length 5 from u to v.
                # These paths do not include the edge (u,v) as an intermediate step 
                # because v is only visited at the end (enforced by neighbors &= ~(1<<v) if depth>1).
                # And the first step is from u. If the first step was v, depth would be 1?
                # No, if we call dfs(u, 5), it looks for neighbors of u.
                # If v is a neighbor, and depth > 1, we skip it.
                # So the path u->v->... is not counted.
                # The path ...->u->v is not counted because u is visited.
                # So this correctly counts paths in G \ {(u,v)}.
                
                dfs_ignore_uf(u, 5, 1 << u)
                if count > 0:
                    return False
        return True

    # 1. Greedy Construction
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        # Fast check: is it C6-free?
        # We can use a lighter check during greedy? 
        # For now, use the exact check. It might be slow for 780 edges.
        # 780 * (DFS cost). 
        # Let's optimize the C6 check for greedy by using the same logic.
        
        # To speed up greedy, we can just add it and check later? No.
        # Let's use the is_c6_free logic but optimized for single edge addition.
        
        # Check if adding (u,v) creates a C6.
        # This is equivalent to: are there paths of length 5 from u to v in G \ {(u,v)}?
        # Since (u,v) is not in G yet, this is just paths of length 5 from u to v in G.
        
        def has_path_len_5(u, v):
            def dfs(current, depth, visited):
                if depth == 0:
                    return current == v
                if depth == 1:
                    return (adj[current] >> v) & 1
                neighbors = adj[current] & ~visited
                if depth > 1:
                    neighbors &= ~(1 << v)
                while neighbors:
                    lsb = neighbors & -neighbors
                    next_v = lsb.bit_length() - 1
                    neighbors ^= lsb
                    if dfs(next_v, depth - 1, visited | lsb):
                        return True
                return False
            return dfs(u, 5, 1 << u)
            
        if not has_path_len_5(u, v):
            add_edge(u, v)
            
    current_edges = sum(bin(a).count("1") for a in adj) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # 2. Simulated Annealing with Penalty
    start_time = time.time()
    time_limit = 110.0
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    lambda_penalty = 10.0 # Penalty for each C6
    
    # We need to track the number of C6s to calculate delta quickly?
    # Or just recalculate? 
    # Recalculating total C6s is O(E * DFS). Too slow.
    # We will use the local delta approach.
    # Score = Edges - Lambda * C6s.
    # When flipping (u,v):
    #   If adding: 
    #     Delta Edges = +1
    #     Delta C6s = + (number of paths len 5 from u to v in G \ {(u,v)})
    #     But wait, adding (u,v) might also destroy C6s? No, adding an edge only creates new cycles.
    #     So Delta C6s is positive.
    #   If removing:
    #     Delta Edges = -1
    #     Delta C6s = - (number of paths len 5 from u to v in G \ {(u,v)})
    #     Removing an edge only destroys cycles containing that edge.
    
    # So we just need to count paths len 5 between u and v in G \ {(u,v)}.
    # This is the same function `count_paths_len_5` if we ensure (u,v) is not counted.
    # My `count_paths_len_5` above uses `adj` directly. 
    # If (u,v) is in `adj`, and we call `count_paths_len_5(u,v)`, 
    # it counts paths in G. 
    # Does it count paths that use (u,v)?
    # The path is u -> ... -> v.
    # If the path uses (u,v), it must be the first or last step.
    # If first step is v: u->v->...->v. Length 5.
    # My DFS skips v if depth > 1. So u->v is only allowed if depth==1?
    # In `count_paths_len_5`, the base case is depth==1.
    # If I call dfs(u, 5), it goes to neighbors.
    # If v is a neighbor, and depth (5) > 1, it skips v.
    # So it does NOT count paths starting with (u,v).
    # What about paths ending with (u,v)?
    # The step into v is at depth 1.
    # The node before v is `current`.
    # The edge is (current, v).
    # If current == u, the edge is (u,v).
    # My `count_paths_len_5` does NOT check if current == u when taking the step to v.
    # So it WILL count paths that use (u,v) as the last step.
    # But a path u->...->u->v is not simple (u repeated).
    # Since u is in visited from the start, `current` can never be u when depth > 0?
    # Wait, `visited` contains u.
    # `neighbors = adj[current] & ~visited`.
    # So u is never a neighbor of any node in the path (except the start, but we move away).
    # So `current` can never be u in the DFS steps.
    # Therefore, the edge (u,v) is never used as (current, v) because current != u.
    # And it's never used as (u, v) at the start because v is skipped if depth > 1.
    # So `count_paths_len_5(u, v)` counts paths in G \ {(u,v)}.
    
    # Great. So we can use `count_paths_len_5` directly.
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Perform a batch of moves
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
                
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            exists = edge_exists(u, v)
            
            # Calculate Delta Score
            # paths = count_paths_len_5(u, v)
            # This is the number of C6s that (u,v) is part of.
            
            # Optimization: If degree of u or v is 0, paths is 0.
            # If graph is sparse, paths is likely 0.
            
            # We only need to count paths if it affects the decision.
            # If adding: Delta Score = 1 - Lambda * paths.
            # If removing: Delta Score = -1 + Lambda * paths.
            
            # If Lambda is large, we avoid adding if paths > 0.
            # We want to allow some C6s to escape local minima.
            
            paths = count_paths_len_5(u, v)
            
            if not exists:
                # Adding
                delta_score = 1.0 - lambda_penalty * paths
                if delta_score > 0:
                    add_edge(u, v)
                    current_edges += 1
                    # Check if valid and update best
                    # Checking validity is expensive. 
                    # We only update best if we think it's good?
                    # Or we track a "valid" flag?
                    # Let's just check validity every K moves or when current_edges > best_edges.
                    if current_edges > best_edges:
                        # Verify if C6-free
                        if is_c6_free():
                            best_edges = current_edges
                            best_adj = adj[:]
                else:
                    # Metropolis
                    if random.random() < np.exp(delta_score / T):
                        add_edge(u, v)
                        current_edges += 1
            else:
                # Removing
                delta_score = -1.0 + lambda_penalty * paths
                if delta_score > 0:
                    remove_edge(u, v)
                    current_edges -= 1
                else:
                    if random.random() < np.exp(delta_score / T):
                        remove_edge(u, v)
                        current_edges -= 1
        
        T *= cooling
        
    # Final check: ensure best_adj is C6-free
    # It was only updated if is_c6_free() passed.
    
    adj = best_adj
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
