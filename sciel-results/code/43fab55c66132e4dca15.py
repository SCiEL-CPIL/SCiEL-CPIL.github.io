import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Identify largest connected component to focus search
    # If the graph is disconnected, the longest path must be within the largest component
    visited_comp = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            comp = []
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > len(largest_comp):
                largest_comp = comp
            
    if not largest_comp:
        return [0]

    # If the largest component is small, we can potentially find the exact longest path
    # or use a more exhaustive search.
    # For larger components, we use IDDFS with pruning.
    
    best_path = []
    best_len = 0
    
    # Initial greedy to get a decent lower bound quickly
    def greedy_warnsdorff(start_node):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            next_node = min(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path

    for i in largest_comp:
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    if time.time() > deadline:
        return best_path

    # IDDFS Strategy
    # We try to find a path of length L + 1, where L starts from best_len.
    # We use a recursive DFS with a depth limit.
    # Pruning: If we are at depth d, and the maximum possible depth is L, 
    # we can prune if the remaining unvisited neighbors don't allow reaching L.
    # More importantly, we use "forced move" logic: 
    # If a node has only 1 unvisited neighbor, we MUST go there.
    # If a node has >1, we explore them, but we prioritize those that lead to "dead ends" 
    # to fail fast? No, we want to succeed. 
    # Actually, standard IDDFS is good for finding *any* path of length L.
    # To maximize, we increment L.
    
    # Optimization: Only search within the largest component.
    # Pre-compute degrees within the component? No, just use global adj and filter by visited.
    
    # To make IDDFS efficient, we order neighbors.
    # Heuristic: Visit neighbors with LOWEST degree first (Warnsdorff) to force the path 
    # into "narrow" parts of the graph, potentially leaving "wide" parts for later? 
    # Or vice versa? 
    # Actually, for longest path, Warnsdorff (lowest degree) is often good for Hamiltonian paths.
    
    def iddfs_search(target_len):
        # Returns a path of length target_len if found, else None
        # We try all start nodes in the largest component
        # To speed up, we can use a global visited array and backtrack manually, 
        # but Python recursion with sets is okay for n=100 if depth is limited.
        
        # We'll use a stack-based approach or recursion. 
        # Given n=100, recursion depth 100 is fine.
        
        # State: path, visited_set
        # We want to find a path of exactly target_len.
        
        # We can prune if len(path) + (number of unvisited nodes reachable from path[-1]) < target_len
        # This is expensive to compute.
        
        # Instead, we rely on the fact that if we can't extend, we backtrack.
        
        # To avoid repeated work, we can use a "memoization" of dead ends? 
        # (path[-1], frozenset(visited)) -> bool. This is likely too big.
        
        # Let's just do the DFS with a strict time check.
        
        def dfs(node, path, visited):
            if time.time() > deadline:
                return None
            if len(path) == target_len:
                return list(path)
            
            # Heuristic: If we are close to target, be more careful.
            # Get unvisited neighbors
            neighbors = [x for x in adj[node] if x not in visited]
            
            if not neighbors:
                return None
            
            # Sort neighbors by degree (Warnsdorff) to try "safe" moves first?
            # Or try "dangerous" moves first to fail fast?
            # Usually, Warnsdorff (lowest degree) is better for finding long paths.
            
            # Forced move check:
            # If any neighbor has only 1 unvisited neighbor (which must be 'node' effectively, 
            # or rather, if we don't go to it, it gets isolated), we must go to it.
            # Wait, if neighbor x has len(adj[x] - visited) == 1, and that neighbor is NOT node, 
            # then x is not a neighbor of node? No, x is a neighbor of node.
            # So if x is a neighbor of node, and x has only 1 unvisited neighbor, 
            # that unvisited neighbor MUST be node? No, node is already visited.
            # So x's unvisited neighbors are in adj[x] - visited.
            # If x is a candidate, x is not in visited.
            # If len(adj[x] - visited) == 1, let y be that neighbor.
            # If y == node, then x is a dead end if we don't go to it? 
            # No, if we don't go to x, x remains unvisited. 
            # But x only connects to node (which is visited) and no other unvisited nodes.
            # So x will never be reachable again. 
            # So if there is a neighbor x such that len(adj[x] - visited) == 1, 
            # we MUST visit x now, otherwise x is lost forever.
            
            forced = []
            for x in neighbors:
                # Count unvisited neighbors of x
                # Note: node is in visited, so it's excluded from adj[x]-visited.
                # If x has any other unvisited neighbors, count > 1.
                # If count == 1, the only unvisited neighbor of x is... wait.
                # adj[x] contains node. node is in visited.
                # So adj[x] - visited contains neighbors of x that are NOT visited.
                # If this set has size 1, let's call it {y}.
                # y is unvisited.
                # If we don't go to x now, can we go to x later?
                # Only if we reach x from y.
                # But if we go to y later, we might not be able to reach x if the path to y doesn't go through x?
                # Actually, if we don't go to x, x is isolated from the rest of the unvisited graph 
                # EXCEPT through y.
                # So if we go to y, we CAN go to x.
                # So it's NOT a forced move unless y is also a dead end?
                
                # Let's refine: 
                # A node x is "forced" if it has only 1 unvisited neighbor y, AND y is also "forced" or similar?
                # This is getting complex. 
                
                # Simpler heuristic: 
                # Prioritize neighbors with LOWEST degree in the remaining graph.
                # This is Warnsdorff.
                
                pass
            
            # Sort by remaining degree
            def rem_deg(x):
                return len(adj[x] - visited)
            
            neighbors.sort(key=rem_deg)
            
            # Try each neighbor
            for x in neighbors:
                path.append(x)
                visited.add(x)
                res = dfs(x, path, visited)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)
                
            return None

        # Try all start nodes
        # To make this faster, we can randomize the start order or use the fact that 
        # endpoints of long paths are often low-degree nodes.
        
        # Sort start nodes by degree (lowest first)
        start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))
        
        for start in start_nodes:
            if time.time() > deadline:
                return None
            path = [start]
            visited = {start}
            res = dfs(start, path, visited)
            if res is not None:
                return res
                
        return None

    # Iterate increasing target length
    # We know best_len. Let's try best_len + 1, then +2, etc.
    # But IDDFS from scratch for each length is inefficient.
    # Better: Run DFS with a depth limit that increases?
    # Or just run a "deep" DFS once with a good heuristic?
    
    # Given the time limit, let's try a few specific lengths.
    # If we can't find best_len + 1, maybe we can find best_len + 5?
    # IDDFS guarantees we find the shortest path of a certain property, but here we want LONGEST.
    # So we should try to find a path of length L. If found, update best.
    # Start L = best_len + 1.
    
    # To avoid getting stuck in one long search, we time-box each L.
    
    current_target = best_len + 1
    max_possible = len(largest_comp)
    
    while current_target <= max_possible and time.time() < deadline - 5.0:
        # Time budget per target length: 2 seconds
        search_start = time.time()
        
        found_path = iddfs_search(current_target)
        
        if found_path:
            best_len = current_target
            best_path = found_path
            current_target += 1
            # Reset search start for next iteration
        else:
            # If we couldn't find a path of length current_target within the time budget,
            # it's likely we can't find it at all, or it's too hard.
            # Since we are using a heuristic (Warnsdorff) inside DFS, failure might be due to heuristic.
            # But if we failed to find ANY path of length L, we can't have a path of length L+1.
            # So we can break.
            break
            
    return best_path
# EVOLVE-BLOCK-END
