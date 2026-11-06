"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
    visited = [False] * n
    largest_cc = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > len(largest_cc):
                largest_cc = comp
    
    if not largest_cc:
        return [0]
    
    cc_set = set(largest_cc)
    cc_nodes = list(largest_cc)
    n_cc = len(cc_nodes)
    
    # Map original nodes to 0..n_cc-1 for easier handling?
    # Or just work with original indices but restrict to cc_set.
    
    # Iterative Deepening DFS for Longest Path
    # We try to find a path of length L, starting from n_cc down to 1.
    
    # Precompute neighbors within CC
    adj_cc = {u: [v for v in adj[u] if v in cc_set] for u in cc_nodes}
    
    def dfs(start, target_len):
        """
        Returns a path of length target_len starting from start, or None.
        Uses backtracking with pruning.
        """
        path = [start]
        visited = {start}
        
        def backtrack(current, remaining_len):
            if remaining_len == 0:
                return path[:]
            
            # Pruning 1: Connectivity of remaining nodes
            # The unvisited nodes must be reachable from 'current' via unvisited nodes.
            # Actually, we just need to ensure that the number of unvisited nodes
            # in the connected component of 'current' (in the graph induced by 
            # unvisited + current) is at least remaining_len.
            
            # Let's compute the reachable unvisited nodes from 'current'
            # through unvisited nodes only.
            # Note: 'current' is visited, so we start from its unvisited neighbors.
            reachable = set()
            stack = []
            for nb in adj_cc[current]:
                if nb not in visited:
                    stack.append(nb)
                    reachable.add(nb)
            
            while stack:
                node = stack.pop()
                for nb in adj_cc[node]:
                    if nb not in visited and nb not in reachable:
                        reachable.add(nb)
                        stack.append(nb)
                        
            if len(reachable) < remaining_len:
                return None
            
            # Pruning 2: Degree constraints (Optional, might be slow to compute)
            # If any node in 'reachable' has degree 1 in the induced subgraph of reachable,
            # it must be an endpoint of the sub-path.
            
            # Choose next node
            # Heuristic: Try neighbors with lower degree first (Warnsdorff)
            candidates = [nb for nb in adj_cc[current] if nb not in visited]
            
            # Sort by degree in the remaining graph?
            # To make it faster, we just sort by global degree or static degree.
            # Dynamic degree is expensive.
            candidates.sort(key=lambda x: len(adj_cc[x]))
            
            for nb in candidates:
                path.append(nb)
                visited.add(nb)
                
                result = backtrack(nb, remaining_len - 1)
                if result is not None:
                    return result
                
                path.pop()
                visited.remove(nb)
                
            return None
        
        return backtrack(start, target_len - 1)

    # Try to find Hamiltonian path (length n_cc)
    # If not, try n_cc - 1, etc.
    # To avoid trying all starts for each length, we can be smarter.
    # But n_cc is at most 100.
    # DFS is exponential, but pruning is strong.
    
    best_path = []
    
    # We can try to find a path of length L for L = n_cc down to 1.
    # Optimization: If we find a path of length L, we don't need to check smaller L for other starts?
    # We want the MAX length. So we stop at the first L for which ANY start yields a path.
    
    for L in range(n_cc, 0, -1):
        found = False
        for start in cc_nodes:
            # Optional: Skip starts that are leaves if L > 1 and start is not a leaf?
            # No, a leaf can be a start.
            # Heuristic: Start from nodes with higher degree? Or lower?
            # Usually, starting from "peripheral" nodes (leaves) is good for Hamiltonian paths in trees.
            # Let's just iterate all.
            
            # Pruning for start:
            # If L > 1, the start node must have at least 1 neighbor.
            if L > 1 and len(adj_cc[start]) == 0:
                continue
                
            path = dfs(start, L)
            if path is not None:
                best_path = path
                found = True
                break
        
        if found:
            break
            
    if not best_path:
        # Fallback: Return a single node or greedy path
        best_path = [cc_nodes[0]]
        
    return best_path
# EVOLVE-BLOCK-END
