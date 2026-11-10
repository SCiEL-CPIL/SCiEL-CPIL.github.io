"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using branch and bound with greedy coloring
    best_clique = []
    
    def greedy_color(candidates):
        coloring = {}
        color = 0
        uncolored = list(candidates)
        while uncolored:
            color += 1
            new_uncolored = []
            for v in uncolored:
                can_color = True
                for u in uncolored:
                    if u == v:
                        continue
                    if coloring.get(u) == color and u in adj[v]:
                        can_color = False
                        break
                if can_color:
                    coloring[v] = color
                else:
                    new_uncolored.append(v)
            uncolored = new_uncolored
        return coloring, color
    
    def expand(clique, candidates):
        nonlocal best_clique
        if len(clique) > len(best_clique):
            best_clique = list(clique)
        
        if not candidates:
            return
        
        coloring, max_color = greedy_color(candidates)
        cands = sorted(candidates, key=lambda x: -coloring[x])
        
        for v in cands:
            if len(clique) + coloring[v] <= len(best_clique):
                break
            new_clique = clique + [v]
            new_cands = [u for u in cands if u in adj[v]]
            expand(new_clique, new_cands)
    
    expand([], list(range(n)))
    
    branches = [int(v) for v in best_clique]
    if len(branches) < 2:
        u, v = P['edges'][0]
        branches = [int(u), int(v)]
    
    # Find vertex-disjoint paths between all pairs of branch vertices
    # Use BFS with vertex constraints
    branch_set = set(branches)
    
    paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            u, v = branches[i], branches[j]
            
            # BFS to find shortest path from u to v avoiding other branches
            # Path must start at u, end at v, and interiors avoid all branches
            from collections import deque
            
            # We need path where interiors don't touch any branch
            # So we can only use non-branch vertices as intermediates
            visited = set()
            visited.add(u)
            queue = deque([(u, [u])])
            found = None
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found = path
                    break
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    # Interiors cannot be branches (except endpoints)
                    if neighbor in branch_set and neighbor != u and neighbor != v:
                        continue
                    if neighbor == v:
                        # Can end at v
                        found = path + [v]
                        break
                    # Non-branch interior vertex
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                if found:
                    break
            
            if found is None:
                # Fallback: try allowing other branches as interiors if no path found
                # But per contract, interiors must avoid every branch
                # So if no path exists, we have a problem
                # For now, just use direct edge if available
                if v in adj[u]:
                    found = [u, v]
                else:
                    # This shouldn't happen for a valid clique in a connected graph
                    found = [u, v]
            
            paths.append(found)
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
