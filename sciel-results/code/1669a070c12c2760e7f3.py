# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list for original graph
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Construct Bipartite Graph
    # Left: 0..n-1 (out)
    # Right: n..2n-1 (in)
    # Edge u->v in orig => u -> v+n in bipartite
    bip_adj = [[] for _ in range(2 * n)]
    for u, v in edges:
        bip_adj[u].append(v + n)
        bip_adj[v].append(u + n)
        
    # 2. Hopcroft-Karp Algorithm
    # pairU[i] = matched node in Right for Left node i, or -1
    # pairV[j] = matched node in Left for Right node j, or -1
    pairU = [-1] * n
    pairV = [-1] * n
    dist = [0] * n
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pairU[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in bip_adj[u]:
                v_left = v - n
                if pairV[v] == -1:
                    found = True
                elif dist[pairV[v]] == -1:
                    dist[pairV[v]] = dist[u] + 1
                    queue.append(pairV[v])
        return found
        
    def dfs(u):
        for v in bip_adj[u]:
            v_left = v - n
            if pairV[v] == -1 or (dist[pairV[v]] == dist[u] + 1 and dfs(pairV[v])):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = -1
        return False
        
    matching_size = 0
    while bfs():
        for u in range(n):
            if pairU[u] == -1:
                if dfs(u):
                    matching_size += 1
                    
    # 3. Reconstruct Paths
    # next_node[v] = u if v_out is matched to u_in (v->u)
    # prev_node[v] = u if v_in is matched to u_out (u->v)
    next_node = [-1] * n
    prev_node = [-1] * n
    
    for u in range(n):
        if pairU[u] != -1:
            v = pairU[u] - n
            next_node[u] = v
        if pairV[u + n] != -1:
            u_in_match = pairV[u + n]
            prev_node[u] = u_in_match
            
    paths = []
    visited = [False] * n
    
    for i in range(n):
        if prev_node[i] == -1 and not visited[i]:
            # Start of a path
            path = [i]
            visited[i] = True
            cur = i
            while next_node[cur] != -1:
                nxt = next_node[cur]
                path.append(nxt)
                visited[nxt] = True
                cur = nxt
            paths.append(path)
            
    # Also handle cycles (if any, though matching on bipartite for path cover shouldn't produce cycles 
    # unless we allow self-loops or something, but here we just trace. 
    # If a node is in a cycle, it will have prev and next. 
    # The above loop starts only at nodes with no prev. 
    # If there is a cycle, those nodes won't be visited. 
    # We should handle cycles by breaking them.)
    
    for i in range(n):
        if not visited[i]:
            # It's in a cycle. Break it by treating it as a path start.
            path = [i]
            visited[i] = True
            cur = i
            while next_node[cur] != -1 and not visited[next_node[cur]]:
                nxt = next_node[cur]
                path.append(nxt)
                visited[nxt] = True
                cur = nxt
            if len(path) > 1:
                paths.append(path)
            else:
                paths.append([i]) # Single node path

    # 4. Merge Paths
    # We want to connect paths using edges in the original graph.
    # Greedy approach: repeatedly find two paths that can be connected and merge them.
    # To allow reversing, we can check both orientations.
    
    changed = True
    while changed:
        changed = False
        merged_paths = []
        used = [False] * len(paths)
        
        for i in range(len(paths)):
            if used[i]: continue
            for j in range(i + 1, len(paths)):
                if used[j]: continue
                
                p1 = paths[i]
                p2 = paths[j]
                
                # Check if end of p1 connects to start of p2
                if p1[-1] in adj[p2[0]]:
                    merged_paths.append(p1 + p2)
                    used[i] = True
                    used[j] = True
                    changed = True
                    break
                # Check if end of p2 connects to start of p1
                elif p2[-1] in adj[p1[0]]:
                    merged_paths.append(p2 + p1)
                    used[i] = True
                    used[j] = True
                    changed = True
                    break
                # Check if start of p1 connects to end of p2 (reverse p1? No, just swap order)
                # Actually, if we can connect end of p2 to start of p1, we did p2+p1.
                # What if we need to reverse a path?
                # If start of p1 connects to end of p2, we can do reverse(p1) + p2?
                # No, reverse(p1) ends at p1[0]. So if p1[0] connects to p2[-1], we can do reverse(p1) + p2?
                # Wait. Path is directed.
                # If we have path A = [a1, a2, a3] and B = [b1, b2].
                # If edge a3-b1 exists, we do A+B.
                # If edge b2-a1 exists, we do B+A.
                # If edge a1-b2 exists, we can reverse A to [a3, a2, a1] and then B + reverse(A)?
                # B ends at b2. reverse(A) starts at a3. No.
                # B ends at b2. reverse(A) starts at a3.
                # We need edge from end of first to start of second.
                # If we use reverse(A) as the second part: B + reverse(A).
                # End of B is b2. Start of reverse(A) is a3.
                # So we need edge b2-a3.
                # We already checked p2[-1] in adj[p1[0]] which is b2 in adj[a1].
                # We did NOT check b2 in adj[a3].
                
                # Let's add checks for reversed connections.
                # If p2[-1] connects to p1[-1], we can do p2 + reverse(p1)
                if p2[-1] in adj[p1[-1]]:
                    merged_paths.append(p2 + p1[::-1])
                    used[i] = True
                    used[j] = True
                    changed = True
                    break
                # If p1[-1] connects to p2[-1], we can do p1 + reverse(p2)
                if p1[-1] in adj[p2[-1]]:
                    merged_paths.append(p1 + p2[::-1])
                    used[i] = True
                    used[j] = True
                    changed = True
                    break

        if changed:
            # Keep unmerged paths
            for i in range(len(paths)):
                if not used[i]:
                    merged_paths.append(paths[i])
            paths = merged_paths
            
    # Return the longest path
    best_path = max(paths, key=len) if paths else []
    return best_path
# EVOLVE-BLOCK-END
