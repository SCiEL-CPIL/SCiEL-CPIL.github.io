import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_edges(n):
        return [(i, j) for i in range(n) for j in range(i+1, n)]

    def count_c6_with_edge(adj, u, v):
        # adj is a list of lists of sets
        # Returns number of C6 cycles in the graph defined by adj that contain edge (u,v)
        # This is the number of simple paths of length 5 from v to u in G \ {(u,v)}
        
        # Optimization: If u and v are not connected by a path of length 5, return 0
        # Quick check: if len(adj[v]) < 1 or len(adj[u]) < 1: return 0
        
        count = 0
        # We use a stack for DFS to avoid recursion overhead
        # Stack items: (node, depth, visited_set)
        # visited_set is a bitmask for speed
        
        # Neighbors of v, excluding u
        start_neighbors = [w for w in adj[v] if w != u]
        
        for w1 in start_neighbors:
            # visited: v is 1, w1 is 2
            # We want to reach u at depth 5
            # Path: v(0) - w1(1) - w2(2) - w3(3) - w4(4) - w5(5) - u(6)
            # Wait, edge (u,v) closes the loop.
            # So the path is v -> ... -> u of length 5 edges.
            # v-w1 (1), w1-w2 (2), w2-w3 (3), w3-w4 (4), w4-w5 (5).
            # Then w5 must be connected to u.
            
            # Let's trace:
            # Depth 1: w1
            # Depth 2: w2
            # Depth 3: w3
            # Depth 4: w4
            # Depth 5: w5
            # Check if w5 is connected to u.
            
            # Bitmask for visited: bit i is set if node i is visited.
            mask = (1 << v) | (1 << w1)
            
            stack = [(w1, 1, mask)]
            
            while stack:
                node, depth, m = stack.pop()
                
                if depth == 4:
                    # We are at w4. Next is w5.
                    # w5 must be neighbor of u.
                    # And w5 must not be visited.
                    # And w5 != v.
                    for w5 in adj[node]:
                        if w5 == u:
                            # Path length 4 to u? No, we need length 5.
                            # If depth is 4, we have taken 4 edges from v.
                            # v-w1-w2-w3-w4.
                            # If w4 is connected to u, that's length 4.
                            # We need length 5.
                            # So w4 must be connected to w5, and w5 to u.
                            pass
                        if w5 in m:
                            continue
                        if w5 == v:
                            continue
                        # Check if w5 is connected to u
                        if u in adj[w5]:
                            # Found a cycle: v-w1-w2-w3-w4-w5-u-v
                            count += 1
                    continue
                
                # depth < 4
                for next_node in adj[node]:
                    if next_node in m:
                        continue
                    if next_node == v:
                        continue
                    # We can go to u only if depth+1 == 5? 
                    # If depth+1 == 5, next_node is w5.
                    # If next_node == u, then we have a path of length 5?
                    # v-w1-w2-w3-w4-u. Length 4 edges? No.
                    # v(0)-w1(1)-w2(2)-w3(3)-w4(4)-u(5).
                    # This is a path of 5 edges.
                    # So if depth is 4 and next_node is u, we found a path of length 5.
                    # But wait, the cycle is closed by edge (u,v).
                    # So the path is v to u.
                    # If the path length is 5, it forms a C6 with edge (u,v).
                    
                    if next_node == u:
                        if depth + 1 == 5:
                            count += 1
                        continue
                    
                    # If next_node is not u, we can add to stack
                    # But we must ensure we don't reach u too early.
                    # If depth + 1 < 5, we can go to u? No, that would be a shorter cycle.
                    # We only care about C6.
                    # So we should not visit u until depth 4 (so next is depth 5).
                    
                    # Actually, if we visit u at depth < 4, we form a smaller cycle, which is not C6.
                    # So we just skip u unless depth+1 == 5.
                    
                    if depth + 1 < 5:
                         # We can't go to u yet
                         if next_node == u:
                             continue
                         
                    stack.append((next_node, depth + 1, m | (1 << next_node)))
                    
        return count

    def solve(n, deadline):
        edges = get_edges(n)
        num_edges = len(edges)
        
        # Initialize random coloring
        colors = np.random.randint(0, 4, size=num_edges)
        
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u].add(v)
            adj[c][v].add(u)
            
        # Calculate initial violations
        # We sum count_c6_with_edge for all edges in color c
        # This counts each C6 exactly 6 times.
        # So violations[c] = sum / 6
        violations = [0] * 4
        for c in range(4):
            total = 0
            for idx, (u, v) in enumerate(edges):
                if colors[idx] == c:
                    total += count_c6_with_edge(adj[c], u, v)
            violations[c] = total // 6
            
        total_violations = sum(violations)
        best_colors = colors.copy()
        best_violations = total_violations
        
        T = 1.0
        cooling = 0.9999
        
        while time.time() < deadline:
            idx = random.randint(0, num_edges - 1)
            u, v = edges[idx]
            c_old = colors[idx]
            
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            delta_old = -count_c6_with_edge(adj[c_old], u, v)
            delta_new = count_c6_with_edge(adj[c_new], u, v)
            
            delta = delta_old + delta_new
            
            if delta < 0:
                pass
            elif delta == 0:
                if random.random() > 0.5:
                    continue
            else:
                if random.random() > np.exp(-delta / T):
                    continue
            
            colors[idx] = c_new
            
            adj[c_old][u].remove(v)
            adj[c_old][v].remove(u)
            adj[c_new][u].add(v)
            adj[c_new][v].add(u)
            
            violations[c_old] += delta_old // 6 # Wait, delta is count of C6s containing the edge.
            # If we remove the edge, we remove exactly count_c6_with_edge C6s.
            # So violations[c_old] should decrease by count_c6_with_edge.
            # But my count_c6_with_edge returns the number of C6s containing the edge.
            # So violations[c_old] -= count.
            # violations[c_new] += count.
            
            # Correction:
            # My previous code: violations[c_old] += delta_old
            # delta_old = -count. So violations[c_old] -= count. Correct.
            # But I initialized violations as total // 6.
            # So the unit of violations is "number of C6s".
            # And count_c6_with_edge returns "number of C6s containing edge".
            # So the update is consistent.
            
            # However, I did `violations[c_old] += delta_old // 6` in the draft above. That's wrong.
            # It should be `violations[c_old] += delta_old`.
            
            violations[c_old] += delta_old
            violations[c_new] += delta_new
            
            total_violations = sum(violations)
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return best_colors, n
                
            T *= cooling
            if T < 0.01:
                T = 1.0

        return best_colors, n

    # Try n=17
    colors, n_found = solve(17, deadline)
    
    if n_found == 17:
         return {'n': 17, 'colors': colors.tolist()}
         
    # Try n=16
    colors, n_found = solve(16, deadline)
    if n_found == 16:
         return {'n': 16, 'colors': colors.tolist()}
         
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
