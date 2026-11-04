import time
import random
import numpy as np

def construct(k, g):
    # For k=4, g=10, we use a voltage graph lift of K_3,3
    if k != 4 or g != 10:
        # Fallback to LUW if parameters change, though task is specific
        return _luw_fallback(k, g)
    
    deadline = time.time() + 110.0
    best_n = float('inf')
    best_edges = None
    
    # Base graph K_3,3
    # Vertices: 0,1,2 (left), 3,4,5 (right)
    # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    # We assign voltages to these 9 edges.
    # Group Z_m
    
    # Minimum m such that 6*m >= 242 is m=41
    # We try m from 41 upwards
    
    for m in range(41, 100):
        if time.time() > deadline:
            break
            
        # Random search for voltage assignment
        # We need to check all cycles of length 4, 6, 8 in K_3,3
        # K_3,3 has 9 edges.
        # Number of 4-cycles: 9
        # Number of 6-cycles: 6
        # Number of 8-cycles: ?
        
        # Instead of enumerating all cycles, we can check the condition 
        # that the voltage sum around any closed walk of length < 10 is non-zero.
        # Since K_3,3 is bipartite, only even lengths.
        
        # Let's use simulated annealing or random restarts
        # Objective: minimize number of zero-sum cycles of length 4, 6, 8.
        
        found = False
        for restart in range(200):
            if time.time() > deadline:
                break
                
            voltages = [random.randint(1, m-1) for _ in range(9)]
            
            # Check constraints
            valid = True
            # 4-cycles in K_3,3:
            # (0,3)-(1,3)-(1,4)-(0,4) -> edges e0, e3, e4, e1 (indices depend on mapping)
            # Let's map edges:
            # e0: 0-3
            # e1: 0-4
            # e2: 0-5
            # e3: 1-3
            # e4: 1-4
            # e5: 1-5
            # e6: 2-3
            # e7: 2-4
            # e8: 2-5
            
            # 4-cycles:
            # 0-3-1-4-0: e0, e3, e4, e1. Sum = v0 - v3 - v4 + v1 (depending on orientation)
            # Actually, sum of voltages along the path.
            # Path 0->3 (v0), 3->1 (-v3), 1->4 (v4), 4->0 (-v1).
            # Sum = v0 - v3 + v4 - v1.
            
            # We can precompute the list of edge indices for each cycle.
            # There are 9 4-cycles.
            # There are 6 6-cycles.
            # There are 12 8-cycles?
            
            # Let's just enumerate all simple cycles of length 4, 6, 8 in K_3,3.
            # Or simpler: check all pairs of vertices at distance 2, 3, 4?
            
            # Efficient check:
            # For each cycle, compute sum.
            
            # List of 4-cycles (edge indices in order):
            # 0-3-1-4-0: [0, 3, 4, 1]
            # 0-3-1-5-0: [0, 3, 5, 2]
            # 0-3-2-4-0: [0, 6, 7, 1]
            # 0-3-2-5-0: [0, 6, 8, 2]
            # 0-4-1-5-0: [1, 4, 5, 2]
            # 0-4-2-5-0: [1, 7, 8, 2]
            # 1-3-2-4-1: [3, 6, 7, 4]
            # 1-3-2-5-1: [3, 6, 8, 5]
            # 1-4-2-5-1: [4, 7, 8, 5]
            
            # 6-cycles:
            # 0-3-1-5-2-4-0: [0, 3, 5, 8, 7, 1]
            # 0-3-2-5-1-4-0: [0, 6, 8, 5, 4, 1]
            # 0-4-1-3-2-5-0: [1, 4, 3, 6, 8, 2]
            # 0-4-2-3-1-5-0: [1, 7, 6, 3, 5, 2]
            # 0-5-1-3-2-4-0: [2, 5, 3, 6, 7, 1]
            # 0-5-2-3-1-4-0: [2, 8, 6, 3, 4, 1]
            
            # 8-cycles:
            # 0-3-1-5-2-4-... wait, 0-3-1-5-2-4-0 is 6-cycle.
            # 8-cycle must visit all 6 vertices? No, 8 vertices in lift.
            # In base, length 8 walk.
            # Example: 0-3-1-4-2-5-1-3-0? No, simple cycle in base.
            # K_3,3 has 8-cycles?
            # Yes, e.g., 0-3-1-4-2-5-... wait, 0-3-1-4-2-5-?
            # 0-3-1-4-2-5-0 is 6-cycle.
            # To get 8, we must repeat a vertex? No, simple cycle.
            # Max cycle length in K_3,3 is 6.
            # So there are NO 8-cycles in K_3,3!
            # Therefore, no 8-cycles in the lift that project to simple 8-cycles in base.
            # But could there be a 8-cycle in the lift that projects to a non-simple walk in base?
            # A cycle in the lift is a simple cycle. Its projection is a closed walk.
            # If the base has no 8-cycles, the projection must revisit vertices.
            # However, if the projection revisits a vertex, it forms a smaller cycle in the base?
            # Not necessarily.
            # But generally, if the base has girth 4 and max cycle 6, 
            # we only need to check 4 and 6 cycles?
            # Actually, any closed walk of length L can be decomposed into simple cycles.
            # If all simple cycles of length < 10 have non-zero sum, 
            # does it imply all closed walks of length < 10 have non-zero sum?
            # No. A sum of non-zero elements can be zero.
            # So we MUST check all closed walks of length 4, 6, 8.
            # In K_3,3, closed walks of length 8 exist (e.g., go around a 4-cycle twice? No, that's not a simple cycle in lift).
            # Wait, if the lift has a cycle of length 8, its projection is a closed walk of length 8.
            # If the projection is a 4-cycle traversed twice, the sum is 2 * (sum of 4-cycle).
            # If sum of 4-cycle is $s \ne 0$, is $2s \ne 0$?
            # In $Z_m$, if $m$ is odd, $2s=0 \implies s=0$.
            # Since $m=41$ is odd, if 4-cycle sum is non-zero, double sum is non-zero.
            # So we don't need to check "double 4-cycles".
            # What about 4+4 cycles?
            # What about 6+2? No 2-cycles.
            # What about 4+6? Length 10.
            # So for $m$ odd, we only need to check simple cycles of length 4 and 6?
            # And maybe 8-cycles that are simple in the lift but project to non-simple walks?
            # If a closed walk of length 8 in base is not a simple cycle, it contains a smaller cycle.
            # If all smaller cycles have non-zero sum, does the whole walk have non-zero sum?
            # Not necessarily.
            # However, for $m=41$, the number of walks is small. We can just enumerate all closed walks of length 4, 6, 8 in the base and check their sums.
            
            # Let's implement a generic checker.
            
            def is_valid_assignment(v, m):
                # v is list of 9 voltages
                # Check all closed walks of length 4, 6, 8 in K_3,3
                # We can do a DFS from each vertex, tracking current vertex, previous vertex, length, and sum.
                # To avoid backtracking, we don't go back to previous vertex.
                
                def dfs(start, curr, prev, length, sum_val, target_len):
                    if length == target_len:
                        if curr == start and sum_val % m == 0:
                            return False
                        return True
                    if length > target_len:
                        return True
                    for next_v in range(6):
                        if next_v == prev:
                            continue
                        # Find edge index between curr and next_v
                        # Map (curr, next_v) to edge index
                        # 0-3:0, 0-4:1, 0-5:2
                        # 1-3:3, 1-4:4, 1-5:5
                        # 2-3:6, 2-4:7, 2-5:8
                        if curr < 3 and next_v >= 3:
                            idx = curr * 3 + (next_v - 3)
                            # 0*3+0=0, 0*3+1=1, 0*3+2=2
                            # 1*3+0=3, ...
                            # 2*3+0=6, ...
                            # This works for left->right
                            new_sum = (sum_val + v[idx]) % m
                        elif curr >= 3 and next_v < 3:
                            idx = next_v * 3 + (curr - 3)
                            new_sum = (sum_val - v[idx]) % m
                        else:
                            continue # Not an edge in K_3,3
                    
                    if not dfs(start, next_v, curr, length + 1, new_sum, target_len):
                        return False
                    # Optimization: if we fail, return False immediately
                # Actually, we need to check if ANY walk has sum 0.
                # So if we find one with sum 0, it's invalid.
                
                for start in range(6):
                    for first in range(6):
                        if first == start: continue
                        # Check if edge exists
                        if (start < 3 and first >= 3) or (start >= 3 and first < 3):
                            if start < 3 and first >= 3:
                                idx = start * 3 + (first - 3)
                                s = v[idx]
                            else:
                                idx = first * 3 + (start - 3)
                                s = -v[idx]
                            if not dfs(start, first, start, 1, s % m, 4) or \
                               not dfs(start, first, start, 1, s % m, 6) or \
                               not dfs(start, first, start, 1, s % m, 8):
                                return False
                return True
            
            if is_valid_assignment(voltages, m):
                # Check connectivity
                # Build the lift graph
                n_lift = 6 * m
                # Vertices are (base_v, coset)
                # Edges: (u, c) -- (v, c + voltage(u,v))
                # We can check connectivity via BFS on the lift.
                # But n_lift is ~246. BFS is fast.
                
                adj = [[] for _ in range(n_lift)]
                for i in range(6):
                    for j in range(6):
                        if (i < 3 and j >= 3) or (i >= 3 and j < 3):
                            if i < 3 and j >= 3:
                                idx = i * 3 + (j - 3)
                                vol = voltages[idx]
                            else:
                                idx = j * 3 + (i - 3)
                                vol = -voltages[idx]
                            for c in range(m):
                                u = i * m + c
                                v_id = j * m + (c + vol) % m
                                adj[u].append(v_id)
                                adj[v_id].append(u)
                
                # BFS
                visited = [False] * n_lift
                visited[0] = True
                queue = [0]
                count = 1
                while queue:
                    u = queue.pop(0)
                    for v_id in adj[u]:
                        if not visited[v_id]:
                            visited[v_id] = True
                            count += 1
                            queue.append(v_id)
                
                if count == n_lift:
                    # Valid!
                    best_n = n_lift
                    best_edges = [(u, v_id) for u in range(n_lift) for v_id in adj[u] if u < v_id]
                    found = True
                    break
        if found:
            break

    if best_edges is None:
        # Fallback to LUW
        return _luw_fallback(k, g)
        
    return best_n, best_edges

def _luw_fallback(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
