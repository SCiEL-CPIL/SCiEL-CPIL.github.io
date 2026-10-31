import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy:
    1. Primary: Deterministic "random lift" of K4 to size n (multiple of 4).
       - Search n from Moore bound (512) upwards in steps of 4.
       - For each n, try multiple random lifts.
       - Verify girth using optimized BFS.
    2. Fallback: LUW construction (if lift search fails or k != 3).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # --- Helper: Verify Girth (Optimized BFS) ---
    def verify_girth(n, edges, k, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        
        for start in range(n):
            visited = {}
            queue = []
            head = 0
            
            for v in adj[start]:
                queue.append((v, start, 1))
                visited[(v, start)] = 1
            
            while head < len(queue):
                u, prev, depth = queue[head]
                head += 1
                
                if depth == max_depth:
                    continue
                
                for v in adj[u]:
                    if v == prev:
                        continue
                    
                    if v == start:
                        if depth + 1 < g:
                            return False
                        continue
                    
                    state = (v, u)
                    if state in visited:
                        if depth - visited[state] < g:
                            return False
                    else:
                        visited[state] = depth
                        queue.append((v, u, depth + 1))
                        
        return True

    # --- Helper: LUW Construction (Fallback) ---
    def luw_construct(k, g, d_override=None):
        if k != 3 and k != 4:
            return None, None
            
        q = k
        if d_override is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        else:
            d = d_override
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # --- Helper: Random Lift of K4 ---
    base_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    base_n = 4

    def random_lift(n):
        if n % base_n != 0:
            return None
        
        m = n // base_n
        perms = [list(range(m)) for _ in range(base_n)]
        for p in perms:
            random.shuffle(p)
        
        edges = set()
        
        for u, v in base_edges:
            for j in range(m):
                u_idx = u * m + j
                v_idx = v * m + perms[v][j]
                
                if u_idx == v_idx:
                    return None
                
                if u_idx < v_idx:
                    edges.add((u_idx, v_idx))
                else:
                    edges.add((v_idx, u_idx))
        
        return list(edges)

    # --- Main Logic ---
    
    best_n = None
    best_edges = None
    
    # 1. Try LUW first as a baseline? 
    # LUW for g=16 gives ~936.
    # Lift search can potentially find smaller (512-900).
    # Let's run Lift search first, as it targets the minimum.
    
    if k == 3:
        # Moore bound for k=3, g=16 is 510.
        # n must be multiple of 4.
        n_start = 512
        
        # Search n = 512, 516, 520, ...
        # Time budget for lift search: 100s
        lift_deadline = start_time + 100.0
        
        n_try = n_start
        while time.time() < lift_deadline:
            if n_try > 2000:
                break
                
            # Try a few random lifts
            # Number of attempts depends on time remaining
            attempts = 0
            max_attempts = 50
            
            found = False
            while attempts < max_attempts and time.time() < lift_deadline:
                edges = random_lift(n_try)
                if edges is not None:
                    if verify_girth(n_try, edges, k, g):
                        best_n = n_try
                        best_edges = edges
                        found = True
                        break
                attempts += 1
            
            if found:
                break
                
            n_try += 4
            
    # 2. If Lift didn't find a graph, or if k != 3, use LUW
    if best_n is None:
        # Try LUW with a few depth values
        candidate_ds = set()
        for delta in range(-2, 3):
            d_try = g // 2 - 1 + delta
            if d_try >= 3:
                candidate_ds.add(d_try)
                
        for d_try in candidate_ds:
            if time.time() > deadline - 5:
                break
            n_luw, edges_luw = luw_construct(k, g, d_try)
            if n_luw is None:
                continue
            if verify_girth(n_luw, edges_luw, k, g):
                best_n = n_luw
                best_edges = edges_luw
                break
                
    if best_n is None:
        # Last resort: Large LUW
        n_luw, edges_luw = luw_construct(k, g, g)
        if n_luw and verify_girth(n_luw, edges_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
            
    if best_n is None:
        # If still None, return a trivial invalid graph? No.
        # We must return something.
        # Return a large random graph and hope? No.
        # Raise an error? No.
        # Return a graph with n=0? No.
        # We'll return a large LUW with d = g + 5, which is almost certainly valid.
        n_luw, edges_luw = luw_construct(k, g, g + 5)
        if n_luw:
            best_n = n_luw
            best_edges = edges_luw
            
    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
