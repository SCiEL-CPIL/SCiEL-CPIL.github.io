import time

def construct():
    start_time = time.time()
    time_limit = 115.0

    def solve(n, budget):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        # adj[c][u] is a bitmask of neighbors
        adj = [ [0] * n for _ in range(4) ]
        colors = [-1] * len(edges)
        
        def is_safe(c, u, v):
            # Check if path of length 5 exists
            # BFS using bitmasks
            from collections import deque
            q = deque([(u, 0)])
            visited = 1 << u
            while q:
                node, depth = q.popleft()
                if depth == 5:
                    if node == v:
                        return False
                    continue
                neighbors = adj[c][node]
                # Exclude u
                neighbors &= ~(1 << u)
                while neighbors:
                    lsb = neighbors & -neighbors
                    neighbor = lsb.bit_length() - 1
                    neighbors ^= lsb
                    if neighbor == v and depth == 4:
                        return False
                    if not (visited & (1 << neighbor)):
                        visited |= (1 << neighbor)
                        q.append((neighbor, depth + 1))
            return True

        def backtrack(idx):
            if time.time() - start_time > budget:
                return False
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            if idx == 0:
                candidates = [0]
            else:
                candidates = [0, 1, 2, 3]
                
            for c in candidates:
                if is_safe(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            return False

        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    res = solve(18, 60.0)
    if res:
        return res

    res = solve(17, 50.0)
    if res:
        return res

    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
