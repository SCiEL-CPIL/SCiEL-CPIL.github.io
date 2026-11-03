import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Try GP graph search
    # n is half the number of vertices
    for n in range(400, 1200):
        if time.time() > deadline:
            break
        for r in range(1, n // 2):
            if time.time() > deadline:
                break
            
            # Build adjacency
            adj = [[] for _ in range(2 * n)]
            for i in range(n):
                adj[i].append((i + 1) % n)
                adj[i].append((i - 1) % n)
                adj[i].append(n + i)
                adj[n + i].append(i)
                adj[n + i].append(n + (i + r) % n)
                adj[n + i].append(n + (i - r) % n)
                
            # Local check
            def local_check(start, depth_limit):
                visited = {start: 0}
                queue = [(start, 0, -1)]
                while queue:
                    node, dist, parent = queue.pop(0)
                    if dist >= depth_limit:
                        continue
                    for nei in adj[node]:
                        if nei == parent:
                            continue
                        if nei in visited:
                            return False
                        visited[nei] = dist + 1
                        queue.append((nei, dist + 1, node))
                return True

            if not local_check(0, g - 1):
                continue
            if not local_check(n, g - 1):
                continue
                
            # Exact check
            min_girth = float('inf')
            for start in range(2 * n):
                visited = [-1] * (2 * n)
                visited[start] = 0
                queue = [(start, -1)]
                while queue:
                    node, parent = queue.pop(0)
                    for nei in adj[node]:
                        if nei == parent:
                            continue
                        if visited[nei] == -1:
                            visited[nei] = visited[node] + 1
                            queue.append((nei, node))
                        else:
                            cycle_len = visited[node] + visited[nei] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                if min_girth < g:
                    break
                    
            if min_girth >= g:
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                    edges.append((i, n + i))
                    edges.append((n + i, n + (i + r) % n))
                return 2 * n, edges

    # Fallback to LUW
    # ...
