import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 12
    
    # The optimal known solution for this constraint set (min degree 3, girth >= 4, 12 vertices)
    # minimizes distinct cycle lengths to 2 (lengths 4 and 6).
    # This is achieved by two disjoint K_{3,3} graphs.
    # K_{3,3} is bipartite, so all cycles are even.
    # Min cycle length is 4.
    # Max cycle length in K_{3,3} is 6 (since each partition has size 3).
    # Thus, cycle lengths are exactly {4, 6}.
    # Two disjoint copies do not add new cycle lengths.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([int(u), int(v)])
            
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([int(u), int(v)])
            
    # Verify constraints internally to ensure validity
    # 1. No loops
    for e in edges:
        if e[0] == e[1]:
            raise ValueError("Loop detected")
    # 2. No duplicates
    edge_set = set()
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key in edge_set:
            raise ValueError("Duplicate edge")
        edge_set.add(key)
    # 3. Min degree >= 3
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    if min(degrees) < 3:
        raise ValueError("Min degree < 3")
    # 4. Triangle free (Girth >= 4)
    # For K_{3,3}, this is guaranteed.
    # We can skip explicit triangle check for performance if we trust the construction,
    # but a quick check is safer.
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        for j in adj[i]:
            if i < j:
                # Check if i and j have a common neighbor
                if adj[i].intersection(adj[j]):
                    # Triangle found
                    raise ValueError("Triangle detected")

    return edges
