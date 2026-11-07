import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    edges = []
    # First K3,3 on vertices 0,1,2 and 3,4,5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K3,3 on vertices 6,7,8 and 9,10,11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Verify all vertices have degree >= 3
    from collections import defaultdict
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    # Check: all 12 vertices should have degree >= 3
    for i in range(12):
        if degree[i] < 3:
            # Fallback: this shouldn't happen with K3,3 components
            pass
    
    return edges
