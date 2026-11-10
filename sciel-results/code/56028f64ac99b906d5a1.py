import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency sets
    adj = [set() for _ in range(m)]
    for i in range(m):
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            adj[i].add(j1)
            adj[i].add(j2)
            
    # Structural solution: Take all x where x % 5 in {0, 2}
    S = []
    for i in range(m):
        if i % 5 == 0 or i % 5 == 2:
            S.append(i)
            
    # Check for any remaining vertices that can be added
    # A vertex v can be added if it has no neighbor in S
    V_rem = []
    for v in range(m):
        if v in S:
            continue
        # Check if v is adjacent to any node in S
        # We can optimize this by precomputing the union of neighbors of S
        # But S is small (82), so checking each v is fast.
        has_neighbor = False
        for u in S:
            if u in adj[v]:
                has_neighbor = True
                break
        if not has_neighbor:
            V_rem.append(v)
            
    # If V_rem is not empty, find a large independent set in G[V_rem]
    if V_rem:
        # Greedy independent set on V_rem
        # Since V_rem is small, we can just pick vertices greedily
        # Sort by degree in V_rem?
        rem_set = set(V_rem)
        # Build degree map within V_rem
        deg = {v: 0 for v in V_rem}
        for v in V_rem:
            for u in adj[v]:
                if u in rem_set:
                    deg[v] += 1
        
        # Greedy: pick vertex with lowest degree? Or highest?
        # For IS, picking low degree is often better to keep options open.
        # But picking high degree removes many conflicts.
        # Let's just pick arbitrary or lowest degree.
        available = set(V_rem)
        while available:
            # Pick vertex with lowest degree in available
            v = min(available, key=lambda x: deg[x])
            S.append(v)
            # Remove v and its neighbors from available
            available.remove(v)
            for u in adj[v]:
                if u in available:
                    available.remove(u)
                    # Update degrees? Not necessary for simple greedy.
                    
    return np.array(S, dtype=np.int64)
