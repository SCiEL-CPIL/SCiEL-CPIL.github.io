import random
import time
import math

def construct(k, g):
    # Target: 4-regular graph with girth >= 10
    # Try n=256 first (Morgenstern size), then 320.
    # Use Simulated Annealing to minimize short cycles.
    
    def generate_random_regular(n, k):
        # Configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        edges = []
        attempts = 0
        while True:
            random.shuffle(half_edges)
            edges = []
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u != v:
                    edges.append((u, v))
            
            # Check for multi-edges
            edge_set = set()
            valid = True
            for u, v in edges:
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
            
            if valid:
                return edges
            
            attempts += 1
            if attempts > 1000:
                # Fallback: simple ring
                edges = [(i, (i+1)%n) for i in range(n)] + [(i, (i+2)%n) for i in range(n)]
                # This is not k-regular for k=4?
                # Degree 4: connect i to i+1, i-1, i+2, i-2
                edges = []
                for i in range(n):
                    edges.append((i, (i+1)%n))
                    edges.append((i, (i-1)%n))
                    edges.append((i, (i+2)%n))
                    edges.append((i, (i-2)%n))
                # Remove duplicates
                unique_edges = set()
                for u, v in edges:
                    if u < v:
                        unique_edges.add((u, v))
                    else:
                        unique_edges.add((v, u))
                return list(unique_edges)

    def count_short_cycles(n, edges, max_girth):
        # Count cycles of length < max_girth
        # Use BFS from each vertex
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        count = 0
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            queue = [(start, 0)]
            dist[start] = 0
            while queue:
                u, d = queue.pop(0)
                if d >= max_girth - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        queue.append((v, d + 1))
                    elif v != parent[u] and dist[v] < d:
                        # Found a cycle
                        # Length is d + 1 + dist[v]?
                        # Actually, just count 1 for now
                        count += 1
        return count

    def swap_edges(n, edges, adj):
        # Pick two random edges and swap
        # Ensure no multi-edges or loops
        for _ in range(100):
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx:
                continue
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Try swap 1: (u1, u2) and (v1, v2)
            if u1 != u2 and v1 != v2 and (u1, u2) not in adj[u1] and (v1, v2) not in adj[v1]:
                # Check if edges exist
                # Simple check
                if u1 not in adj[u2] and v1 not in adj[v2]:
                    # Perform swap
                    edges[e1_idx] = (u1, u2)
                    edges[e2_idx] = (v1, v2)
                    # Update adj
                    adj[u1].remove(v1)
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[u2].remove(v2)
                    adj[u2].append(v1)
                    adj[v1].remove(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)
                    adj[v2].remove(u2)
                    adj[v2].append(v1) # Wait, v2 connects to v1? No, v2 connects to v1 in new edge?
                    # Edge is (v1, v2). So v1-v2.
                    # Old: u1-v1, u2-v2.
                    # New: u1-u2, v1-v2.
                    # Remove u1-v1: adj[u1]-v1, adj[v1]-u1.
                    # Remove u2-v2: adj[u2]-v2, adj[v2]-u2.
                    # Add u1-u2: adj[u1]+u2, adj[u2]+u1.
                    # Add v1-v2: adj[v1]+v2, adj[v2]+v1.
                    
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)
                    
                    return True
        return False

    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=256
    n = 256
    edges = generate_random_regular(n, k)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Initial cost
    cost = count_short_cycles(n, edges, g)
    
    T = 100.0
    cooling = 0.99
    
    while time.time() < deadline:
        if cost == 0:
            break
        
        # Propose move
        new_edges = edges[:]
        new_adj = [lst[:] for lst in adj]
        
        if swap_edges(n, new_edges, new_adj):
            new_cost = count_short_cycles(n, new_edges, g)
            delta = new_cost - cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                edges = new_edges
                adj = new_adj
                cost = new_cost
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    # If n=256 failed, try n=320
    if cost > 0 and time.time() < deadline - 30:
        n = 320
        edges = generate_random_regular(n, k)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cost = count_short_cycles(n, edges, g)
        
        T = 100.0
        cooling = 0.99
        while time.time() < deadline:
            if cost == 0:
                break
            new_edges = edges[:]
            new_adj = [lst[:] for lst in adj]
            if swap_edges(n, new_edges, new_adj):
                new_cost = count_short_cycles(n, new_edges, g)
                delta = new_cost - cost
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    edges = new_edges
                    adj = new_adj
                    cost = new_cost
            T *= cooling
            if T < 0.1:
                T = 0.1

    # Clean up edges (remove duplicates if any, though swap should prevent it)
    unique_edges = set()
    for u, v in edges:
        if u < v:
            unique_edges.add((u, v))
        else:
            unique_edges.add((v, u))
            
    return n, list(unique_edges)
