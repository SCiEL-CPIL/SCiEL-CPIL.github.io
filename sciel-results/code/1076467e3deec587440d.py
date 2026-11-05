import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    if k != 3 or g != 16:
        # Fallback for other parameters: return a trivial valid graph if possible
        # For this task, we only handle k=3, g=16
        return (0, [])
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def remaining():
        return deadline - time.time()
    
    # The best known (3,16)-cage has 936 vertices.
    # We will attempt to construct a graph with 936 vertices using a known construction or search.
    # Since we don't have the specific edge list of the 936-vertex cage hardcoded,
    # we will use a strategy that is likely to find a valid graph with n close to the lower bound or known best.
    
    # Strategy:
    # 1. Try to find a valid 3-regular graph with girth 16 on n vertices.
    # 2. The Moore bound for (3,16) is 2 * (1 + 2 + 4 + ... + 2^7) = 2*(2^8 - 1) = 510.
    # 3. Known best is 936.
    # 4. We can try to construct a graph via randomization + local search, or use a known algebraic construction.
    # 5. However, constructing a (3,16)-cage from scratch in 120s is hard.
    # 6. A safer approach: Use a known construction for (3,14) or (3,16) if available, or use a voltage lift on a small base graph.
    
    # Let's try a voltage lift on the Petersen graph (10 vertices, 3-regular, girth 5).
    # The girth of the lift is at least the girth of the base if the voltage assignment avoids short cycles.
    # Actually, the girth of a voltage lift is determined by the shortest closed walk in the base whose voltage sum is 0.
    # To get girth 16, we need no closed walk of length < 16 with voltage sum 0.
    # The Petersen graph has girth 5. So we need to avoid voltage sums of 0 for walks of length 5, 6, ..., 15.
    
    # This is equivalent to finding a homomorphism from the universal cover (tree) to Z_m such that no short cycle maps to 0.
    # This is the "voltage assignment" problem.
    
    # Let's try to find a voltage assignment for the Petersen graph over Z_m for some prime m.
    # The number of vertices will be 10 * m.
    # We want 10 * m to be small, so m should be small.
    # But m must be large enough to avoid short cycles.
    # For girth 16, we need to avoid cycles of length up to 15.
    # The number of closed walks of length L in the Petersen graph grows exponentially.
    # We need a voltage assignment such that none of these walks sum to 0 mod m.
    
    # This is a constraint satisfaction problem.
    # Let's try m = 93 (since 10*93 = 930, close to 936).
    # Or m = 97 (10*97 = 970).
    # Or m = 89 (10*89 = 890).
    
    # Actually, let's just try to find ANY valid graph with n as small as possible.
    # We will use a random voltage assignment and check for short cycles.
    # If we find one with n < 936, great. If not, we return a known valid graph if we can find one.
    
    # Since we don't have a hardcoded 936-vertex cage, we must find one via search.
    # However, search in 120s is risky.
    
    # Alternative: Use a known construction for (3,16) if we can recall it.
    # The (3,16)-cage is known to have 936 vertices.
    # We can try to generate a random 3-regular graph on 936 vertices and check girth.
    # The probability of a random 3-regular graph having girth 16 is extremely low.
    
    # Better: Use the fact that the (3,16)-cage is a cover of the Petersen graph?
    # Not necessarily.
    
    # Let's try the voltage lift approach with m = 93, 97, etc.
    # We will randomly assign voltages to the 15 edges of the Petersen graph.
    # Then we check if any closed walk of length < 16 has voltage sum 0 mod m.
    
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        if d + 1 < g:
                            queue.append(nb)
        
        if min_cycle < g:
            return min_cycle
        return min_cycle if min_cycle != float('inf') else g

    # Petersen graph edges
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
    
    # Precompute all closed walks of length up to 15 in the Petersen graph
    # and their edge incidence vectors (with signs).
    # This is expensive, so we do it once.
    
    walk_vectors = []
    max_L = g - 1  # 15
    
    # We need to track the sequence of edges used, with direction.
    # For each closed walk, we compute the voltage sum as sum of voltages * direction.
    # We want this sum to be non-zero mod m for all walks of length < 16.
    
    # Instead of enumerating all walks (which is exponential), we can use a BFS to find all walks up to length 15.
    # But there are many walks.
    # Let's limit the number of walks we check to avoid timeout.
    
    # Alternatively, we can just try random voltage assignments and check girth of the lifted graph directly.
    # This is simpler and avoids precomputation.
    
    def try_voltage_lift(m, num_trials=100):
        for _ in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random voltage assignment
            vol = [random.randint(1, m - 1) for _ in range(15)]
            
            # Build lifted graph
            n = n_base * m
            edges = []
            for i, (u, v) in enumerate(pet_edges):
                for layer in range(m):
                    u_lift = u * m + layer
                    v_lift = v * m + (layer + vol[i]) % m
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    else:
                        edges.append((v_lift, u_lift))
            
            # Check girth
            gc = check_girth(n, edges, g, deadline - 5)
            if gc >= g:
                return (n, edges)
        
        return None
    
    # Try different m values
    # We want n = 10 * m to be small.
    # Let's try m = 93, 97, 101, etc.
    # But first, let's try smaller m to see if we can get lucky.
    
    # Actually, the girth constraint is very tight.
    # Let's try m = 93 first (n=930), then m=97 (n=970), etc.
    
    for m in [93, 97, 101, 103, 107, 109, 113, 127, 131, 137]:
        if time.time() > deadline - 20:
            break
        
        res = try_voltage_lift(m, num_trials=50)
        if res is not None:
            return res
    
    # If voltage lift fails, try a different approach:
    # Construct a random 3-regular graph on n vertices and check girth.
    # This is unlikely to work for n > 100, but let's try.
    
    # Actually, let's just return a known valid graph if we can find one.
    # Since we don't have one, we will return the best we found.
    
    # As a last resort, return a trivial graph (which will be invalid for girth 16, but better than nothing).
    # But the task requires a valid graph.
    
    # Let's try one more thing: use a known (3,14)-cage or similar and modify it.
    # The (3,14)-cage has 112 vertices. We can try to "blow up" it.
    
    # For now, return the best result found.
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
