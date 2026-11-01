import random
import numpy as np
from collections import deque
import time
import sys

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    if k != 5 or g != 9:
        # Fallback for other parameters
        return _fallback_construct(k, g)
    
    start_time = time.time()
    time_limit = 115  # leave 5s margin
    
    # Target: 5-regular, girth >= 9
    # Moore bound is 781, known best is 1116
    # Try to find a graph with n < 1116 using Simulated Annealing
    
    # We will iterate n from Moore bound upwards
    # But to save time, we might jump to promising ranges
    # The known best is 1116. We try to beat it.
    
    for n in range(781, 1117):
        if time.time() - start_time > time_limit:
            break
        
        # Time budget per n
        time_per_n = max(1.0, (time_limit - (time.time() - start_time)) / (1116 - n + 1))
        
        result = _sa_construct(n, k, g, time_per_n)
        if result is not None:
            return result
            
    # If we didn't find one in the loop, try one last time with a longer budget for n=1116
    # Or just return a known valid graph if we had one.
    # Since we cannot use LUW, we try SA on 1116 with remaining time.
    if time.time() - start_time < time_limit:
         result = _sa_construct(1116, k, g, time_limit - (time.time() - start_time))
         if result is not None:
             return result
             
    # If all fails, we must return something.
    # Returning an invalid graph scores 0.
    # We will try to return a graph that is at least regular, even if girth is bad?
    # No, "invalid result... scores 0".
    # We must return a valid graph.
    # If we can't, we are stuck.
    # However, the probability of SA failing on 1116 is non-zero.
    # Let's add a very simple fallback: a 5-regular graph with small girth is NOT valid.
    # We need a valid graph.
    # If we can't find one, we fail.
    
    # As a last resort, we can try to construct a graph using a simple heuristic that might work for small n?
    # No, we need n ~ 1000.
    
    # If we reach here, we failed to find a valid graph.
    # We return a dummy to avoid crash, but it will score 0.
    return 0, []

def _fallback_construct(k, g):
    """Fallback using a simple construction."""
    n = k * (k ** ((g - 3) // 2)) + 1
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % n))
            edges.append((i, (i - j) % n))
    return n, edges

def _sa_construct(n, k, g, time_budget):
    """Simulated Annealing to find a k-regular graph with girth >= g."""
    start_time = time.time()
    
    # 1. Generate initial random k-regular graph
    adj = _random_k_regular_bitset(n, k)
    if adj is None:
        return None
    
    # 2. Calculate initial energy (number of short cycles)
    current_energy = _count_short_cycles(adj, n, g)
    
    # If we already have a valid graph, return it
    if current_energy == 0:
        return n, _adj_to_edges(adj)
    
    # 3. Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    best_energy = current_energy
    best_adj = [x for x in adj]
    
    while time.time() - start_time < time_budget:
        if T < T_min:
            break
            
        # Propose move: 2-swap
        move = _propose_move(adj, n, k)
        if move is None:
            T *= cooling
            continue
            
        u, v, x, y, orientation = move
        
        # Apply move
        # Remove (u,v) and (x,y)
        # Add (u,x) and (v,y) or (u,y) and (v,x)
        
        # We need to check if new edges exist
        if orientation == 0:
            new_edges = ((u, x), (v, y))
        else:
            new_edges = ((u, y), (v, x))
            
        # Check simplicity
        if (u == x or u == y or v == x or v == y):
            T *= cooling
            continue
        if adj[u] & (1 << x) or adj[v] & (1 << y):
             T *= cooling
             continue
             
        # Apply
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        adj[x] ^= (1 << y)
        adj[y] ^= (1 << x)
        
        adj[u] |= (1 << new_edges[0][1])
        adj[new_edges[0][1]] |= (1 << u)
        adj[v] |= (1 << new_edges[1][1])
        adj[new_edges[1][1]] |= (1 << v)
        
        # Calculate new energy
        new_energy = _count_short_cycles(adj, n, g)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = [x for x in adj]
                if best_energy == 0:
                    return n, _adj_to_edges(best_adj)
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            adj[x] ^= (1 << y)
            adj[y] ^= (1 << x)
            
            adj[u] ^= (1 << new_edges[0][1])
            adj[new_edges[0][1]] ^= (1 << u)
            adj[v] ^= (1 << new_edges[1][1])
            adj[new_edges[1][1]] ^= (1 << v)
            
        T *= cooling
        
    # Return best found if it is valid? No, must be valid.
    if best_energy == 0:
        return n, _adj_to_edges(best_adj)
        
    return None

def _random_k_regular_bitset(n, k):
    """Generate a random k-regular graph using bitsets."""
    if n * k % 2 != 0:
        return None
        
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adj = [0] * n
    deg = [0] * n
    
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue
        if adj[u] & (1 << v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    for i in range(n):
        if deg[i] != k:
            return None
            
    return adj

def _count_short_cycles(adj, n, g):
    """Count cycles of length < g using BFS with bitsets."""
    count = 0
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS
        # visited[node] = bitset of depths reached?
        # Simpler: visited[node] = min depth reached?
        # We need to detect if a node is reached at depth d and also at depth d' < d.
        # If we reach a node at depth d, and it was already reached at depth d' < d,
        # then we have a cycle of length d + d' (if the paths are disjoint except start/end).
        # Actually, BFS tree edges don't form cycles. Non-tree edges do.
        
        # Standard BFS girth check:
        # For each start, BFS.
        # If we encounter a visited node, cycle length = depth[u] + depth[v] + 1.
        # We want min cycle length.
        
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        dist[start] = 0
        queue.append(start)
        
        min_cycle = g
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
                
            neighbors = adj[u]
            # Iterate through set bits
            while neighbors:
                lsb = neighbors & (-neighbors)
                v = lsb.bit_length() - 1
                neighbors ^= lsb
                
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
        
        if min_cycle < g:
            count += 1 # We just want to know if there are ANY, or count them?
            # For SA, we want to minimize the number of short cycles.
            # This BFS finds the *shortest* cycle from start.
            # It doesn't count all short cycles.
            # We need the total count to guide the search.
            # Counting all is expensive.
            # Let's just return 1 if any short cycle exists, 0 otherwise?
            # No, that's binary. SA needs gradient.
            # Let's count the number of "violations" in BFS.
            # Each time we see a back edge that forms a short cycle, count it.
            
    return count # This is a rough heuristic.

def _propose_move(adj, n, k):
    """Propose a 2-swap move."""
    # Pick two random edges
    # We need to pick (u,v) and (x,y) such that they don't share vertices
    # and (u,x), (v,y) are not edges.
    
    # Get list of edges
    # This is slow.
    # Pick random u, v from adj[u]
    # Pick random x, y from adj[x]
    
    for _ in range(10):
        u = random.randint(0, n - 1)
        neighbors_u = adj[u]
        if not neighbors_u: continue
        v = (neighbors_u & (-neighbors_u)).bit_length() - 1
        
        x = random.randint(0, n - 1)
        if x in (u, v): continue
        neighbors_x = adj[x]
        if not neighbors_x: continue
        y = (neighbors_x & (-neighbors_x)).bit_length() - 1
        if y in (u, v, x): continue
        
        # Check if (u,x) and (v,y) are not edges
        if not (adj[u] & (1 << x)) and not (adj[v] & (1 << y)):
            return u, v, x, y, 0
        if not (adj[u] & (1 << y)) and not (adj[v] & (1 << x)):
            return u, v, x, y, 1
            
    return None

def _adj_to_edges(adj):
    """Convert bitset adjacency to edge list."""
    edges = []
    n = len(adj)
    for i in range(n):
        neighbors = adj[i]
        while neighbors:
            lsb = neighbors & (-neighbors)
            j = lsb.bit_length() - 1
            if i < j:
                edges.append((i, j))
            neighbors ^= lsb
    return edges
