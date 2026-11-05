import random
import time
import math

def has_k4(adj, n):
    """Check if graph contains K4. Returns True if K4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    # If w is connected to any other vertex in common, we have K4
                    if adj[w] & common:
                        return True
                    c ^= lsb
    return False

def has_independent_set_6(adj, n):
    """Check if graph contains independent set of size 6. Returns True if found."""
    # Build complement adjacency masks
    comp = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        # Complement of adj[i] excluding i itself
        comp[i] = (full_mask ^ (1 << i)) & ~adj[i]
        
    def dfs(candidates_mask, size):
        if size == 0:
            return True
        if candidates_mask == 0:
            return False
        if size == 1:
            return True
        # Pruning: not enough candidates
        if bin(candidates_mask).count('1') < size:
            return False
        
        # Pick a vertex (lowest set bit)
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Include v: new candidates must be neighbors of v in complement
        new_cands = candidates_mask & comp[v]
        if dfs(new_cands, size - 1):
            return True
        
        # Exclude v
        candidates_mask ^= lsb
        if dfs(candidates_mask, size):
            return True
        
        return False

    return dfs(full_mask, 6)

def count_k4s(adj, n):
    """Count the number of K4 subgraphs."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    # Check if w is connected to any other in common
                    # To count, we need to be careful not to double count
                    # But for penalty, approximate is fine
                    if adj[w] & common:
                        count += 1
                    c ^= lsb
    return count

def count_indep_6s(adj, n):
    """Count independent sets of size 6 (approximate or exact if small)."""
    # This is expensive. For penalty, we might just use 0/1 indicator.
    # Or we can count maximal independent sets?
    # Let's just use the boolean check for now, SA might struggle with binary landscape.
    # We will use a weighted penalty: 
    # If valid, 0.
    # If not, 1.
    # This is what the seed does effectively.
    pass

def construct():
    best_graph = None
    best_n = 0
    
    random.seed(42)
    start_time = time.time()
    time_limit = 110.0
    
    # We aim for n=35 primarily
    for n in [35, 34, 33, 32, 31]:
        if time.time() - start_time > time_limit:
            break
            
        # Simulated Annealing
        # Initial state: Random graph with p=0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Check initial
        if not has_k4(adj, n) and not has_independent_set_6(adj, n):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
        current_valid = False
        # We use a simple local search with restarts
        # But we improve the move: flip 1 or 2 edges
        
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Reset to random if stuck
            if restart > 0 and restart % 10 == 0:
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
            # Local search loop
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                    
                if not has_k4(adj, n) and not has_independent_set_6(adj, n):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # Move: Flip one edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
    # Fallback
    return {'n': 1, 'edges': []}
