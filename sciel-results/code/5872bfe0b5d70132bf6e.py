import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer for safety, limit is 150s
    
    # Try to find a graph for n from 35 down to 1
    for n in range(35, 0, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for this n
        found = False
        attempts = 0
        max_attempts = 200  # Increase attempts
        
        while not found and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            adj = [0] * n
            # Initialize with random edges, bias towards density that might work
            # R(4,6) graphs are often around density 0.5
            p = 0.5
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local search
            valid = False
            for step in range(5000):
                if time.time() > deadline - 10:
                    break
                
                # Check K4
                k4 = find_k4(adj, n)
                if k4:
                    # Remove a random edge from the K4
                    i, j, k, l = k4
                    edges = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                    u, v = random.choice(edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    continue
                
                # Check Independent 6 (K6 in complement)
                # First, a fast randomized check
                ind6_found = False
                for _ in range(50):
                    subset = random.sample(range(n), 6)
                    if is_independent(subset, adj):
                        ind6_found = True
                        # Add a random edge within this independent set
                        u, v = random.sample(subset, 2)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        break
                
                if ind6_found:
                    continue
                
                # If randomized check passes, do exact check
                # This is expensive, so only do it when we think we are close
                if has_k6_exact(adj, n):
                    # Find a K6 to fix
                    # We can reuse the search to return one
                    k6 = find_k6_exact(adj, n)
                    if k6:
                        u, v = random.sample(k6, 2)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                else:
                    # No K4, no Ind6. Valid!
                    valid = True
                    break
            
            if valid:
                # Final verification
                if is_valid(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback: Return a known small valid graph
    # Paley 17 might have alpha 6?
    # Let's use a very small graph to be safe, e.g., n=1
    # Or try to construct a small valid one quickly.
    # For n=1, valid.
    return {'n': 1, 'edges': []}

def find_k4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                cn = adj[i] & adj[j]
                temp = cn
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    temp ^= lsb
                    if cn & adj[k]:
                        l_mask = cn & adj[k]
                        if l_mask:
                            l = (l_mask & -l_mask).bit_length() - 1
                            return (i, j, k, l)
    return None

def is_independent(subset, adj):
    # Check if all pairs in subset are non-adjacent
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            u, v = subset[i], subset[j]
            if adj[u] & (1 << v):
                return False
    return True

def has_k6_exact(adj, n):
    # Check if there is a K6 in the complement
    # Equivalent to Independent Set of size 6 in G
    full_mask = (1 << n) - 1
    cadj = [0] * n
    for i in range(n):
        cadj[i] = (full_mask ^ (1 << i)) & ~adj[i]
    
    def search(candidates_mask, depth):
        if depth == 6:
            return True
        if candidates_mask == 0:
            return False
        
        # Pruning: if remaining vertices < needed
        if candidates_mask.bit_count() < 6 - depth:
            return False
            
        mask = candidates_mask
        while mask:
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in the induced subgraph?
            # For speed, just pick the first one.
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            
            new_mask = cadj[v] & mask
            if new_mask.bit_count() < 6 - depth - 1:
                continue
                
            if search(new_mask, depth + 1):
                return True
                
        return False
        
    return search(full_mask, 0)

def find_k6_exact(adj, n):
    # Similar to has_k6_exact but returns the vertices
    full_mask = (1 << n) - 1
    cadj = [0] * n
    for i in range(n):
        cadj[i] = (full_mask ^ (1 << i)) & ~adj[i]
        
    result = []
    
    def search(candidates_mask, depth, path):
        if depth == 6:
            result.extend(path)
            return True
        if candidates_mask == 0:
            return False
            
        if candidates_mask.bit_count() < 6 - depth:
            return False
            
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            
            new_mask = cadj[v] & mask
            if new_mask.bit_count() < 6 - depth - 1:
                continue
                
            path.append(v)
            if search(new_mask, depth + 1, path):
                return True
            path.pop()
            
        return False
        
    if search(full_mask, 0, []):
        return result
    return None

def is_valid(adj, n):
    if find_k4(adj, n):
        return False
    if has_k6_exact(adj, n):
        return False
    return True
