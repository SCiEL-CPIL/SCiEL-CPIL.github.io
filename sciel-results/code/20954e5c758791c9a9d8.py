import random
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def find_k4(adj, n):
        """Find a K4 in the graph. Returns (a,b,c,d) or None."""
        for a in range(n):
            for b in list(adj[a]):
                if b <= a: continue
                for c in list(adj[b]):
                    if c <= b: continue
                    if c not in adj[a]: continue
                    for d in list(adj[c]):
                        if d <= c: continue
                        if d in adj[a] and d in adj[b]:
                            return (a, b, c, d)
        return None
    
    def find_indep_6(adj, n):
        """Find an independent set of size 6. Returns set of 6 vertices or None."""
        # Use backtracking with bitmasks for speed
        # Represent adj as list of bitmasks
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        def backtrack(remaining_mask, chosen, depth):
            if depth == 6:
                return chosen
            if remaining_mask == 0:
                return None
            # Count bits in remaining_mask
            if bin(remaining_mask).count('1') < depth:
                return None
            
            # Pick a vertex from remaining
            # Try to pick one with fewest options (minimum degree in induced subgraph)
            best_v = -1
            best_count = n + 1
            temp = remaining_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                # Count non-neighbors of v in remaining
                non_neighbors = remaining_mask & ~(adj_mask[v] | (1 << v))
                cnt = bin(non_neighbors).count('1')
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    if cnt == 0:
                        break
            
            if best_v == -1:
                return None
            
            v = best_v
            # Option 1: Include v
            new_remaining = remaining_mask & ~(adj_mask[v] | (1 << v))
            result = backtrack(new_remaining, chosen | (1 << v), depth + 1)
            if result is not None:
                return result
            
            # Option 2: Exclude v
            return backtrack(remaining_mask & ~(1 << v), chosen, depth)
        
        result = backtrack((1 << n) - 1, 0, 0)
        if result is None:
            return None
        # Convert bitmask to list
        verts = []
        for i in range(n):
            if result & (1 << i):
                verts.append(i)
        return verts
    
    def count_k4s(adj, n):
        """Count number of K4s."""
        count = 0
        for a in range(n):
            for b in list(adj[a]):
                if b <= a: continue
                for c in list(adj[b]):
                    if c <= b: continue
                    if c not in adj[a]: continue
                    for d in list(adj[c]):
                        if d <= c: continue
                        if d in adj[a] and d in adj[b]:
                            count += 1
        return count
    
    def count_indep_6s(adj, n, max_count=100):
        """Count number of independent 6-sets (up to max_count)."""
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        count = 0
        def backtrack(remaining_mask, depth):
            nonlocal count
            if count >= max_count:
                return
            if depth == 6:
                count += 1
                return
            if remaining_mask == 0:
                return
            if bin(remaining_mask).count('1') < depth:
                return
            
            # Pick a vertex
            v = (remaining_mask & -remaining_mask).bit_length() - 1
            # Include v
            new_remaining = remaining_mask & ~(adj_mask[v] | (1 << v))
            backtrack(new_remaining, depth + 1)
            # Exclude v
            backtrack(remaining_mask & ~(1 << v), depth)
        
        backtrack((1 << n) - 1, 0)
        return count
    
    # Start with n=35
    n = 35
    
    # Initialize with a random graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)
    
    best_adj = None
    best_score = -1
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Check for K4
        k4 = find_k4(adj, n)
        if k4:
            # Remove one edge from the K4
            a, b, c, d = k4
            # Pick a random edge to remove
            edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            u, v = random.choice(edges_in_k4)
            adj[u].discard(v)
            adj[v].discard(u)
            continue
        
        # No K4, check for independent 6-set
        indep = find_indep_6(adj, n)
        if indep is None:
            # Found a valid graph!
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Has independent 6-set, add an edge to break it
        # Pick two vertices from the independent set that are not already connected
        # (they shouldn't be, since it's independent)
        u, v = random.sample(indep, 2)
        # But adding this edge might create a K4. Check first.
        # Common neighbors of u and v
        cn = adj[u].intersection(adj[v])
        # Check if any two vertices in cn are connected
        creates_k4 = False
        cn_list = list(cn)
        for i in range(len(cn_list)):
            for j in range(i+1, len(cn_list)):
                if cn_list[j] in adj[cn_list[i]]:
                    creates_k4 = True
                    break
            if creates_k4:
                break
        
        if not creates_k4:
            adj[u].add(v)
            adj[v].add(u)
        else:
            # Try a different pair or just skip
            # Try all pairs until we find one that doesn't create K4
            found = False
            for i in range(len(indep)):
                for j in range(i+1, len(indep)):
                    u2, v2 = indep[i], indep[j]
                    cn2 = adj[u2].intersection(adj[v2])
                    creates_k42 = False
                    cn2_list = list(cn2)
                    for x in range(len(cn2_list)):
                        for y in range(x+1, len(cn2_list)):
                            if cn2_list[y] in adj[cn2_list[x]]:
                                creates_k42 = True
                                break
                        if creates_k42:
                            break
                    if not creates_k42:
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        found = True
                        break
                if found:
                    break
    
    # If we didn't find a valid graph, try a different approach:
    # Use a known construction. The Paley graph of order 17 is K4-free and has no independent set of size 5,
    # but we need no independent set of size 6. Let's try a different fallback.
    
    # Fallback: return a smaller valid graph
    # Paley 17: vertices 0..16, edge (i,j) if (j-i) mod 17 is a quadratic residue mod 17
    squares = {x*x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i+1, 17):
            if (j-i) % 17 in squares:
                edges.append([i, j])
    return {'n': 17, 'edges': edges}
