import random
import time

def construct():
    n = 16
    # Represent graph as a list of bitmasks for adjacency
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    adj = [0] * n
    
    # Helper to check if adding edge (u, v) creates a C4
    def creates_c4(u, v, adj):
        # Check if there is a path u-w-x-v in the current graph
        # w is neighbor of u
        # x is neighbor of v
        # w and x are neighbors
        # Iterate over neighbors of u
        w = adj[u]
        while w:
            # Get lowest set bit
            lsb = w & -w
            w -= lsb
            wi = lsb.bit_length() - 1
            # Check if any neighbor of wi is a neighbor of v
            # common = adj[wi] & adj[v]
            if adj[wi] & adj[v]:
                return True
        return False

    # Helper to get number of edges
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    best_edges = 0
    best_adj = None
    
    deadline = time.time() + 110  # Leave 10s margin
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Try to start dense
        current_adj = [0] * n
        # Add random edges
        # We can start with a random graph with p=0.3
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
        
        # Remove C4s
        # Simple loop: find a C4, remove an edge
        # Repeat until no C4
        # To speed up, we can just check all pairs?
        # Since n is small, we can just iteratively fix
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if (current_adj[i] >> j) & 1:
                        # Check if this edge is part of a C4
                        # If we remove (i,j), does it break all C4s containing it?
                        # Actually, just check if current graph has C4.
                        # If yes, remove a random edge from it.
                        pass
            
            # Better: Find any C4
            # A C4 is i-j-k-l-i
            # Check all quadruples? 16C4 = 1820. Very fast.
            c4_found = None
            for i in range(n):
                for j in range(i+1, n):
                    if not ((current_adj[i] >> j) & 1):
                        continue
                    for k in range(j+1, n):
                        if not ((current_adj[j] >> k) & 1):
                            continue
                        # Need l such that k-l and l-i are edges
                        # l != i, j, k
                        candidates = current_adj[k] & current_adj[i]
                        # Remove i and k from candidates?
                        # l must be neighbor of k and i
                        # l != k, l != i
                        candidates &= ~(1 << k)
                        candidates &= ~(1 << i)
                        if candidates:
                            lsb = candidates & -candidates
                            l = lsb.bit_length() - 1
                            # Check if l-j is edge? No, the cycle is i-j-k-l-i
                            # We checked i-j, j-k.
                            # We need k-l and l-i.
                            # l is neighbor of k and i.
                            # So k-l and l-i are edges.
                            # So i-j-k-l-i is a C4.
                            c4_found = (i, j, k, l)
                            break
                        if c4_found:
                            break
                    if c4_found:
                        break
                if c4_found:
                    break
            if c4_found:
                # Remove one edge from the C4
                i, j, k, l = c4_found
                # Randomly pick one of the 4 edges to remove
                edges = [(i,j), (j,k), (k,l), (l,i)]
                u, v = random.choice(edges)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                changed = True
        
        # Now current_adj is C4-free
        cur_edges = count_edges(current_adj)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = current_adj[:]
        
        # Try to improve by adding edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order of missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if not ((current_adj[i] >> j) & 1):
                        missing.append((i,j))
            random.shuffle(missing)
            
            for u, v in missing:
                if not creates_c4(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    cur_edges += 1
                    improved = True
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_adj = current_adj[:]
        
        # Local search: swap edges
        # Remove one edge, try to add another
        # This might help escape local optima
        if time.time() < deadline:
             for _ in range(100):
                 if time.time() > deadline:
                     break
                 # Pick a random edge to remove
                 all_edges = []
                 for i in range(n):
                     for j in range(i+1, n):
                         if (current_adj[i] >> j) & 1:
                             all_edges.append((i,j))
                 if not all_edges:
                     break
                 u, v = random.choice(all_edges)
                 current_adj[u] &= ~(1 << v)
                 current_adj[v] &= ~(1 << u)
                 cur_edges -= 1
                 
                 # Try to add a random missing edge
                 missing = []
                 for i in range(n):
                     for j in range(i+1, n):
                         if not ((current_adj[i] >> j) & 1):
                             missing.append((i,j))
                 random.shuffle(missing)
                 
                 added = False
                 for a, b in missing:
                     if not creates_c4(a, b, current_adj):
                         current_adj[a] |= (1 << b)
                         current_adj[b] |= (1 << a)
                         cur_edges += 1
                         added = True
                         if cur_edges > best_edges:
                             best_edges = cur_edges
                             best_adj = current_adj[:]
                         break
                 
                 if not added:
                     # Restore removed edge
                     current_adj[u] |= (1 << v)
                     current_adj[v] |= (1 << u)
                     cur_edges += 1

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
