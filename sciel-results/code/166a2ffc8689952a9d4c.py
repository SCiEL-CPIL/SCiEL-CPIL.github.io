import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def has_k4(adj, n):
        """
        Check for K4 using bitmasks.
        adj is a list of integers (bitmasks) of length n.
        Returns True if K4 exists, else False.
        """
        for u in range(n):
            au = adj[u]
            # Only consider neighbors v > u to avoid duplicates
            # But we need to check edges (u,v).
            # Iterate v
            v = u + 1
            while v < n:
                # Find next set bit in au from v onwards
                # Mask out bits <= v
                mask = au & ((1 << (n - v)) - 1)
                if mask == 0:
                    break
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if adj[u] & (1 << v):
                    common = au & adj[v]
                    # Check if common has an edge
                    # Iterate bits in common
                    c = common
                    while c:
                        lsb_c = c & -c
                        i = lsb_c.bit_length() - 1
                        c ^= lsb_c
                        if adj[i] & c:
                            return True
        return False

    def get_k4(adj, n):
        """
        Returns a tuple of 4 vertices forming a K4, or None.
        """
        for u in range(n):
            au = adj[u]
            v = u + 1
            while v < n:
                mask = au & ((1 << (n - v)) - 1)
                if mask == 0:
                    break
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if adj[u] & (1 << v):
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb_c = c & -c
                        i = lsb_c.bit_length() - 1
                        c ^= lsb_c
                        if adj[i] & c:
                            # Found K4: u, v, i, and some j in c adjacent to i
                            # Find j
                            cj = adj[i] & c
                            if cj:
                                lsb_j = cj & -cj
                                j = lsb_j.bit_length() - 1
                                return (u, v, i, j)
        return None

    def has_indep_6(adj, n):
        """
        Check for independent set of size 6 using backtracking with bitmasks.
        Returns True if found, else False.
        """
        # Pre-compute non-adjacency masks? 
        # non_adj[v] = ~adj[v] & ((1<<n)-1)
        # But we need to be careful with the 'chosen' logic.
        
        def backtrack(remaining, chosen_count):
            if chosen_count == 6:
                return True
            if remaining.bit_count() < 6 - chosen_count:
                return False
            
            # Heuristic: pick vertex with max degree in remaining
            # to prune the 'include' branch
            max_deg = -1
            best_v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Degree in remaining is count of neighbors of v in remaining
                deg = (adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            v = best_v
            bit_v = 1 << v
            
            # Option 1: Include v
            # New remaining: remove v and all its neighbors
            new_rem_include = remaining & ~(adj[v] | bit_v)
            if backtrack(new_rem_include, chosen_count + 1):
                return True
            
            # Option 2: Exclude v
            new_rem_exclude = remaining & ~bit_v
            if backtrack(new_rem_exclude, chosen_count):
                return True
                
            return False

        return backtrack((1 << n) - 1, 0)

    def get_indep_6(adj, n):
        """
        Returns a list of 6 vertices forming an independent set, or None.
        """
        path = []
        
        def backtrack(remaining, chosen):
            if len(chosen) == 6:
                path.extend(chosen)
                return True
            if remaining.bit_count() < 6 - len(chosen):
                return False
            
            # Heuristic: pick vertex with max degree in remaining
            max_deg = -1
            best_v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            v = best_v
            bit_v = 1 << v
            
            # Try include
            new_rem_include = remaining & ~(adj[v] | bit_v)
            if backtrack(new_rem_include, chosen + [v]):
                return True
                
            # Try exclude
            new_rem_exclude = remaining & ~bit_v
            if backtrack(new_rem_exclude, chosen):
                return True
                
            return False

        if backtrack((1 << n) - 1, []):
            return path
        return None

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def make_adj_from_edges(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # --- Search for n=36 ---
    # Try a few random restarts for n=36
    found_36 = False
    best_adj_36 = None
    
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        n = 36
        # Random graph with p=0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair loop
        for step in range(5000):
            if time.time() > deadline - 10:
                break
                
            k4 = get_k4(adj, n)
            if k4:
                u, v, i, j = k4
                # Remove a random edge from the K4
                # Edges: (u,v), (u,i), (u,j), (v,i), (v,j), (i,j)
                # Pick one randomly
                pairs = [(u,v), (u,i), (u,j), (v,i), (v,j), (i,j)]
                pu, pv = random.choice(pairs)
                adj[pu] &= ~(1 << pv)
                adj[pv] &= ~(1 << pu)
                continue
            
            indep = get_indep_6(adj, n)
            if indep:
                # Add a random edge within the independent set
                u, v = random.sample(indep, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            # Valid!
            best_adj_36 = adj
            found_36 = True
            break
        
        if found_36:
            break

    if found_36:
        return {'n': 36, 'edges': extract_edges(best_adj_36, 36)}

    # --- Fallback: Search for n=35 ---
    # We know 35 exists. Use similar repair.
    # Or use the known Exoo construction if we can code it quickly.
    # The product graph construction from the current code is a good start.
    # Let's try to repair the product graph for n=35.
    
    n35 = 35
    D5 = {1, 4}
    D7 = {1, 2, 4}
    
    adj35 = [0] * n35
    for i in range(n35):
        a1, b1 = divmod(i, 7)
        for j in range(i+1, n35):
            a2, b2 = divmod(j, 7)
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i] |= (1 << j)
                adj35[j] |= (1 << i)
                
    # Check if it's already valid
    if not has_k4(adj35, n35) and not has_indep_6(adj35, n35):
        return {'n': 35, 'edges': extract_edges(adj35, n35)}
        
    # Repair it
    for step in range(5000):
        if time.time() > deadline - 2:
            break
            
        k4 = get_k4(adj35, n35)
        if k4:
            u, v, i, j = k4
            pairs = [(u,v), (u,i), (u,j), (v,i), (v,j), (i,j)]
            pu, pv = random.choice(pairs)
            adj35[pu] &= ~(1 << pv)
            adj35[pv] &= ~(1 << pu)
            continue
            
        indep = get_indep_6(adj35, n35)
        if indep:
            u, v = random.sample(indep, 2)
            adj35[u] |= (1 << v)
            adj35[v] |= (1 << u)
            continue
            
        # Valid
        return {'n': 35, 'edges': extract_edges(adj35, n35)}
        
    # If repair failed (shouldn't happen for 35), return a smaller known valid graph
    # Paley 17
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj_paley = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % 17 in squares:
                adj_paley[i] |= (1 << j)
                adj_paley[j] |= (1 << i)
                
    return {'n': 17, 'edges': extract_edges(adj_paley, n)}
