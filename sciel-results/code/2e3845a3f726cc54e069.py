import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # Safety margin: 5s buffer before 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bitset adjacency: adj[i] is an integer where bit j is set if (i,j) is an edge
    adj = [0] * n
    
    def can_add(adj, u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        Fast check using bit operations.
        """
        Nv = adj[v]
        Nu = adj[u]
        
        # Iterate x in N(v), w in N(u)
        # We need to check if there's a path of length 2 between N(x) and N(w)
        # that avoids u, v, x, w
        
        # For speed, iterate over the smaller of N(v) and N(u) first
        if bin(Nv).count('1') <= bin(Nu).count('1'):
            outer = Nv
            inner = Nu
            swap = False
        else:
            outer = Nu
            inner = Nv
            swap = True
        
        mask = outer
        while mask:
            x = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            
            if x == u: continue
            
            imask = inner
            while imask:
                w = (imask & -imask).bit_length() - 1
                imask &= imask - 1
                
                if w == v or w == x: continue
                
                # S_x = N(x) \ {u, v, w}
                S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                # S_w = N(w) \ {u, v, x}
                S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # Check if there's an edge between S_x and S_w (excluding self-loops)
                # For each y in S_x, check if adj[y] & (S_w & ~(1<<y)) != 0
                ymask = S_x
                while ymask:
                    y = (ymask & -ymask).bit_length() - 1
                    ymask &= ymask - 1
                    if adj[y] & (S_w & ~(1 << y)):
                        return False
        return True

    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    
    best_edges = 0
    best_adj = [0] * n
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Multiple random greedy trials with different orderings
    for trial in range(80):
        if time_left() < 15:
            break
            
        adj_try = [0] * n
        ec = 0
        
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 10:
                break
            
            if can_add(adj_try, u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]
    
    # Phase 2: Local search with 1-for-1 and 1-for-2 exchanges
    adj = best_adj[:]
    current_edges = best_edges
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time_left() > 5:
        progress = False
        
        # Try adding a single edge
        for _ in range(100):
            if time_left() < 5:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                progress = True
                no_improve_count = 0
                break
        
        if progress:
            continue
        
        # Try 1-for-2 exchange: remove one edge, try to add two new ones
        found_edge = False
        u_rem, v_rem = -1, -1
        for _ in range(50):
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                u_rem, v_rem = u, v
                found_edge = True
                break
        
        if found_edge:
            adj_backup = adj[:]
            
            # Remove the edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            
            added_count = 0
            added_pairs = []
            candidates = all_pairs[:]
            random.shuffle(candidates)
            
            for cu, cv in candidates:
                if time_left() < 3:
                    break
                if added_count >= 2:
                    break
                if adj[cu] & (1 << cv):
                    continue
                if (cu, cv) == (u_rem, v_rem) or (cu, cv) == (v_rem, u_rem):
                    continue
                if can_add(adj, cu, cv):
                    adj[cu] |= (1 << cv)
                    adj[cv] |= (1 << cu)
                    added_count += 1
                    added_pairs.append((cu, cv))
            
            if added_count == 2:
                current_edges += added_count - 1
                no_improve_count = 0
            else:
                adj = adj_backup[:]
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Try a more aggressive escape: remove 2 random edges, add up to 3
            u1, v1 = -1, -1
            u2, v2 = -1, -1
            edge_list = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    j = (mask & -mask).bit_length() - 1
                    mask &= mask - 1
                    if i < j:
                        edge_list.append((i, j))
            
            if len(edge_list) >= 2:
                e1, e2 = random.sample(edge_list, 2)
                u1, v1 = e1
                u2, v2 = e2
                
                adj_backup = adj[:]
                
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                
                candidates = all_pairs[:]
                random.shuffle(candidates)
                
                added_count = 0
                for cu, cv in candidates:
                    if time_left() < 3:
                        break
                    if added_count >= 3:
                        break
                    if adj[cu] & (1 << cv):
                        continue
                    if (cu, cv) in [(u1, v1), (v1, u1), (u2, v2), (v2, u2)]:
                        continue
                    if can_add(adj, cu, cv):
                        adj[cu] |= (1 << cv)
                        adj[cv] |= (1 << cu)
                        added_count += 1
                
                if added_count > 2:
                    current_edges += added_count - 2
                    no_improve_count = 0
                else:
                    adj = adj_backup[:]
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Update best if improved
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    return get_edges(best_adj)
