import time
import random
import numpy as np

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()
    
    adj = [0] * n
    A2 = [0] * n
    
    def update_A2():
        for i in range(n):
            mask = 0
            a = adj[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj[k]
                a ^= lsb
            A2[i] = mask

    def check_c6_add(u, v):
        # Returns True if adding (u,v) creates a C6
        a_u = adj[u]
        a_v = adj[v]
        
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            w_mask = a_v
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                # Check if there is a path of length 3 between x and w
                # This is equivalent to N(x) & A2[w] != 0
                if adj[x] & A2[w]:
                    return True
        
        return False

    def get_edges_count(adj_list):
        return sum(bin(a).count("1") for a in adj_list) // 2

    def generate_random_regular_graph(degree):
        # Generate a random degree-regular graph
        adj_local = [0] * n
        stubs = []
        for i in range(n):
            for _ in range(degree):
                stubs.append(i)
        
        for _ in range(100):
            random.shuffle(stubs)
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v or (adj_local[u] >> v) & 1:
                    valid = False
                    break
                adj_local[u] |= (1 << v)
                adj_local[v] |= (1 << u)
            if valid:
                return adj_local
        return None

    best_edges = 0
    best_adj = [0] * n

    # Strategy: 
    # 1. Try to build a 6-regular graph (120 edges) - likely too dense for C6-free on 40 nodes? 
    #    ex(40, C6) is ~105. So 5-regular (100 edges) is a safe start.
    # 2. Use a deterministic greedy completion: 
    #    - Iterate through all non-edges in a fixed order (or sorted by heuristic).
    #    - Add edge if valid.
    # 3. Use "2-swap" moves: Remove one edge, add two edges. 
    #    Systematically scan for such moves.

    # Phase 1: Initialization and Greedy Completion
    # We will try multiple random 5-regular starts and greedily complete them.
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        # Reset
        for i in range(n):
            adj[i] = 0
            A2[i] = 0
            
        # Start with a random 5-regular graph
        init_adj = generate_random_regular_graph(5)
        if init_adj is None:
            continue
            
        for i in range(n):
            adj[i] = init_adj[i]
            
        update_A2()
        
        # Greedy Completion: 
        # We want to add edges. 
        # Heuristic: Add edges that connect vertices with high "potential" but don't create C6.
        # To make it deterministic and fast, we just iterate all non-edges.
        
        # Pre-compute list of all non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
                    
        # Sort non-edges by a heuristic? 
        # Let's try sorting by the sum of degrees of the endpoints (preferential attachment-like)
        # or just random order to avoid bias? 
        # The current program uses random. Let's use a fixed order for determinism in this phase, 
        # but randomize the initial graph.
        
        # Actually, a better heuristic for C6-free is to add edges that create the "least" new C6s 
        # (i.e., edges where N(u) and N(v) have few common neighbors' neighbors).
        # But that's expensive. Let's stick to a simple greedy: 
        # Try to add all possible edges. 
        # To escape local optima of the greedy, we will do a "swap" phase later.
        
        changed = True
        while changed:
            changed = False
            # Shuffle non-edges to avoid bias in the greedy path
            random.shuffle(non_edges)
            for u, v in non_edges:
                if (adj[u] >> v) & 1:
                    continue
                if not check_c6_add(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    update_A2()
                    changed = True
                    # Remove from non_edges? Or just mark as added. 
                    # Since we update adj, the check `(adj[u] >> v) & 1` will catch it next time.
                    
        current_edges = get_edges_count(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Phase 2: Structured Swap Search
        # Try to remove one edge and add two edges.
        # This is a "2-for-1" move.
        # We will systematically try removing each edge and see if we can add two new edges.
        
        # Get list of current edges
        edges_list = []
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges_list.append((i, j))
                a ^= lsb
                
        # For each edge, try to remove it and add two others
        # This is O(E * E^2) which is too slow. 
        # Instead, let's do a random search for swaps, but more structured than the previous SA.
        # We will try to find a "triangle" of edges where removing one allows adding two.
        
        # Alternative: Use the "kick" approach from the inspiration, but deterministic?
        # Let's stick to the greedy completion which seems to be the bottleneck.
        # The previous program's SA was slow. 
        # If greedy completion already reaches 105, we are good.
        
        # If not, we can try a few random swaps.
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            if not edges_list:
                break
            a, b = random.choice(edges_list)
            
            # Backup
            adj_backup = adj[:]
            A2_backup = A2[:]
            
            # Remove (a,b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            update_A2()
            
            # Try to add two random edges
            added = 0
            c = random.randint(0, n - 1)
            d = random.randint(0, n - 1)
            if c != d and not (adj[c] >> d) & 1:
                if not check_c6_add(c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    update_A2()
                    added += 1
            
            if added == 1:
                # Try to add one more
                c2 = random.randint(0, n - 1)
                d2 = random.randint(0, n - 1)
                if c2 != d2 and not (adj[c2] >> d2) & 1:
                    if not check_c6_add(c2, d2):
                        adj[c2] |= (1 << d2)
                        adj[d2] |= (1 << c2)
                        update_A2()
                        added += 1
                        
            if added >= 1:
                # Accept if we increased edge count or maintained it (for exploration)
                # Here we remove 1 and add 1 or 2.
                # If we add 2, net gain +1. If we add 1, net gain 0.
                # We accept if net gain > 0. If net gain == 0, accept with prob 0.5?
                # Let's just accept if net gain > 0.
                new_edges = get_edges_count(adj)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
                # Keep the state if it's good for further swaps?
                # If we didn't improve best, revert to allow exploration?
                # Let's keep the state if it's a valid graph, to allow chaining swaps.
                # But we must update edges_list? It's expensive.
                # Let's just revert if we didn't improve the current graph's edge count.
                # Current graph edge count before swap was current_edges.
                # If new_edges <= current_edges, revert.
                if new_edges <= current_edges:
                    adj = adj_backup
                    A2 = A2_backup
            else:
                adj = adj_backup
                A2 = A2_backup

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        a = best_adj[u]
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            a ^= lsb
            
    return edges
