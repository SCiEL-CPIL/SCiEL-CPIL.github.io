import time
import random
import sys

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()
    
    # Bitset representation of adjacency
    adj = [0] * n
    
    # A2[i] is an integer where bit j is 1 if there is a path of length 2 from i to j
    # A2[i] = OR of adj[k] for all k in adj[i]
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

    # 1. Initial Graph: Random 5-regular
    # We aim for ~100 edges to have a dense start.
    
    max_attempts = 100
    success = False
    
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Reset adj
        for i in range(n):
            adj[i] = 0
            
        # Create list of stubs
        stubs = []
        for i in range(n):
            for _ in range(5):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair them up
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            
            if u == v:
                valid = False
                break
            if (adj[u] >> v) & 1:
                valid = False
                break
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if valid:
            success = True
            break
    
    if not success:
        # Fallback: Disjoint K5s
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    update_A2()
    
    best_edges = sum(bin(a).count("1") for a in adj) // 2
    best_adj = adj[:]
    
    # 2. Backtracking Search
    # We try to add edges one by one.
    # To be fast, we pre-compute the list of candidate edges (non-edges).
    # We shuffle them to break symmetry and randomize the search path.
    
    def get_candidates():
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    candidates.append((u, v))
        random.shuffle(candidates)
        return candidates

    # We will use a recursive backtracking function.
    # To avoid deep recursion issues and to manage time, we will implement a manual stack or limit depth.
    # Given the time limit, we might not explore the full tree.
    # We will try to find a graph with > 105 edges.
    
    # Optimization: 
    # Instead of full backtracking which might be too slow, 
    # we will do a "Greedy with Backtracking" hybrid.
    # We maintain a stack of states.
    # Actually, a simpler approach: 
    # Run a randomized greedy, but if stuck, backtrack.
    
    # Let's try a specific backtracking search for a few seconds.
    # We'll use a depth-first search on the graph of adding edges.
    
    # State: adj, A2, edge_count
    # We want to maximize edge_count.
    
    # To make it fast, we only backtrack if we hit a "dead end" (no more edges can be added) 
    # and the edge count is not significantly better than best.
    
    # Actually, let's just do a randomized greedy with a "restart" mechanism 
    # but with a smarter initial state or a different move set.
    
    # Wait, the prompt asks for a MATERIAL change.
    # Backtracking is a good candidate.
    
    # Let's implement a simple backtracker that tries to add edges.
    # We will limit the search depth or time.
    
    def backtrack(adj_state, A2_state, current_edges, depth):
        nonlocal best_edges, best_adj
        
        if time.time() > deadline - 2:
            return
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj_state[:]
            
        # Find all valid edges to add
        # To speed up, we can just iterate through all pairs.
        # n=40, 780 pairs.
        
        added_any = False
        # We want to try adding edges.
        # We'll pick a random valid edge to add, recurse, then remove it.
        
        # To avoid O(N^2) overhead per node if we do it naively, 
        # we can pre-compute valid edges.
        
        valid_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_state[u] >> v) & 1:
                    # Check if adding is valid
                    # We need to check C6.
                    # check_c6_add uses global adj and A2.
                    # We need a version that takes state.
                    
                    # For simplicity and speed, we'll just use the global check 
                    # but we must swap state.
                    # This is slow.
                    
                    # Let's write a fast check that uses the passed state.
                    # check_c6_add_state(u, v, adj_state, A2_state)
                    
                    pass
                    
        # This approach is too slow in Python if we check all pairs every time.
        # We need a faster way.
        
        # Alternative:
        # Use the existing check_c6_add but swap the global variables.
        # This is O(1) swap, but check is O(D^2).
        
        # Let's try to add ONE random valid edge, recurse, then remove.
        # This is a random walk in the solution space, but with backtracking.
        
        # Pick a random non-edge
        found = False
        for _ in range(10): # Try a few random picks
             u = random.randint(0, n - 1)
             v = random.randint(0, n - 1)
             if u == v or (adj_state[u] >> v) & 1:
                 continue
             
             # Check if valid
             # We need to check against adj_state and A2_state
             # Let's implement a helper
             
             def is_valid_in_state(u, v, adj_s, A2_s):
                 a_u = adj_s[u]
                 a_v = adj_s[v]
                 
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
                         
                         if adj_s[x] & A2_s[w]:
                             return False
                 return True
             
             if is_valid_in_state(u, v, adj_state, A2_state):
                 # Add
                 adj_state[u] |= (1 << v)
                 adj_state[v] |= (1 << u)
                 
                 # Update A2 for affected nodes
                 # A2[u] changes, A2[v] changes, and A2[x] for x in N(u) U N(v)
                 # Actually, A2[i] depends on adj[i] and adj[neighbors].
                 # If we add (u,v):
                 # 1. adj[u] and adj[v] change.
                 # 2. A2[u] = OR(adj[k] for k in N(u)). N(u) gains v. So A2[u] |= adj[v].
                 #    Also, for any k in N(u), adj[k] is unchanged, but A2[k] includes adj[u].
                 #    So A2[k] |= adj[v] for k in N(u)?
                 #    Wait. A2[k] = OR(adj[z] for z in N(k)).
                 #    If z=u, then adj[u] changed. So A2[k] changes.
                 #    Specifically, A2[k] gains adj[v] (since u is now connected to v).
                 #    So for all k in N(u) (before adding), A2[k] |= adj[v].
                 #    Similarly for N(v).
                 
                 # Let's just recompute A2 for all nodes? O(N*D^2) ~ 40*25 = 1000 ops.
                 # This is fast enough for a few thousand steps.
                 
                 for i in range(n):
                     mask = 0
                     a = adj_state[i]
                     while a:
                         lsb = a & -a
                         k = lsb.bit_length() - 1
                         mask |= adj_state[k]
                         a ^= lsb
                     A2_state[i] = mask
                     
                 backtrack(adj_state, A2_state, current_edges + 1, depth + 1)
                 
                 # Remove
                 adj_state[u] &= ~(1 << v)
                 adj_state[v] &= ~(1 << u)
                 
                 for i in range(n):
                     mask = 0
                     a = adj_state[i]
                     while a:
                         lsb = a & -a
                         k = lsb.bit_length() - 1
                         mask |= adj_state[k]
                         a ^= lsb
                     A2_state[i] = mask
                     
                 found = True
                 break
                 
        if not found:
             # Dead end
             pass

    # We can't run deep backtracking easily without hitting recursion limits or time.
    # Let's stick to a modified Simulated Annealing but with a "jump" mechanism.
    # The current SA is stuck.
    # Let's try a "Restart" strategy:
    # Run SA for 5 seconds.
    # If no improvement, restart with a different initial graph.
    # Or, use a different heuristic: 
    # "Maximize minimum degree" or "Maximize spectral gap".
    
    # Actually, let's try to implement a **Local Search with Tabu List** or **Iterated Local Search (ILS)**.
    # ILS: 
    # 1. Local Search (Greedy Add) to local optimum.
    # 2. Perturbation (Remove K edges, Add K edges randomly).
    # 3. Local Search again.
    # 4. Accept if better.
    
    # This is a standard metaheuristic that often beats simple SA.
    
    def local_search(adj_s, A2_s):
        improved = True
        edges = sum(bin(a).count("1") for a in adj_s) // 2
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any edge
            # To be fast, we iterate through all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_s[u] >> v) & 1:
                        continue
                    if not check_c6_add_state(u, v, adj_s, A2_s):
                        # Add
                        adj_s[u] |= (1 << v)
                        adj_s[v] |= (1 << u)
                        update_A2_state(adj_s, A2_s)
                        edges += 1
                        improved = True
                        # Restart loop to check again?
                        # Or just continue?
                        # If we add one, we might enable others.
                        # Let's break and restart the while loop
                        break
                if improved:
                    break
        return edges

    def check_c6_add_state(u, v, adj_s, A2_s):
        a_u = adj_s[u]
        a_v = adj_s[v]
        
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
                
                if adj_s[x] & A2_s[w]:
                    return True
        
        return False

    def update_A2_state(adj_s, A2_s):
        for i in range(n):
            mask = 0
            a = adj_s[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj_s[k]
                a ^= lsb
            A2_s[i] = mask

    # ILS Loop
    current_adj = adj[:]
    current_A2 = A2[:]
    current_edges = best_edges
    
    while time.time() < deadline - 2:
        # 1. Local Search
        new_edges = local_search(current_adj, current_A2)
        
        if new_edges > best_edges:
            best_edges = new_edges
            best_adj = current_adj[:]
            
        # 2. Perturbation
        # Remove 5 random edges
        edges_list = []
        for i in range(n):
            a = current_adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges_list.append((i, j))
                a ^= lsb
        
        if not edges_list:
            break
            
        to_remove = random.sample(edges_list, min(5, len(edges_list)))
        for u, v in to_remove:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
        update_A2_state(current_adj, current_A2)
        
        # Add 5 random edges (if valid)
        added = 0
        attempts = 0
        while added < 5 and attempts < 50:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (current_adj[u] >> v) & 1:
                continue
            if not check_c6_add_state(u, v, current_adj, current_A2):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                update_A2_state(current_adj, current_A2)
                added += 1
            attempts += 1
            
        # 3. Accept if better (or always accept to explore?)
        # ILS usually accepts if better, or with probability.
        # Let's accept if better.
        # If not better, we stay at the perturbed state? 
        # No, usually we go back to the best found or the previous local optimum.
        # Let's keep the perturbed state as the new start for the next iteration.
        
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
