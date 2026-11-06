import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit
    
    n = 40
    
    # --- C6 Checking Logic ---
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) to the graph represented by adj creates a C6.
        Returns True if safe, False if C6 created.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # We need to check if there is a path of length 3 between u and v
        # u - a - b - c - d - v
        # a in N(u), d in N(v), a != d
        # b in N(a) \ {u,v,d}, c in N(d) \ {u,v,a}
        # b ~ c
        
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # Candidates for b: neighbors of a, excluding u, v, d, a
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                
                # Candidates for c: neighbors of d, excluding u, v, a, d
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Let U = union of neighbors of all b in b_candidates
                # If U & c_candidates is non-empty, then a C6 exists.
                
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    
                    # Early exit if we already found a connection
                    if union_neighbors & c_candidates:
                        return False
                        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    # --- Simulated Annealing Setup ---
    
    # Initial state: Start with a moderately dense random graph
    # We need a valid starting point. Let's start with 8 disjoint K5s (80 edges)
    # and then try to add more via SA, or start sparser and add.
    # Actually, starting from a valid dense graph is better for SA swaps.
    # Let's initialize with 8 disjoint K5s.
    
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
                
    current_adj = base_adj[:]
    current_edges = 80
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # List of existing edges and non-existing edges for faster selection
    # We will maintain a list of existing edges to pick removals from
    # and a list of non-existing edges to pick additions from
    
    def get_edge_lists(adj):
        existing = []
        non_existing = []
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))
        return existing, non_existing

    existing_edges, non_existing_edges = get_edge_lists(current_adj)
    
    best_adj = current_adj[:]
    best_count = current_edges
    
    # SA Parameters
    # Temperature starts high and decays
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    # We will perform steps until time limit
    # Each step:
    # 1. Pick a random edge to remove (u1, v1) from existing
    # 2. Pick a random edge to add (u2, v2) from non_existing
    # 3. Check if adding (u2, v2) is valid after removing (u1, v1)
    #    Actually, we can check validity of adding (u2, v2) to the graph WITHOUT (u1, v1)
    #    This is slightly more complex.
    #    Alternative:
    #    a. Remove (u1, v1) from current_adj
    #    b. Check if (u2, v2) can be added to the new state
    #    c. If yes, accept move based on SA criteria (here delta is 0 since edge count is constant)
    #       Wait, SA for maximizing edges usually involves moves that change edge count.
    #       But if we restrict to "swap" moves, edge count stays constant.
    #       This explores the space of graphs with fixed edge count.
    #       To increase edge count, we need "add" moves.
    #       To decrease edge count (to escape traps), we need "remove" moves.
    
    # Let's use a mix of moves:
    # 1. Add an edge (if valid) -> Increases count by 1
    # 2. Remove an edge -> Decreases count by 1
    # 3. Swap an edge (remove one, add one) -> Count unchanged
    
    # Since we want to maximize edges, "Add" moves are always accepted if valid.
    # "Remove" moves are accepted with probability exp(-1/T) (penalty for losing edge)
    # "Swap" moves are accepted with probability 1 (neutral) or based on some heuristic?
    # Actually, if we just do Add/Remove, we might oscillate.
    # A better strategy for this specific problem:
    # Focus on "Add" moves primarily. If stuck, do "Remove" + "Add" (Swap) to rearrange structure.
    
    # Let's implement a simpler variant:
    # Loop:
    # 1. Try to add a random non-existing edge. If valid, accept.
    # 2. If no edge could be added in last K iterations, try a swap:
    #    Remove a random edge, then try to add a different random edge.
    #    If the add is valid, accept the swap.
    
    # This "Add" strategy is essentially greedy with randomization.
    # To make it SA-like, we allow "bad" moves (removing edges) to escape local optima.
    
    # Let's stick to the standard SA formulation:
    # State: Graph
    # Move:
    #   - With prob 0.7: Try to add a random non-edge. If valid, accept (Delta E = +1).
    #   - With prob 0.3: Try to remove a random edge and add a random non-edge.
    #     This is a swap. Delta E = 0.
    #     But checking validity of the add part requires the remove part to be done.
    #     So: Temporarily remove (u1,v1). Check if (u2,v2) is valid.
    #     If valid, we have a new valid graph with same edge count.
    #     Do we accept? Since Delta E = 0, we always accept?
    #     If we always accept neutral moves, we just random walk the fixed-edge-count space.
    #     This helps find a configuration where an "Add" move becomes possible.
    
    # Let's refine:
    # 1. Pick a move type.
    #    - Type A: Add edge. (Prob 0.5)
    #    - Type B: Swap edge. (Prob 0.5)
    
    # Type A:
    #   Pick random non-edge (u,v).
    #   If can_add_edge(current_adj, u, v):
    #       Add it.
    #       current_edges += 1
    #       Update lists
    #       If current_edges > best_count: update best
    #   Else:
    #       Reject (no change)
        
    # Type B:
    #   Pick random existing edge (u1, v1).
    #   Pick random non-edge (u2, v2).
    #   Temporarily remove (u1, v1).
    #   If can_add_edge(current_adj, u2, v2):
    #       Add (u2, v2).
    #       # Edge count unchanged.
    #       # Accept this move? 
    #       # Yes, because it might open up future Add moves.
    #       # We don't have a penalty for this move in terms of objective value.
    #       Update lists.
    #   Else:
    #       # Reject swap.
    #       Re-add (u1, v1).
    #       No change.
        
    # This is not really Simulated Annealing in the strict sense (no temperature based acceptance of worse states),
    # but it is a Randomized Local Search with Neutral Moves.
    # To make it more robust, we can allow "Remove" moves (decreasing edge count) with a small probability
    # to escape deep local minima where no Add or Swap works.
    
    # Let's add Type C: Remove edge. (Prob 0.1)
    #   Pick random existing edge (u,v).
    #   Remove it.
    #   current_edges -= 1
    #   Update lists.
    #   (This is a "worse" move, but we do it to escape. We don't accept based on T, we just do it 
    #    and hope to recover. This is like a random restart of the local structure.)
    
    # Actually, a cleaner approach for this time limit:
    # Just do random restarts of the Greedy Add algorithm, but with a "shake" step.
    # 1. Start with 8 K5s.
    # 2. Greedily add edges until stuck.
    # 3. If stuck, remove a random edge (or few) and try to add others.
    # 4. Repeat.
    
    # Let's implement a "Stochastic Local Search" that focuses on finding Add moves.
    
    iterations = 0
    stuck_counter = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Check time more frequently
        if time.time() > deadline - 2:
            break
            
        # Choose move type
        # If we are not stuck, prioritize Add
        # If we are stuck, prioritize Swap or Remove
        if stuck_counter < 10:
            move_type = 'add'
        elif stuck_counter < 50:
            move_type = random.choice(['add', 'swap'])
        else:
            move_type = random.choice(['add', 'swap', 'remove'])
            
        if move_type == 'add':
            if not non_existing_edges:
                break # No more edges to add
            u, v = random.choice(non_existing_edges)
            if can_add_edge(current_adj, u, v):
                # Accept
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                stuck_counter = 0
                
                # Update lists
                # Remove (u,v) from non_existing, add to existing
                # This is O(N) if using lists. 
                # For efficiency, we can just rebuild lists every K iterations or use sets.
                # Given 120s limit and ~1000-5000 iterations, O(N) list updates might be okay.
                # N=780. 5000 * 780 = 3.9M ops. Should be fine.
                
                non_existing_edges.remove((u, v))
                existing_edges.append((u, v))
                
                if current_edges > best_count:
                    best_count = current_edges
                    best_adj = current_adj[:]
                    if best_count >= 105:
                        break
            else:
                stuck_counter += 1
                
        elif move_type == 'swap':
            if not existing_edges or not non_existing_edges:
                continue
            u1, v1 = random.choice(existing_edges)
            u2, v2 = random.choice(non_existing_edges)
            
            # Temporarily remove (u1, v1)
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            # Check if (u2, v2) can be added
            if can_add_edge(current_adj, u2, v2):
                # Accept swap
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                # Update lists
                # (u1, v1) becomes non-existing, (u2, v2) becomes existing
                existing_edges.remove((u1, v1))
                non_existing_edges.append((u1, v1))
                
                non_existing_edges.remove((u2, v2))
                existing_edges.append((u2, v2))
                
                # Edge count unchanged, but structure changed.
                # Reset stuck counter slightly? Or keep it?
                # Since we made a change, reset stuck counter.
                stuck_counter = 0
            else:
                # Reject swap, re-add (u1, v1)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                stuck_counter += 1
                
        elif move_type == 'remove':
            if not existing_edges:
                continue
            u, v = random.choice(existing_edges)
            
            # Remove edge
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges -= 1
            
            # Update lists
            existing_edges.remove((u, v))
            non_existing_edges.append((u, v))
            
            # We accepted a "worse" move. 
            # Reset stuck counter to allow exploration.
            stuck_counter = 0
            
            # Note: We don't update best_adj here because count decreased.
            # But we might recover.
            
    # Return best found
    if best_count > 0:
        best_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])
        return best_edges_list
    else:
        # Fallback: return base
        best_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (base_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])
        return best_edges_list
