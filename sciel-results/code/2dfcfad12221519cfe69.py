import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    def count_edges(adj):
        """Count edges in adjacency bitmask list."""
        return sum(bin(adj[i]).count('1') for i in range(16)) // 2

    def is_3_colorable(adj):
        """
        Check if graph is 3-colorable using fast backtracking with bitmask operations.
        Returns True if 3-colorable, False if not.
        """
        # Precompute degrees and order
        degrees = [bin(adj[i]).count('1') for i in range(16)]
        # Order vertices by descending degree
        order = sorted(range(16), key=lambda x: -degrees[x])
        
        # Map original vertex to position in order
        pos = [0] * 16
        for i, v in enumerate(order):
            pos[v] = i
            
        # Precompute earlier neighbor masks in this order
        # earlier_mask[i] is bitmask of neighbors of order[i] that appear earlier in the order
        earlier_mask = [0] * 16
        for i, v in enumerate(order):
            mask = 0
            # Check all neighbors
            neighbors = adj[v]
            while neighbors:
                lsb = neighbors & (-neighbors)
                u = lsb.bit_length() - 1
                neighbors ^= lsb
                if pos[u] < i:
                    mask |= lsb
            earlier_mask[i] = mask

        color_masks = [0, 0, 0]
        
        def backtrack(i):
            if i == 16:
                return True
            
            v = order[i]
            em = earlier_mask[i]
            
            # Determine forbidden colors
            # Color c is forbidden if (adj[v] & color_masks[c]) != 0
            # But we only care about earlier neighbors? No, color_masks contains ALL colored vertices.
            # However, in standard backtracking with fixed order, we only assign colors to earlier vertices.
            # So checking adj[v] & color_masks[c] is correct.
            
            forbidden = 0
            if adj[v] & color_masks[0]:
                forbidden |= 1
            if adj[v] & color_masks[1]:
                forbidden |= 2
            if adj[v] & color_masks[2]:
                forbidden |= 4
            
            available = 7 & (~forbidden)
            
            if available == 0:
                return False
            
            # Try available colors
            # Heuristic: try color 0, 1, 2
            while available:
                lsb = available & (-available)
                c = 0 if lsb == 1 else (1 if lsb == 2 else 2)
                available ^= lsb
                
                color_masks[c] |= (1 << v)
                
                if backtrack(i + 1):
                    return True
                
                color_masks[c] ^= (1 << v)
            
            return False
        
        return backtrack(0)

    def build_grotzsch():
        """Build the Grötzsch graph on vertices 0-10, others isolated."""
        adj = [0] * 16
        # C5 on 0-4
        for i in range(5):
            j = (i + 1) % 5
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        
        # Mycielski: 5-9 are copies of 0-4, 10 is universal to 5-9
        for i in range(5):
            # 5+i is connected to neighbors of i in C5
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            # Connect 5+i to 10
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        
        return adj

    def adj_to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without creating a triangle."""
        if adj[u] & (1 << v):
            return False
        # Check common neighbors
        if adj[u] & adj[v]:
            return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def simulated_annealing(start_adj, deadline):
        """
        Run Simulated Annealing to maximize edges in a triangle-free, 4-chromatic graph.
        """
        current_adj = start_adj[:]
        current_edges = count_edges(current_adj)
        current_is_4ch = not is_3_colorable(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # SA Parameters
        T_start = 10.0
        T_end = 0.1
        # Estimate steps: 150s / ~10us per step (optimistic) = 15M steps.
        # But colorability check is ~100us. 
        # We will do many cheap moves (add edge to 4ch) and fewer expensive ones.
        # Let's just run until deadline.
        
        all_pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        
        # Penalty for being 3-colorable
        P = 100.0
        
        steps = 0
        while time.time() < deadline:
            # Cool temperature
            # Use a slow cooling schedule
            T = T_start * (T_end / T_start) ** (steps / 50000.0)
            if T < T_end:
                T = T_end
            
            # Pick random edge
            u, v = random.choice(all_pairs)
            
            edge_exists = bool(current_adj[u] & (1 << v))
            
            new_adj = current_adj[:]
            new_edges = current_edges
            new_is_4ch = current_is_4ch
            
            if edge_exists:
                # Remove edge
                remove_edge(new_adj, u, v)
                new_edges -= 1
                
                if current_is_4ch:
                    # Might become 3-colorable
                    new_is_4ch = not is_3_colorable(new_adj)
                else:
                    # Stays 3-colorable
                    new_is_4ch = False
            else:
                # Add edge
                if not can_add_edge(current_adj, u, v):
                    # Triangle violation, reject
                    steps += 1
                    continue
                
                add_edge(new_adj, u, v)
                new_edges += 1
                
                if current_is_4ch:
                    # Stays 4-chromatic (monotonicity)
                    new_is_4ch = True
                else:
                    # Might become 4-chromatic
                    new_is_4ch = not is_3_colorable(new_adj)
            
            # Calculate Energy
            # E = -edges + P * (0 if 4ch else 1)
            E_old = -current_edges + (0.0 if current_is_4ch else P)
            E_new = -new_edges + (0.0 if new_is_4ch else P)
            
            delta_E = E_new - E_old
            
            accept = False
            if delta_E <= 0:
                accept = True
            else:
                # Accept with probability exp(-delta_E / T)
                if T > 0:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                current_adj = new_adj
                current_edges = new_edges
                current_is_4ch = new_is_4ch
                
                # Update best if valid (4-chromatic) and better
                if current_is_4ch and current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            
            steps += 1
            
            # Safety break if stuck or time up
            if time.time() > deadline - 5:
                break
        
        return best_adj, best_edges

    def local_search_maximize(adj, deadline):
        """
        Greedy local search: add any edge that preserves 4-chromaticity and triangle-freeness.
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        is_4ch = not is_3_colorable(current_adj)
        
        if not is_4ch:
            return current_adj, current_edges
        
        all_pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u, v in all_pairs:
                if time.time() > deadline - 2:
                    break
                if current_adj[u] & (1 << v):
                    continue
                if not can_add_edge(current_adj, u, v):
                    continue
                
                # Try adding
                add_edge(current_adj, u, v)
                # Since we are 4-chromatic, adding edge keeps it 4-chromatic
                # No need to check colorability!
                current_edges += 1
                improved = True
                
        return current_adj, current_edges

    # Main Logic
    best_overall_adj = None
    best_overall_edges = 0
    
    # Phase 1: Start from Grötzsch
    grotzsch = build_grotzsch()
    # Extend Grötzsch with some random edges to give SA a head start?
    # Or just let SA do it.
    
    # Run SA with multiple restarts
    # Restart 1: Pure Grötzsch
    adj1, cnt1 = simulated_annealing(grotzsch, deadline - 30)
    if cnt1 > best_overall_edges:
        best_overall_edges = cnt1
        best_overall_adj = adj1[:]
    
    # Local search polish
    if best_overall_adj:
        adj_polished, cnt_polished = local_search_maximize(best_overall_adj, deadline - 10)
        if cnt_polished > best_overall_edges:
            best_overall_edges = cnt_polished
            best_overall_adj = adj_polished[:]
    
    # Restart 2: Perturb best found
    if best_overall_adj and time.time() < deadline - 40:
        perturbed = best_overall_adj[:]
        # Remove 5 random edges
        edges_to_remove = []
        for i in range(16):
            for j in range(i + 1, 16):
                if perturbed[i] & (1 << j):
                    edges_to_remove.append((i, j))
        
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:5]:
            remove_edge(perturbed, u, v)
        
        adj2, cnt2 = simulated_annealing(perturbed, deadline - 10)
        if cnt2 > best_overall_edges:
            best_overall_edges = cnt2
            best_overall_adj = adj2[:]
        
        if best_overall_adj:
            adj_polished, cnt_polished = local_search_maximize(best_overall_adj, deadline - 5)
            if cnt_polished > best_overall_edges:
                best_overall_edges = cnt_polished
                best_overall_adj = adj_polished[:]
                
    # Restart 3: Random dense triangle-free graph (might be 3-colorable, SA will try to fix)
    if time.time() < deadline - 20:
        random_adj = [0] * 16
        # Generate random triangle-free graph
        pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add_edge(random_adj, u, v):
                if random.random() < 0.5: # Sparse to start
                    add_edge(random_adj, u, v)
        
        # Check if it's 4-chromatic. If not, SA might struggle to get there.
        # But if it's 3-colorable, SA will try to add edges to make it 4-chromatic.
        # This is a valid strategy.
        adj3, cnt3 = simulated_annealing(random_adj, deadline - 5)
        if cnt3 > best_overall_edges:
            best_overall_edges = cnt3
            best_overall_adj = adj3[:]
            
        if best_overall_adj:
            adj_polished, cnt_polished = local_search_maximize(best_overall_adj, deadline)
            if cnt_polished > best_overall_edges:
                best_overall_edges = cnt_polished
                best_overall_adj = adj_polished[:]

    if best_overall_adj is None:
        # Fallback
        return adj_to_edges(build_grotzsch())
        
    return adj_to_edges(best_overall_adj)

# EVOLVE-BLOCK-END
