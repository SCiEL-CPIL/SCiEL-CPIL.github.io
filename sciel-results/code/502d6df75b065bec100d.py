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
    
    def update_A2(adj_state, A2_state):
        for i in range(n):
            mask = 0
            a = adj_state[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj_state[k]
                a ^= lsb
            A2_state[i] = mask

    def check_c6_add(u, v, adj_state, A2_state):
        """
        Returns True if adding (u,v) creates a C4 or C6.
        """
        # Check C4 first (faster)
        # C4 check: N(u) & A2[v] != 0
        if adj_state[u] & A2_state[v]:
            return True
        if adj_state[v] & A2_state[u]:
            return True
            
        # Check C6
        a_u = adj_state[u]
        a_v = adj_state[v]
        
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
                if adj_state[x] & A2_state[w]:
                    return True
        return False

    # 1. Initial Graph: Start with a C4-free structure to ensure safety.
    # A disjoint union of K5s is C4-free and C6-free.
    # 8 * K5 = 40 vertices.
    # Edges: 8 * 10 = 80 edges.
    
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    update_A2(adj, A2)
    
    best_edges = sum(bin(a).count("1") for a in adj) // 2
    best_adj = adj[:]
    
    # 2. Simulated Annealing
    # We use edge-swap moves: remove one edge, add one edge.
    # This allows us to escape local optima where no single edge can be added.
    
    current_adj = adj[:]
    current_A2 = A2[:]
    current_edges = best_edges
    
    # Temperature schedule
    # Start high to allow moves that decrease edge count (removals),
    # then cool down to accept only improvements.
    # Since we are maximizing edges, the "energy" is -edges.
    # We accept a move if it improves (dE < 0) or with probability exp(dE/T).
    # Here dE = -delta_edges. If we add an edge, delta_edges = +1, dE = -1 (improvement).
    # If we remove an edge, delta_edges = -1, dE = +1 (worsening).
    # If we swap (remove one, add one), delta_edges = 0, dE = 0 (neutral, always accept? or random).
    # Actually, a swap that keeps edge count same but changes structure is useful for exploration.
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline - 5:
        if T < T_min:
            # Reheat slightly or restart?
            # Let's just stop or do a local search restart.
            # For now, break if we cool down too much.
            break
        
        # Generate a random move:
        # 1. Pick a random edge to remove (u1, v1)
        # 2. Pick a random non-edge to add (u2, v2)
        # Check validity of the new state.
        
        # Collect current edges
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
            
        # Pick edge to remove
        u1, v1 = random.choice(edges_list)
        
        # Temporarily remove it
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        update_A2(current_adj, current_A2)
        
        # Pick a candidate edge to add
        # To make it efficient, we can pick a random pair (u2, v2) that is not an edge.
        # But we want to bias towards pairs that are "close" or likely to be valid?
        # Random is fine for SA.
        
        # To avoid picking the same edge we just removed, or other edges.
        # We can just pick a random u2, v2.
        
        found_add = False
        for _ in range(100): # Try a few times to find a valid non-edge
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (current_adj[u2] >> v2) & 1:
                continue
            
            # Check if adding (u2, v2) is valid
            if not check_c6_add(u2, v2, current_adj, current_A2):
                # Valid move!
                # Apply it
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                update_A2(current_adj, current_A2)
                
                # This is a swap: removed (u1,v1), added (u2,v2).
                # Edge count remains the same.
                # We always accept neutral moves in SA to explore the landscape?
                # Or we can treat it as a step.
                # Let's accept it.
                found_add = True
                break
        
        if not found_add:
            # If we couldn't find a valid add, we must restore the removed edge
            # to keep the graph valid.
            current_adj[u1] |= (1 << v1)
            current_adj[v1] |= (1 << u1)
            update_A2(current_adj, current_A2)
        else:
            # We accepted the swap.
            # Check if we improved? No, edge count is same.
            # But we might be in a better position for future adds.
            pass
        
        # After a swap, we might be able to add more edges greedily?
        # Let's try a quick greedy add phase after every few swaps.
        # This hybrid approach (SA + Greedy) is powerful.
        
        if random.random() < 0.1: # 10% chance to do a greedy add burst
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                valid_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (current_adj[u] >> v) & 1:
                            continue
                        if not check_c6_add(u, v, current_adj, current_A2):
                            valid_edges.append((u, v))
                if not valid_edges:
                    break
                u, v = random.choice(valid_edges)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                update_A2(current_adj, current_A2)
                current_edges += 1
                changed = True
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
        
        # Cool down
        T *= cooling_rate
        
    # Final Greedy Polish
    # Try to add any remaining valid edges
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        valid_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    continue
                # Check against best_adj
                # We need A2 for best_adj
                # Let's just recompute A2 for best_adj if we want to be safe,
                # or just use the check on the fly?
                # The check function takes adj and A2.
                # Let's compute A2 for best_adj once.
                pass
        
        # Simpler: just run the greedy add on best_adj
        A2_best = [0] * n
        update_A2(best_adj, A2_best)
        
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    continue
                if not check_c6_add(u, v, best_adj, A2_best):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    update_A2(best_adj, A2_best)
                    best_edges += 1
                    changed = True
        if not changed:
            break

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
