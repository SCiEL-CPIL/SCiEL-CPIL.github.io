"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    n = 40

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                Na = adj[a]
                Nd = adj[d]
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def k5_seed():
        edges = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    edges.add((u, v))
        return edges

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val):
        """
        Improve the edge set by adding edges and swapping.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        
        # Pass 1: Add any possible non-edge
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Pass 2: Try swaps (remove 1, add 2)
        removable = list(current_edges)
        random.shuffle(removable)
        
        # Re-generate non_edges list as it changed
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 50
        
        while time.time() < deadline - 2:
            if consecutive_failures > max_failures:
                break
                
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add 2 non-edges
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 2:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                # Success: Edge count increased by (added - 1)
                consecutive_failures = 0
                # Update removable and non_edges for next iteration
                removable = list(current_edges)
                random.shuffle(removable)
                
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                # Continue searching for more improvements
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        return current_edges

    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        # Remove num_remove random edges
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        # Rebuild adj
        adj = build_adj(edges)
        
        # Run greedy
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        # Run local search
        edges = local_search(edges, seed_val)
        
        return edges

    # Main search loop
    best_edges = k5_seed()
    
    # Initial greedy
    best_edges = run_greedy(best_edges, 42)
    
    # Local search
    best_edges = local_search(best_edges, 42)
    
    best_count = len(best_edges)
    
    # Multiple restarts with perturbation
    restart_seed = 100
    for i in range(30):
        if time.time() > deadline - 10:
            break
        
        # Perturb the best solution
        num_remove = random.randint(2, 6)
        perturbed = perturb_and_refine(best_edges, restart_seed + i, num_remove)
        
        if len(perturbed) > best_count:
            best_edges = perturbed
            best_count = len(perturbed)
        
        restart_seed += 1
    
    # Convert to list of lists
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
