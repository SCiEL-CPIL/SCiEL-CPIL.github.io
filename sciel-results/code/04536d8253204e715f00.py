import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 280
    
    def verify_graph(adj, n, size=5):
        """Full verification that no K5 or I5 exists using efficient bit operations."""
        # Precompute neighbor lists as sets for faster intersection checks if needed, 
        # but bitmasks are already fast enough for n<=42.
        # We check all combinations of size 5.
        # Optimization: Iterate through all 5-subsets.
        # 42C5 = 850668.
        # For each, check 10 edges. 8.5M ops. ~1-2s in Python.
        
        # A faster check:
        # For each 5-subset, check if clique or independent.
        for subset in combinations(range(n), size):
            is_clique = True
            is_indep = True
            for i in range(size):
                for j in range(i+1, size):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    def get_k4s_i4s_bitmasks(adj, n):
        """Get all K4s and I4s as bitmasks for fast subset testing."""
        k4s = []
        i4s = []
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i+1, 4):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                        if not is_indep:
                            break
                    if not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_clique:
                k4s.append(mask)
            if is_indep:
                i4s.append(mask)
        return k4s, i4s

    def try_add_vertex(adj, k4_masks, i4_masks, current_n, deadline):
        """
        Try to add a new vertex by finding a neighbor set S such that:
        1. No K4 bitmask is a subset of S (no K5 with new vertex)
        2. No I4 bitmask is a subset of V\S (no I5 with new vertex)
        Uses bitmask local search.
        """
        full_mask = (1 << current_n) - 1
        
        for restart in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize S: start with ~50% inclusion
            S_mask = 0
            for v in range(current_n):
                if random.random() < 0.5:
                    S_mask |= (1 << v)
            
            # Local search
            for it in range(3000):
                if time.time() > deadline - 15:
                    break
                
                # Check violations using bitmask subset test
                bad_k4 = None
                for km in k4_masks:
                    if (km & S_mask) == km:  # km is subset of S_mask
                        bad_k4 = km
                        break
                
                bad_i4 = None
                if bad_k4 is None:
                    V_minus_S = full_mask ^ S_mask
                    for im in i4_masks:
                        if (im & V_minus_S) == im:  # im is subset of V\S
                            bad_i4 = im
                            break
                
                if bad_k4 is None and bad_i4 is None:
                    # Success
                    S_set = set(v for v in range(current_n) if (S_mask >> v) & 1)
                    return S_set
                
                if bad_k4 is not None:
                    # Remove a random vertex from bad_k4 that is in S
                    candidates = [v for v in range(current_n) if (bad_k4 >> v) & 1 and (S_mask >> v) & 1]
                    if candidates:
                        v_to_remove = random.choice(candidates)
                        S_mask &= ~(1 << v_to_remove)
                    else:
                        break
                elif bad_i4 is not None:
                    # Add a random vertex from bad_i4 that is NOT in S
                    candidates = [v for v in range(current_n) if (bad_i4 >> v) & 1 and not ((S_mask >> v) & 1)]
                    if candidates:
                        v_to_add = random.choice(candidates)
                        S_mask |= (1 << v_to_add)
                    else:
                        break
            
        return None

    # Start from Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    current_adj = adj
    
    # Verify the starting graph
    if not verify_graph(current_adj, current_n):
        # Fallback: return edges as is if verification fails (should not happen for Paley 37)
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Precompute K4s and I4s as bitmasks
    k4_masks, i4_masks = get_k4s_i4s_bitmasks(current_adj, current_n)
    
    # Try to add vertices one at a time up to 42
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 40:
            break
        
        # Try multiple times to add a vertex
        success = False
        for attempt in range(30):
            if time.time() > deadline - 40:
                break
            
            S = try_add_vertex(current_adj, k4_masks, i4_masks, current_n, deadline)
            if S is not None:
                # Add new vertex
                new_idx = current_n
                current_adj.append(0)
                for v in S:
                    current_adj[v] |= (1 << new_idx)
                    current_adj[new_idx] |= (1 << v)
                
                current_n += 1
                
                # Update K4s and I4s - recompute
                k4_masks, i4_masks = get_k4s_i4s_bitmasks(current_adj, current_n)
                
                success = True
                break
        
        if not success:
            break
    
    # Final verification
    if verify_graph(current_adj, current_n):
        edges = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                if (current_adj[i] >> j) & 1:
                    edges.append((i, j))
        return current_n, edges
    
    # Fallback to Paley graph on 37
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
