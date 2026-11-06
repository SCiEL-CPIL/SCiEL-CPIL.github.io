import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 280
    
    def verify_graph_fast(adj, n, size=5):
        """Fast verification using bitmasks."""
        for subset in combinations(range(n), size):
            is_clique = True
            is_indep = True
            for i in range(size):
                u = subset[i]
                for j in range(i+1, size):
                    v = subset[j]
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
        """Get all K4s and I4s as bitmasks."""
        k4s = []
        i4s = []
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                u = subset[i]
                for j in range(i+1, 4):
                    v = subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            
            if is_clique or is_indep:
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_clique:
                    k4s.append(mask)
                if is_indep:
                    i4s.append(mask)
        return k4s, i4s

    def try_add_vertex_fast(adj, k4_masks, i4_masks, current_n, deadline):
        """
        Optimized local search to find a valid neighbor set S for a new vertex.
        """
        full_mask = (1 << current_n) - 1
        
        # Precompute vertex lists for K4s and I4s for faster random selection if needed
        # But bitmask check is fast enough if we limit restarts/steps
        
        for restart in range(200):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            S_mask = 0
            for v in range(current_n):
                if random.random() < 0.5:
                    S_mask |= (1 << v)
            
            # Greedy Repair Loop
            steps = 0
            max_steps = 2000
            last_state = S_mask
            stuck_count = 0
            
            while steps < max_steps:
                if time.time() > deadline - 10:
                    break
                
                # Check for K4 violation
                bad_k4 = None
                for km in k4_masks:
                    if (km & S_mask) == km:
                        bad_k4 = km
                        break
                
                if bad_k4 is not None:
                    # Remove a vertex from bad_k4 that is in S
                    # Pick one that minimizes potential new violations? 
                    # For now, random choice among candidates in S
                    candidates = []
                    km = bad_k4
                    while km:
                        lsb = km & -km
                        v = lsb.bit_length() - 1
                        if (S_mask >> v) & 1:
                            candidates.append(v)
                        km ^= lsb
                    
                    if candidates:
                        v_to_remove = random.choice(candidates)
                        S_mask &= ~(1 << v_to_remove)
                    else:
                        break # Should not happen if bad_k4 is subset of S
                
                # Check for I4 violation
                bad_i4 = None
                V_minus_S = full_mask ^ S_mask
                for im in i4_masks:
                    if (im & V_minus_S) == im:
                        bad_i4 = im
                        break
                
                if bad_i4 is not None:
                    # Add a vertex from bad_i4 that is NOT in S
                    candidates = []
                    im = bad_i4
                    while im:
                        lsb = im & -im
                        v = lsb.bit_length() - 1
                        if not ((S_mask >> v) & 1):
                            candidates.append(v)
                        im ^= lsb
                    
                    if candidates:
                        v_to_add = random.choice(candidates)
                        S_mask |= (1 << v_to_add)
                    else:
                        break # Should not happen if bad_i4 is subset of V\S
                
                if bad_k4 is None and bad_i4 is None:
                    # Success
                    S_set = set()
                    temp_mask = S_mask
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        v = lsb.bit_length() - 1
                        S_set.add(v)
                        temp_mask ^= lsb
                    return S_set
                
                # Detect cycling
                if S_mask == last_state:
                    stuck_count += 1
                    if stuck_count > 50:
                        break
                else:
                    stuck_count = 0
                    last_state = S_mask
                
                steps += 1
        
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
    if not verify_graph_fast(current_adj, current_n):
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Precompute K4s and I4s as bitmasks
    k4_masks, i4_masks = get_k4s_i4s_bitmasks(current_adj, current_n)
    
    # Try to add vertices one at a time up to 42
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 20:
            break
        
        # Try multiple times to add a vertex
        success = False
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            S = try_add_vertex_fast(current_adj, k4_masks, i4_masks, current_n, deadline)
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
    if verify_graph_fast(current_adj, current_n):
        edges = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                if (current_adj[i] >> j) & 1:
                    edges.append((i, j))
        return current_n, edges
    
    # Fallback to Paley graph on 37
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
