import time
import random
import math

def _build_bitmasks(n, edges):
    """Build adjacency bitmasks for fast clique/independent set checks."""
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _count_k5_and_ind5(n, masks):
    """Count K5s and independent 5-sets."""
    k5_count = 0
    ind5_count = 0
    
    # Precompute K3s and Ind3s? 
    # It's easier to just iterate all 5-sets for initial count? 
    # Binom(38,5) is ~500k. 
    # Each check is 10 edge checks. 5M ops. Acceptable for init.
    
    # But for delta, we need the structure.
    # Let's just use the delta method directly in the loop.
    # For initial count, we can use the same logic by "flipping" from an empty graph?
    # Or just brute force initial count.
    
    for i in range(n):
        for j in range(i+1, n):
            if not (masks[i] >> j) & 1:
                continue
            common_ij = masks[i] & masks[j]
            for l in range(j+1, n):
                if not (common_ij >> l) & 1:
                    continue
                common_ijl = common_ij & masks[l]
                for m in range(l+1, n):
                    if not (common_ijl >> m) & 1:
                        continue
                    common_ijlm = common_ijl & masks[m]
                    for p in range(m+1, n):
                        if (common_ijlm >> p) & 1:
                            k5_count += 1
                            
    # Independent sets
    # Complement graph masks
    # ind5 in G is k5 in G_complement
    # G_complement masks: ~masks[i] & ((1<<n)-1) ^ (1<<i)
    comp_masks = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_masks[i] = full_mask ^ masks[i] ^ (1 << i)
        
    for i in range(n):
        for j in range(i+1, n):
            if not (comp_masks[i] >> j) & 1:
                continue
            common_ij = comp_masks[i] & comp_masks[j]
            for l in range(j+1, n):
                if not (common_ij >> l) & 1:
                    continue
                common_ijl = common_ij & comp_masks[l]
                for m in range(l+1, n):
                    if not (common_ijl >> m) & 1:
                        continue
                    common_ijlm = common_ijl & comp_masks[m]
                    for p in range(m+1, n):
                        if (common_ijlm >> p) & 1:
                            ind5_count += 1
                            
    return k5_count, ind5_count

def _calculate_delta_penalty(n, masks, u, v):
    """
    Calculate change in penalty if edge (u, v) is flipped.
    Returns (delta_k5, delta_ind5).
    """
    # Current state of edge (u, v)
    is_edge = (masks[u] >> v) & 1
    
    # We need to iterate over all triples (a, b, c) from V \ {u, v}
    # For each triple, we determine if {u, v, a, b, c} is a K5 or Ind5
    # before and after flip.
    
    # To be efficient, we only care about triples that CAN form K5 or Ind5.
    # For K5: The triple must be a K3 in the subgraph induced by V\{u,v}
    #          AND u,v must be connected to all 3 (if edge u-v exists, or will exist?)
    # Actually, the status of the triple's internal edges doesn't change.
    # Let T = {a, b, c}.
    # Internal edges of T are fixed.
    # Let E_T be the number of edges in T.
    # If T is not a K3 (E_T < 3), it can never be part of a K5.
    # If T is not an Ind3 (E_T > 0), it can never be part of an Ind5.
    
    # So we can precompute for the current graph:
    # K3s: list of triples that are K3s.
    # Ind3s: list of triples that are Ind3s.
    
    # However, the set of K3s/Ind3s changes as we flip edges?
    # No, we are flipping (u, v). The internal edges of T (a,b,c) are NOT (u,v).
    # So the set of K3s in V\{u,v} is FIXED for this flip.
    
    # So, for a specific flip (u, v):
    # 1. Identify all triples T in V\{u,v} that are K3s.
    # 2. Identify all triples T in V\{u,v} that are Ind3s.
    
    # How to find them fast?
    # We can just iterate all triples? Binom(36,3) = 7140.
    # For each triple, check if it's K3 or Ind3.
    # Then check connections to u, v.
    
    delta_k5 = 0
    delta_ind5 = 0
    
    # Precompute masks for u and v
    mask_u = masks[u]
    mask_v = masks[v]
    
    # Iterate over all pairs (a, b) with a < b, a,b != u,v
    # Then find c > b, c != u,v
    # This is O(n^3). 36^3 ~ 46k.
    # Inside, we do bit checks.
    
    # Optimization:
    # We can iterate over all triples (a,b,c) from 0..n-1, skipping u,v.
    
    # Let's collect the triples first? No, just loop.
    
    # To speed up, we can precompute the list of K3s and Ind3s for the CURRENT graph
    # excluding u and v?
    # That's expensive to recompute for every flip.
    
    # Alternative:
    # Just loop over all triples.
    # For each triple (a,b,c):
    #   Check if T is K3.
    #   Check if T is Ind3.
    #   If K3:
    #      Check if {u,v,a,b,c} is K5 before flip.
    #      Check if {u,v,a,b,c} is K5 after flip.
    #      Update delta.
    #   If Ind3:
    #      Check if {u,v,a,b,c} is Ind5 before flip.
    #      Check if {u,v,a,b,c} is Ind5 after flip.
    #      Update delta.
    
    # Cost: 7140 triples * (check K3 + check Ind3 + check connections).
    # Check K3: 3 bit ops.
    # Check Ind3: 3 bit ops (or just not K3 and not edges? No, Ind3 means 0 edges).
    # Check connections: 6 bit ops.
    # Total ~ 12 ops per triple. 7140 * 12 ~ 85k ops.
    # In Python, this might be ~10-20ms.
    # 300s / 20ms = 15000 iterations.
    # This is quite low for SA.
    
    # We need to be faster.
    # Precompute K3s and Ind3s for the whole graph?
    # But they change.
    
    # What if we maintain a list of K3s and Ind3s dynamically?
    # When we flip (u, v), only K3s containing u or v change.
    # Number of K3s containing u is O(n^2).
    # So updating the list is O(n^2).
    # Then for the flip, we only iterate over K3s that are in V\{u,v}?
    # No, the K3s in V\{u,v} are the ones that DON'T contain u or v.
    # The set of K3s in V\{u,v} is the Global K3s minus those containing u or v.
    # If we maintain Global K3s, we can filter.
    
    # Let's try to optimize the triple iteration.
    # Instead of looping 0..n, we can use bitmasks to find common neighbors.
    
    # For K5 detection involving (u, v):
    # We need T={a,b,c} such that T is K3, and u,v connected to all.
    # This is equivalent to:
    # Intersection of N(u) and N(v) and N(T) ...
    
    # Let's stick to the simple loop but optimize it with local variables and minimal overhead.
    
    # We will iterate over a, b, c.
    # To avoid skipping logic, we can create a list of vertices V' = V \ {u, v}.
    # Then iterate over combinations of V'.
    
    V_prime = [i for i in range(n) if i != u and i != v]
    n_prime = len(V_prime)
    
    # Map original indices to new? No, just use original indices in bitmasks.
    
    # Precompute internal edge status for all pairs in V_prime?
    # That's O(n^2).
    
    # Let's just do the nested loops on V_prime.
    
    for i in range(n_prime):
        a = V_prime[i]
        for j in range(i+1, n_prime):
            b = V_prime[j]
            # Check if a~b
            ab_edge = (masks[a] >> b) & 1
            for k in range(j+1, n_prime):
                c = V_prime[k]
                # Check if a~c, b~c
                ac_edge = (masks[a] >> c) & 1
                bc_edge = (masks[b] >> c) & 1
                
                is_k3 = ab_edge and ac_edge and bc_edge
                is_ind3 = (not ab_edge) and (not ac_edge) and (not bc_edge)
                
                if not is_k3 and not is_ind3:
                    continue
                
                # Check connections to u and v
                # u~a, u~b, u~c
                ua = (mask_u >> a) & 1
                ub = (mask_u >> b) & 1
                uc = (mask_u >> c) & 1
                # v~a, v~b, v~c
                va = (mask_v >> a) & 1
                vb = (mask_v >> b) & 1
                vc = (mask_v >> c) & 1
                
                # Before flip:
                # Edge u-v exists? is_edge
                # If is_k3:
                #   K5 before = is_edge and ua and ub and uc and va and vb and vc
                # If is_ind3:
                #   Ind5 before = (not is_edge) and (not ua) and (not ub) and (not uc) and (not va) and (not vb) and (not vc)
                
                # After flip:
                # Edge u-v exists? not is_edge
                # If is_k3:
                #   K5 after = (not is_edge) and ua and ub and uc and va and vb and vc
                # If is_ind3:
                #   Ind5 after = is_edge and (not ua) and (not ub) and (not uc) and (not va) and (not vb) and (not vc)
                
                if is_k3:
                    conn_u = ua and ub and uc
                    conn_v = va and vb and vc
                    if conn_u and conn_v:
                        if is_edge:
                            delta_k5 -= 1 # Was K5, now not
                        else:
                            delta_k5 += 1 # Was not K5, now K5
                            
                if is_ind3:
                    conn_u = (not ua) and (not ub) and (not uc)
                    conn_v = (not va) and (not vb) and (not vc)
                    if conn_u and conn_v:
                        if is_edge:
                            delta_ind5 += 1 # Was Ind5, now not? 
                            # Wait. If is_edge is True, before: u-v edge exists.
                            # Ind5 requires NO edges. So u-v edge means NOT Ind5.
                            # After flip: u-v edge removed.
                            # If all other edges are missing, it BECOMES Ind5.
                            # So delta_ind5 += 1.
                        else:
                            delta_ind5 -= 1 # Was Ind5, now not
                            
    return delta_k5, delta_ind5

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Try n = 38, 39, 40
    # Start with 38
    
    for target_n in [38, 39, 40]:
        if time.time() > deadline - 30:
            break
            
        best_edges = None
        
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            # Initialize random graph
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.5:
                        edges.append((i, j))
            
            masks = _build_bitmasks(target_n, edges)
            k5, ind5 = _count_k5_and_ind5(target_n, masks)
            penalty = k5 + ind5
            
            T = 1.0
            min_T = 0.01
            cool_rate = 0.99
            
            max_iters = 2000
            iters = 0
            stuck_count = 0
            
            while iters < max_iters and time.time() < deadline - 5:
                iters += 1
                
                if penalty == 0:
                    best_edges = edges
                    break
                    
                # Pick random edge
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                    
                # Calculate delta
                dk5, dind5 = _calculate_delta_penalty(target_n, masks, u, v)
                delta = dk5 + dind5
                
                # Accept?
                if delta <= 0:
                    accept = True
                else:
                    if T > 0:
                        accept = random.random() < math.exp(-delta / T)
                    else:
                        accept = False
                        
                if accept:
                    # Flip edge
                    if (masks[u] >> v) & 1:
                        # Remove edge
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        edges.remove((u, v))
                    else:
                        # Add edge
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        edges.append((u, v))
                        
                    penalty += delta
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 500:
                        break
                        
                T *= cool_rate
                
            if best_edges is not None:
                break
                
        if best_edges is not None:
            return target_n, best_edges
            
    # Fallback: Return Paley 37 if we fail to find 38
    # Generate Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges
