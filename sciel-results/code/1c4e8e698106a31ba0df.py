import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 55.0  # Keep 5s margin for safety against 60s limit

    # Candidate pool: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # x must be divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates (it is, as 255 % 3 == 0)
    if 255 not in candidates:
        candidates.append(255)
        
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # Precompute adjacency as bitmasks
    # adj_mask[i] has bit j set if gcd(candidates[i], candidates[j]) > 1
    adj_mask = [0] * n
    for i in range(n):
        vi = candidates[i]
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            vj = candidates[j]
            if gcd(vi, vj) > 1:
                mask_i |= (1 << j)
        adj_mask[i] = mask_i

    best_clique_mask = 0
    best_clique_size = 0

    # Bron-Kerbosch with pivot selection using bitmasks
    # R: current clique
    # P: candidates to add
    # X: excluded candidates (already processed)
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_mask, best_clique_size
        
        if time.time() > deadline:
            return
            
        # Bound: if current size + max possible extension <= best, prune
        if R.bit_count() + P.bit_count() <= best_clique_size:
            return
            
        if P == 0 and X == 0:
            if R.bit_count() > best_clique_size:
                best_clique_size = R.bit_count()
                best_clique_mask = R
            return
        
        # Pivot selection: pick u in P | X with max |P & adj_mask[u]|
        # We want to minimize branching, so we branch on P \ N(u)
        # If we pick u with max neighbors in P, P \ N(u) is minimized
        pivot = 0
        max_count = -1
        # Iterate over bits in P | X
        temp = P | X
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            
            # Count neighbors of u in P
            count = (P & adj_mask[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                
        # Branch on P \ N(pivot)
        # We need to iterate over vertices in P that are NOT neighbors of pivot
        # These are in P & ~adj_mask[pivot]
        # Note: pivot itself is in P or X. If pivot in P, it is in P & ~adj_mask[pivot] only if not self-loop (no self loops in adj_mask)
        # Wait, adj_mask[u] does not include u. So u is in P & ~adj_mask[u] if u is in P.
        # This is correct. We branch on all v in P such that v is not neighbor of pivot.
        
        # To optimize, we can order the vertices we branch on.
        # Vertices with higher degree in P are more likely to be in large cliques.
        # However, sorting at each step might be expensive.
        # Let's just iterate.
        
        # We need to process vertices one by one, removing them from P and adding to X.
        # With bitmasks, we can't easily "remove" one bit and keep the rest for the loop while modifying global state?
        # Actually, we can extract the set of vertices to branch on:
        # branch_set = P & ~adj_mask[pivot]
        
        # We iterate over bits in branch_set
        # For each v:
        #   new_R = R | (1 << v)
        #   new_P = P & adj_mask[v]
        #   new_X = X & adj_mask[v]
        #   recurse
        #   P = P & ~(1 << v)
        #   X = X | (1 << v)
        
        branch_set = P & ~adj_mask[pivot]
        
        while branch_set:
            if time.time() > deadline:
                return
            
            # Extract one vertex v
            lsb = branch_set & -branch_set
            v = lsb.bit_length() - 1
            branch_set ^= lsb
            
            v_bit = 1 << v
            
            new_R = R | v_bit
            new_P = P & adj_mask[v]
            new_X = X & adj_mask[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Update P and X for the next iteration of the loop
            P &= ~v_bit
            X |= v_bit

    # Initialize
    # R = 0
    # P = all vertices
    # X = 0
    all_vertices_mask = (1 << n) - 1
    bron_kerbosch(0, all_vertices_mask, 0)
    
    # Convert best_clique_mask back to list of values
    if best_clique_mask != 0:
        result_vals = []
        temp_mask = best_clique_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            result_vals.append(candidates[idx])
            temp_mask ^= lsb
    else:
        # Fallback: multiples of 3
        result_vals = [i for i in range(1, 256) if i % 3 == 0]
        
    # Ensure 255 is in the result
    if 255 not in result_vals:
        # If 255 is not in the max clique found (which shouldn't happen as it's universal),
        # add it if it doesn't break the clique property? 
        # Since 255 is universal in the candidate pool, adding it to any clique of candidates is valid.
        # But if the max clique found was somehow not including it due to a bug, 
        # and we add it, is it still a clique? Yes, because it's connected to all.
        # However, the max clique MUST contain it. 
        # If best_clique_size is 0, we use fallback.
        # If best_clique_size > 0, 255 should be there.
        # Let's double check.
        if 255 in candidates:
             idx_255 = idx_map[255]
             if not (best_clique_mask & (1 << idx_255)):
                 # This indicates a potential issue or that the max clique found is not maximal?
                 # Or maybe the graph construction missed something?
                 # For safety, if we have a valid clique, we can add 255.
                 # But we must ensure the result is a clique.
                 # Since 255 is universal, S U {255} is a clique if S is a clique.
                 # So we can just append it.
                 result_vals.append(255)

    # Final verification
    # Check pairwise gcd > 1
    valid = True
    for i in range(len(result_vals)):
        for j in range(i+1, len(result_vals)):
            if gcd(result_vals[i], result_vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to known good set
        result_vals = [i for i in range(1, 256) if i % 3 == 0]
        
    return result_vals
