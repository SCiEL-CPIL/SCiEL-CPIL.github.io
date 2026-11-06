"""Rewritten program using a bitmask-based local search with swap operations."""
import math

# EVOLVE-BLOCK-START
def construct():
    N = 256
    
    # Precompute conflict masks
    # adj[i] is a bitmask of nodes conflicting with i
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Initial Set: All Odds
    # Indices 0, 2, 4... correspond to numbers 1, 3, 5...
    S = 0
    for i in range(N):
        if (i + 1) % 2 != 0:
            S |= (1 << i)
            
    # Function to get list of indices from bitmask
    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]
        
    # Greedy Addition
    changed = True
    while changed:
        changed = False
        # Find all nodes not in S that don't conflict with S
        # safe = ~S & ~conflict_with_S
        # conflict_with_S = OR of adj[i] for i in S? No.
        # A node u is safe if (adj[u] & S) == 0.
        
        # We can iterate through all nodes not in S
        not_in_S = ((1 << N) - 1) ^ S
        # Filter not_in_S to find safe nodes
        # This is O(N^2) if done naively.
        # Better: Iterate through bits of not_in_S
        
        temp = not_in_S
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            if (adj[idx] & S) == 0:
                S |= lsb
                changed = True
                # Update not_in_S? Or just break and restart?
                # Breaking and restarting is simpler and ensures we pick the best next one?
                # Actually, adding one might block others.
                # Greedy: Add first found.
                break
            temp ^= lsb
            
    # Local Search with Swaps
    # We try to remove one node from S and add multiple nodes not in S.
    # To speed up, we only consider removing nodes that have high "cost" (conflict with many potential adds)?
    # Or just iterate all.
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_move = None # (remove_idx, add_mask)
        
        # Iterate through all nodes in S
        s_temp = S
        while s_temp:
            lsb = s_temp & -s_temp
            v_idx = lsb.bit_length() - 1
            
            # Temporarily remove v
            S_without_v = S ^ lsb
            
            # Find all nodes u not in S_without_v that are safe
            # u must not be in S_without_v
            # u must not conflict with S_without_v
            # Note: u can be v? No, v is removed, we want to add OTHERS.
            # We want to add a set A such that A is independent and disjoint from S_without_v
            # and no conflicts with S_without_v.
            
            # Candidates: All nodes not in S (including v? No, v is removed, so v is a candidate for re-adding? 
            # If we remove v and add v back, gain is 0. We want gain > 0.
            # So we look for nodes in ~S that are safe in S_without_v.
            
            candidates = 0
            not_in_S = ((1 << N) - 1) ^ S
            # We can add v back? If we add v back, we are back to S.
            # We want to add nodes from ~S.
            
            c_temp = not_in_S
            while c_temp:
                lsb_c = c_temp & -c_temp
                u_idx = lsb_c.bit_length() - 1
                
                # Check if u is safe in S_without_v
                if (adj[u_idx] & S_without_v) == 0:
                    # Also, if we add multiple, they must not conflict with each other.
                    # For now, let's just find all individually safe nodes.
                    # Then we need to pick an independent subset of these candidates.
                    # This is a subproblem.
                    # Heuristic: Just take all? No, they might conflict.
                    # Let's just add the first few?
                    # To keep it simple and fast, let's assume we add all candidates that are mutually non-conflicting?
                    # That's hard.
                    
                    # Simple Heuristic:
                    # If we remove v, we can add u if u is safe.
                    # If we can add more than 1, we have a gain.
                    # Let's collect all safe candidates.
                    candidates |= lsb_c
                c_temp ^= lsb_c
            
            # Now, from 'candidates', we want to pick an independent set.
            # Greedy Independent Set on candidates:
            # Sort by degree?
            # Just iterate and add if safe.
            
            add_set = 0
            c_temp = candidates
            # To make it deterministic and fast, we can just try to add them one by one.
            # But the order matters.
            # Let's just count the max independent set in candidates?
            # Too slow.
            
            # Heuristic: If candidates has > 1 bit, we likely have a gain.
            # Let's try to construct a valid add_set.
            # We'll just pick the first one that fits, then the next, etc.
            
            # Actually, a simpler move:
            # Remove v. Add u.
            # Gain = 1 (add) - 1 (remove) = 0. No gain.
            # We need to add > 1 nodes.
            
            # Let's try to add ALL candidates that are mutually non-conflicting.
            # Greedy MIS on candidates:
            greedy_add = 0
            c_temp = candidates
            while c_temp:
                lsb_c = c_temp & -c_temp
                u_idx = lsb_c.bit_length() - 1
                # Check if u is safe with current greedy_add
                if (adj[u_idx] & greedy_add) == 0:
                    greedy_add |= lsb_c
                c_temp ^= lsb_c
                
            added_count = bin(greedy_add).count('1')
            
            if added_count > 1:
                # Gain = added_count - 1
                if added_count - 1 > best_gain:
                    best_gain = added_count - 1
                    best_move = (v_idx, greedy_add)
            
            s_temp ^= lsb
            if best_gain > 0:
                # We found a good move, we can break early?
                # Or find the best one?
                # Let's continue to find the best one in this pass.
                pass
                
        if best_move is not None:
            v_idx, add_mask = best_move
            # Apply move
            S = (S ^ (1 << v_idx)) | add_mask
            improved = True
            
    return mask_to_list(S)
# EVOLVE-BLOCK-END
