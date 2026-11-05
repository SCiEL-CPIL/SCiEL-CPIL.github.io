"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Precompute neighbor masks for each vertex
    # neighbors_mask[v] is a bitmask where bit u is set if u is a neighbor of v
    neighbors_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            mask |= 1 << (v ^ (1 << b))
        neighbors_mask[v] = mask
    
    best_path = None
    best_len = 0
    
    def greedy_extend(start, rng, max_time):
        """
        Greedy extension using bitmasks.
        Returns the path list.
        """
        path = [start]
        used = 1 << start
        excluded = 0
        cur = start
        
        while time.time() < max_time:
            # Candidates are neighbors of cur that are not used and not excluded
            cands = neighbors_mask[cur] & ~(used | excluded)
            
            if cands == 0:
                break
            
            # Pick a random candidate
            # Convert bitmask to list of indices or pick a random set bit
            # Efficient way: pick a random position from the set bits
            # Since N=9, max 9 bits set.
            # We can iterate through bits.
            next_v = -1
            # Randomize order of checking bits to avoid bias
            # Or just pick a random set bit.
            # Let's pick a random set bit.
            # cands is an int.
            # We can use a loop to find set bits and pick one.
            # Or, simpler: iterate 9 bits, check if set, collect valid, pick random.
            valid_bits = []
            c = cands
            while c:
                lsb = c & -c
                b = lsb.bit_length() - 1
                valid_bits.append(b)
                c ^= lsb
            
            if not valid_bits:
                break
                
            idx = rng.randint(0, len(valid_bits) - 1)
            b = valid_bits[idx]
            next_v = cur ^ (1 << b)
            
            path.append(next_v)
            used |= (1 << next_v)
            # Update excluded: add neighbors of the vertex we are leaving (cur)
            # Because future vertices cannot be adjacent to cur
            excluded |= neighbors_mask[cur]
            cur = next_v
            
        return path

    def backtrack_extend(path, used, excluded, cur, depth, rng, max_time):
        """
        Limited backtracking.
        If greedy fails, try to backtrack a few steps and explore alternatives.
        """
        if time.time() > max_time:
            return path, used, excluded, cur, False
        
        if depth > 5: # Limit backtracking depth
            return path, used, excluded, cur, False
            
        # Try to find a valid next step
        cands = neighbors_mask[cur] & ~(used | excluded)
        
        if cands == 0:
            # Dead end. Backtrack.
            if len(path) < 2:
                return path, used, excluded, cur, False
            
            # Remove last vertex
            last_v = path.pop()
            used ^= (1 << last_v)
            # Restore excluded: remove neighbors of the new cur (which was last_v's predecessor)
            # Wait, `excluded` contains neighbors of v_0...v_{k-1}.
            # If we pop v_k, the new cur is v_{k-1}.
            # The `excluded` state currently contains N(v_{k-1})?
            # Let's trace:
            # State before adding v_k: excluded = N(v_0...v_{k-2}).
            # We added v_k. State became: excluded = N(v_0...v_{k-1}).
            # Now we pop v_k. We want to go back to state before adding v_k.
            # So we need to remove N(v_{k-1}) from excluded.
            # But we don't have a separate mask for just N(v_{k-1}) easily if we OR'd it.
            # However, we can just re-calculate or store history.
            # For simplicity in this structural change, we will stick to simple greedy restarts 
            # or a very shallow backtracking that re-computes state.
            # Actually, re-computing excluded for a short path is fast.
            
            # Let's just return failure for now and rely on restarts, 
            # or implement a simpler backtracking that stores state snapshots.
            # Given the speed of greedy, many restarts might be better than complex backtracking.
            
            return path, used, excluded, cur, False

        # If we have candidates, this function isn't really doing backtracking yet.
        # Let's just do greedy in the main loop.
        return path, used, excluded, cur, True

    # Main search loop
    for restart in range(10000):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 1)
        
        # Start from 0 or random
        if restart == 0:
            start = 0
        else:
            start = rng.randint(0, (1 << N) - 1)
        
        # Run greedy extension
        path = greedy_extend(start, rng, deadline - 2.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        # Optional: Try to extend from the end of a good path with different random seeds
        # If we have time, we can try "trim and re-extend"
        if time.time() < deadline - 5.0 and best_len > 100:
             # Take the best path found so far
             # Trim last few vertices and try to re-extend with different RNG
             for trim in range(1, 4):
                 if time.time() > deadline - 5.0:
                     break
                 if len(best_path) <= trim:
                     continue
                 trimmed_path = best_path[:-trim]
                 # Re-run greedy from the end of trimmed path
                 # We need to reconstruct used/excluded for the trimmed path
                 # This is O(L^2) if done naively, but L is small (~150)
                 # Or we can just start greedy from the last vertex of trimmed_path
                 # But we need the correct 'excluded' state.
                 
                 # Reconstructing state:
                 used = 0
                 for v in trimmed_path:
                     used |= 1 << v
                 excluded = 0
                 # Excluded is neighbors of all vertices except the last one
                 # Wait, the definition was: excluded = N(v_0 ... v_{k-1}) when at v_k.
                 # If path is [v_0, ..., v_m], and we are at v_m.
                 # excluded = N(v_0 ... v_{m-1}).
                 for v in trimmed_path[:-1]:
                     excluded |= neighbors_mask[v]
                     
                 cur = trimmed_path[-1]
                 
                 # Now extend from this state
                 # We can just call a modified greedy that takes initial state
                 # Or simpler: just run greedy_extend from cur, but it resets used/excluded.
                 # So we need a function that accepts initial state.
                 
                 # Let's just do a quick local search or restart from this point.
                 # To keep it simple and fast, we'll just do a few random restarts from the end of the best path.
                 # This is effectively "restart with bias".
                 
                 # Actually, the simplest improvement is just more restarts.
                 # The bit-parallel speed allows ~10k-50k restarts in 140s.
                 # Let's just ensure the loop runs as fast as possible.
                 pass

    if best_path is None:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
