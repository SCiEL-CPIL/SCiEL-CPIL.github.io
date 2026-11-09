"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # We aim for the maximum known n, which is 34.
    # If we fail, we step down.
    for n in range(34, 4, -1):
        if time.time() > deadline - 5:
            break
            
        # If we are trying a smaller n, we can be faster or use a simpler init
        # But let's stick to the robust local search for all n >= 20
        
        # For very small n, random search might suffice, but local search is fine.
        
        # 1. Setup
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # 2. Precompute dependencies
        # For each triple, which 4-sets contain it?
        # A 4-set is defined by 4 vertices.
        # We need to map 4-sets to an ID to store counts.
        # Instead of mapping 4-sets to IDs, we can just store the counts in a dict or array.
        # Number of 4-sets is C(n,4). For n=34, this is ~46k.
        # Number of 5-sets is C(n,5). For n=34, this is ~278k.
        
        # Let's use arrays for counts.
        # We need a mapping from a 4-set (tuple of 4 ints) to an index.
        # This is expensive to build for every n.
        # Alternative: Use the triple index as the key? No, counts are per set.
        
        # Efficient approach:
        # Store for each triple index `t_idx`:
        #   list of 4-set IDs containing it
        #   list of 5-set IDs containing it
        
        # To get 4-set IDs, we can just use the tuple itself as a key in a dict,
        # but accessing dict keys in a tight loop is slow.
        # Better: Precompute a list of 4-sets and 5-sets, and map them to indices.
        
        all_4sets = list(combinations(range(n), 4))
        all_5sets = list(combinations(range(n), 5))
        
        # Map set tuple to index
        # This is O(C(n,4)) and O(C(n,5)) which is fine for n=34.
        # 46k and 278k entries.
        
        # However, building the reverse map (triple -> sets) is the costly part.
        # For each 4-set, it has 4 triples.
        # For each 5-set, it has 10 triples.
        
        # Let's initialize the lists
        triple_4sets = [[] for _ in range(num_triples)]
        triple_5sets = [[] for _ in range(num_triples)]
        
        # We don't even need to map 4-sets to unique IDs if we just store the counts in a dict keyed by the tuple.
        # But updating a dict in a tight loop is slower than updating a list.
        # Let's try to use lists.
        
        # Map 4-set tuple to index
        # This might take a bit of time for n=34.
        # Let's estimate: 46376 tuples.
        
        # Actually, we can avoid global set IDs by just iterating over the specific sets affected.
        # When flipping triple t, we need to update the count for every 4-set containing t.
        # We can just store the 4-sets themselves in the list.
        
        # Re-building dependencies:
        for s in all_4sets:
            w, x, y, z = s
            t1 = (w, x, y)
            t2 = (w, x, z)
            t3 = (w, y, z)
            t4 = (x, y, z)
            triple_4sets[triple_to_idx[t1]].append(s)
            triple_4sets[triple_to_idx[t2]].append(s)
            triple_4sets[triple_to_idx[t3]].append(s)
            triple_4sets[triple_to_idx[t4]].append(s)
            
        for s in all_5sets:
            w, x, y, z, v = s
            triples_in_5 = [
                (w,x,y), (w,x,z), (w,x,v), (w,y,z), (w,y,v),
                (w,z,v), (x,y,z), (x,y,v), (x,z,v), (y,z,v)
            ]
            for t in triples_in_5:
                triple_5sets[triple_to_idx[t]].append(s)
                
        # Now we have the dependencies.
        # We need to track counts for each 4-set and 5-set.
        # Since we are using the tuple as the key in the lists, we can use a dictionary for counts.
        # red_count_4: dict {tuple: int}
        # blue_count_5: dict {tuple: int}
        
        # Initialize colors
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Initialize counts
        red_count_4 = {s: 0 for s in all_4sets}
        blue_count_5 = {s: 0 for s in all_5sets}
        
        # Calculate initial counts
        for t_idx, t in enumerate(triple_list):
            c = colors[t_idx]
            if c == 0: # Red
                for s in triple_4sets[t_idx]:
                    red_count_4[s] += 1
            else: # Blue
                for s in triple_5sets[t_idx]:
                    blue_count_5[s] += 1
                    # Note: Blue count for 4-sets is not needed for red violation, 
                    # and Red count for 5-sets is not needed for blue violation.
                    # Wait, the violation for Blue K5 is "all 10 triples blue".
                    # So we need count of BLUE triples in each 5-set.
                    # The violation for Red K4 is "all 4 triples red".
                    # So we need count of RED triples in each 4-set.
                    
        # Calculate initial violations
        viol_4 = sum(1 for s in all_4sets if red_count_4[s] == 4)
        viol_5 = sum(1 for s in all_5sets if blue_count_5[s] == 10)
        total_viol = viol_4 + viol_5
        
        if total_viol == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Local Search
        # We will try to reduce total_viol.
        # To speed up, we can keep viol_4 and viol_5 separate.
        
        # Random restarts if stuck
        max_stagnation = 10000
        stagnation = 0
        best_total_viol = total_viol
        best_colors = colors.copy()
        
        # Time limit per n
        n_deadline = time.time() + (deadline - time.time()) / (34 - n + 1) # Rough distribution
        # Actually, just use the global deadline.
        
        while time.time() < deadline - 2:
            if total_viol == 0:
                break
                
            # Pick a random triple
            t_idx = random.randint(0, num_triples - 1)
            t = triple_list[t_idx]
            old_c = colors[t_idx]
            new_c = 1 - old_c
            
            delta_viol = 0
            
            # Update Red 4-set violations
            if old_c == 0: # Red -> Blue
                # red_count decreases by 1
                for s in triple_4sets[t_idx]:
                    cnt = red_count_4[s]
                    if cnt == 4:
                        delta_viol -= 1 # Violation removed
                    # If cnt was 3, it becomes 2, no change in violation count
            else: # Blue -> Red
                # red_count increases by 1
                for s in triple_4sets[t_idx]:
                    cnt = red_count_4[s]
                    if cnt == 3:
                        delta_viol += 1 # New violation created
                        # If cnt was 4, it becomes 5, no change (still violation, but count > 4)
                        # Wait, if cnt is 4, it's already a violation. Increasing to 5 doesn't change the fact it's a violation.
                        # So only 3->4 creates a violation.
            
            # Update Blue 5-set violations
            if old_c == 1: # Blue -> Red
                # blue_count decreases by 1
                for s in triple_5sets[t_idx]:
                    cnt = blue_count_5[s]
                    if cnt == 10:
                        delta_viol -= 1 # Violation removed
            else: # Red -> Blue
                # blue_count increases by 1
                for s in triple_5sets[t_idx]:
                    cnt = blue_count_5[s]
                    if cnt == 9:
                        delta_viol += 1 # New violation created
            
            if delta_viol < 0:
                # Apply flip
                colors[t_idx] = new_c
                
                # Update counts
                if new_c == 0: # Became Red
                    for s in triple_4sets[t_idx]:
                        red_count_4[s] += 1
                else: # Became Blue
                    for s in triple_5sets[t_idx]:
                        blue_count_5[s] += 1
                # Note: We don't need to update the other count (e.g. blue_count_4) because we don't track them.
                # But we DO need to update the count that we ARE tracking for the NEW color?
                # No, we only track red_count_4 and blue_count_5.
                # If it became Red, we updated red_count_4.
                # If it became Blue, we updated blue_count_5.
                
                total_viol += delta_viol
                stagnation = 0
                
                if total_viol < best_total_viol:
                    best_total_viol = total_viol
                    best_colors = colors.copy()
            else:
                stagnation += 1
                # If stagnation is too high, we might be stuck.
                # We can force a random move or restart.
                # For now, let's just keep trying. The random selection usually escapes shallow minima.
                # If we really get stuck, we can break and restart with a new seed.
                if stagnation > max_stagnation:
                    # Restart local search with new random colors
                    colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                    red_count_4 = {s: 0 for s in all_4sets}
                    blue_count_5 = {s: 0 for s in all_5sets}
                    
                    for t_idx, t in enumerate(triple_list):
                        c = colors[t_idx]
                        if c == 0:
                            for s in triple_4sets[t_idx]:
                                red_count_4[s] += 1
                        else:
                            for s in triple_5sets[t_idx]:
                                blue_count_5[s] += 1
                                
                    total_viol = sum(1 for s in all_4sets if red_count_4[s] == 4) + \
                                 sum(1 for s in all_5sets if blue_count_5[s] == 10)
                    stagnation = 0
                    best_total_viol = total_viol
                    best_colors = colors.copy()
                    
        # After time limit for this n, check if we found a valid one
        if total_viol == 0:
            return {'n': n, 'colors': colors.tolist()}
        elif best_total_viol == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        
        # If we couldn't find a valid coloring for this n, try smaller n
        # But we must ensure we return SOMETHING valid.
        # The loop continues to n-1.
        
    # Fallback: If we ran out of time trying large n, we must return a valid small n
    # We know n=5 is always possible (e.g. all red, no blue K5, but check red K4? 
    # For n=5, red K4 is possible. 
    # Let's just return a known valid small coloring if we fail.
    # Actually, the loop goes down to 5.
    # If we fail for n=5 (unlikely), we return a default.
    
    # A valid coloring for n=5:
    # Avoid Red K4 and Blue K5.
    # If we color all triples Red, we have Red K4s.
    # If we color all triples Blue, we have Blue K5 (only 1 such set, the whole set).
    # So for n=5, we need to break the Blue K5 and Red K4s.
    # Simple strategy for small n: Random search is fast.
    
    # Since the loop iterates down, if we reach n=5 and fail, we return a hardcoded valid one.
    # But we should rarely reach n=5.
    
    # Construct a valid coloring for n=10 as a safe fallback?
    # Or just return the best n we found with 0 violations?
    # The loop breaks if time is up.
    
    # If we exit the loop without returning, we must return a valid coloring.
    # Let's ensure we always return a valid one.
    # We can store the best valid coloring found in any iteration.
    
    # Re-structure to store best valid found.
    # Since the current structure tries n=34 first, and if it fails, tries 33, etc.
    # If it fails for all, we need a fallback.
    
    # For safety, let's implement a very fast random search for n=10 as a fallback.
    # Or just return n=5 with a specific pattern.
    
    # Let's assume the local search finds n=34.
    # If not, it finds n=33, etc.
    
    # Fallback for absolute safety (should not be hit):
    n = 5
    # All blue except one red triple to break K5?
    # If 1 triple is red, no Blue K5 (needs 10 blue).
    # But we might have Red K4? No, only 1 red triple.
    # So: 1 red, 9 blue is valid for n=5.
    colors = [1] * 10
    colors[0] = 0
    return {'n': 5, 'colors': colors}

# EVOLVE-BLOCK-END
