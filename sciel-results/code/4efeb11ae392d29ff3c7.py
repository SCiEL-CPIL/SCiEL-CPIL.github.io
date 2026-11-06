import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try n=34 first, then fall back
    for target_n in range(34, 29, -1):
        if time.time() > deadline - 10:
            break
            
        result = solve_for_n(target_n, deadline)
        if result is not None:
            return result
            
    # Fallback to a very small valid construction if all else fails
    # n=9 is small enough to be found instantly or hardcoded if needed, 
    # but let's just return a trivial valid one for n=4 (no K4 possible if n<4? No, K4 needs 4 vertices. 
    # If n=3, no K4 or K5. 
    # Let's return n=4 with all red? No, K4 is 4 vertices all red. 
    # So for n=4, we must have at least one blue. 
    # Actually, any coloring of n=3 is valid.
    n = 3
    triples = list(combinations(range(n), 3))
    return {'n': n, 'colors': [0] * len(triples)}

def solve_for_n(n, deadline):
    """
    Attempt to find a valid coloring for specific n using Min-Conflicts local search.
    """
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple -> index
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4 and K5 structures
    k4_triple_indices = []
    k4_map = {}
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_triple_indices.append(idxs)
        k4_map[quad] = len(k4_triple_indices) - 1

    k5_triple_indices = []
    k5_map = {}
    for quint in combinations(range(n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_triple_indices.append(idxs)
        k5_map[quint] = len(k5_triple_indices) - 1
        
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    # Reverse maps: triple -> list of K4 indices, triple -> list of K5 indices
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
            
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)

    # Local Search with Min-Conflicts
    max_restarts = 20
    max_iters_per_restart = 500000 # High limit, controlled by time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Initialize counts
        red_counts_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_counts_k5 = np.zeros(num_k5, dtype=np.int8)
        
        for k4_i, idxs in enumerate(k4_triple_indices):
            red_counts_k4[k4_i] = np.sum(colors[idxs] == 0)
            
        for k5_i, idxs in enumerate(k5_triple_indices):
            blue_counts_k5[k5_i] = np.sum(colors[idxs] == 1)
            
        # Identify active violations
        active_k4 = np.where(red_counts_k4 == 4)[0].tolist()
        active_k5 = np.where(blue_counts_k5 == 10)[0].tolist()
        
        # Min-Conflicts Loop
        iters = 0
        while active_k4 or active_k5:
            if time.time() > deadline - 5:
                break
                
            iters += 1
            if iters > max_iters_per_restart:
                break
                
            # Pick a violation
            # Weight by size? K5 is bigger, maybe harder? 
            # Just pick randomly from union
            if active_k4 and (not active_k5 or random.random() < 0.5):
                v_idx = random.choice(active_k4)
                is_k4 = True
            else:
                v_idx = random.choice(active_k5)
                is_k4 = False
                
            # Pick a random edge in this violation
            if is_k4:
                tri_idx = random.choice(k4_triple_indices[v_idx])
                # This edge is currently Red (0). Flip to Blue (1).
                # Update:
                # 1. The K4 itself: red_count 4 -> 3. Remove from active_k4.
                red_counts_k4[v_idx] = 3
                # Remove v_idx from active_k4
                # Efficient removal: swap with last and pop? Or just filter? 
                # Since we are in a loop, we can just mark it and filter periodically, 
                # or use a set for active violations.
                # Let's use lists and remove by value (O(N) but N is smallish ~40k)
                # Actually, active_k4 size can be large initially.
                # Better: maintain a set? Or just let it be a list and clean up?
                # For speed, let's just update counts and rebuild active lists every K steps?
                # No, that's slow.
                
                # Let's use a dictionary or set for active violations to allow O(1) removal.
                # But we are using lists for random choice.
                # Compromise: Use lists, but when removing, if the item is at the end, pop. 
                # Otherwise, swap with a random item? No, we need to remove specific index.
                # Let's just use a set for membership check and a list for random choice, 
                # and clean up the list when it gets too dirty?
                
                # Actually, for n=34, num_k4 ~ 46k. 
                # If we do O(N) removal, it's 46k ops per flip. 500k flips = 23e9 ops. Too slow.
                
                # We need O(1) or O(1) amortized removal.
                # We can maintain `active_k4` as a list, and `active_k4_set` as a set.
                # When removing, remove from set.
                # When picking random, pick from list.
                # If the picked item is not in set, it's stale. Pick again.
                # This is "lazy deletion".
                
                # Let's refactor to use lazy deletion.
                
            # --- Refactored Min-Conflicts with Lazy Deletion ---
            
        # The logic above needs to be wrapped in a proper function with lazy deletion structures.
        
    return None # Placeholder, will be implemented below

# Re-implementation with Lazy Deletion for efficiency

def solve_for_n(n, deadline):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    k4_triple_indices = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_triple_indices.append(idxs)
        
    k5_triple_indices = []
    for quint in combinations(range(n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_triple_indices.append(idxs)
        
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
            
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)

    max_restarts = 30
    max_iters_per_restart = 1000000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        colors = np.random.randint(0, 2, size=num_triples)
        red_counts_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_counts_k5 = np.zeros(num_k5, dtype=np.int8)
        
        for k4_i, idxs in enumerate(k4_triple_indices):
            red_counts_k4[k4_i] = np.sum(colors[idxs] == 0)
        for k5_i, idxs in enumerate(k5_triple_indices):
            blue_counts_k5[k5_i] = np.sum(colors[idxs] == 1)
            
        # Lazy deletion structures
        active_k4_list = np.where(red_counts_k4 == 4)[0].tolist()
        active_k4_set = set(active_k4_list)
        
        active_k5_list = np.where(blue_counts_k5 == 10)[0].tolist()
        active_k5_set = set(active_k5_list)
        
        iters = 0
        while active_k4_set or active_k5_set:
            if time.time() > deadline - 5:
                break
            iters += 1
            if iters > max_iters_per_restart:
                break
                
            # Pick a violation
            # If one set is empty, pick from the other
            if not active_k4_set:
                v_idx = random.choice(active_k5_list)
                if v_idx not in active_k5_set: continue # Stale
                is_k4 = False
            elif not active_k5_set:
                v_idx = random.choice(active_k4_list)
                if v_idx not in active_k4_set: continue # Stale
                is_k4 = True
            else:
                if random.random() < 0.5:
                    v_idx = random.choice(active_k4_list)
                    if v_idx not in active_k4_set: continue
                    is_k4 = True
                else:
                    v_idx = random.choice(active_k5_list)
                    if v_idx not in active_k5_set: continue
                    is_k4 = False
                    
            # Pick random edge in violation
            if is_k4:
                tri_idx = random.choice(k4_triple_indices[v_idx])
                # Flip Red (0) -> Blue (1)
                colors[tri_idx] = 1
                
                # Update K4s
                # The specific K4 v_idx: red_count 4 -> 3
                red_counts_k4[v_idx] = 3
                active_k4_set.discard(v_idx)
                
                # Other K4s containing tri_idx
                for k4_i in triple_to_k4s[tri_idx]:
                    if k4_i == v_idx: continue
                    old_count = red_counts_k4[k4_i]
                    new_count = old_count - 1
                    red_counts_k4[k4_i] = new_count
                    # If it was a violation (4), it's now 3, so remove from active
                    if old_count == 4:
                        active_k4_set.discard(k4_i)
                    # If it becomes a violation? No, red count decreased.
                    
                # Update K5s
                # Blue count increases for K5s containing tri_idx
                for k5_i in triple_to_k5s[tri_idx]:
                    old_count = blue_counts_k5[k5_i]
                    new_count = old_count + 1
                    blue_counts_k5[k5_i] = new_count
                    # If it becomes a violation (10), add to active
                    if new_count == 10:
                        if k5_i not in active_k5_set:
                            active_k5_set.add(k5_i)
                            active_k5_list.append(k5_i)
                            # Periodic cleanup of list to prevent it growing too large with stale entries
                            # But we only add when it becomes active.
                            # Stale entries are those that were active, then became non-active.
                            # We don't remove from list, just from set.
                            
            else: # is_k5
                tri_idx = random.choice(k5_triple_indices[v_idx])
                # Flip Blue (1) -> Red (0)
                colors[tri_idx] = 0
                
                # Update K5s
                # The specific K5 v_idx: blue_count 10 -> 9
                blue_counts_k5[v_idx] = 9
                active_k5_set.discard(v_idx)
                
                # Other K5s containing tri_idx
                for k5_i in triple_to_k5s[tri_idx]:
                    if k5_i == v_idx: continue
                    old_count = blue_counts_k5[k5_i]
                    new_count = old_count - 1
                    blue_counts_k5[k5_i] = new_count
                    if old_count == 10:
                        active_k5_set.discard(k5_i)
                        
                # Update K4s
                # Red count increases for K4s containing tri_idx
                for k4_i in triple_to_k4s[tri_idx]:
                    old_count = red_counts_k4[k4_i]
                    new_count = old_count + 1
                    red_counts_k4[k4_i] = new_count
                    if new_count == 4:
                        if k4_i not in active_k4_set:
                            active_k4_set.add(k4_i)
                            active_k4_list.append(k4_i)
                            
        if not active_k4_set and not active_k5_set:
            return {'n': n, 'colors': colors.tolist()}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    for target_n in range(34, 29, -1):
        if time.time() > deadline - 10:
            break
        result = solve_for_n(target_n, deadline)
        if result is not None:
            return result
            
    # Fallback
    n = 3
    triples = list(combinations(range(n), 3))
    return {'n': n, 'colors': [0] * len(triples)}
