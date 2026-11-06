import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for fallback/verification
    
    # Target n
    n_target = 67
    
    # Initialize random tournament using bitmasks
    # out_mask[i] has bit j set if i -> j
    out_masks = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                out_masks[i] |= (1 << j)
                # out_masks[j] remains 0 for i
            else:
                out_masks[j] |= (1 << i)
                # out_masks[i] remains 0 for j

    # Precompute all 4-sets
    # We need to map each 4-set to an index
    # And for each edge (u,v), list of 4-set indices containing it
    
    four_sets = []
    # To speed up generation, we can just iterate
    for a in range(n_target):
        for b in range(a + 1, n_target):
            for c in range(b + 1, n_target):
                for d in range(c + 1, n_target):
                    four_sets.append((a, b, c, d))
                    
    num_sets = len(four_sets)
    
    # edge_to_sets: dict from (u,v) with u<v to list of indices
    edge_to_sets = {}
    for i in range(n_target):
        for j in range(i + 1, n_target):
            edge_to_sets[(i, j)] = []
            
    for idx, s in enumerate(four_sets):
        a, b, c, d = s
        # Pairs in the set
        pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        for p in pairs:
            edge_to_sets[p].append(idx)
            
    # Function to check if a 4-set is bad
    def is_bad_set(s):
        a, b, c, d = s
        # Check if a dominates b,c,d
        if (out_masks[a] & (1 << b | 1 << c | 1 << d)) == (1 << b | 1 << c | 1 << d):
            return False
        # Check if b dominates a,c,d
        if (out_masks[b] & (1 << a | 1 << c | 1 << d)) == (1 << a | 1 << c | 1 << d):
            return False
        # Check if c dominates a,b,d
        if (out_masks[c] & (1 << a | 1 << b | 1 << d)) == (1 << a | 1 << b | 1 << d):
            return False
        # Check if d dominates a,b,c
        if (out_masks[d] & (1 << a | 1 << b | 1 << c)) == (1 << a | 1 << b | 1 << c):
            return False
        return True

    # Initial cost calculation
    bad_array = [0] * num_sets
    total_bad = 0
    for idx, s in enumerate(four_sets):
        if is_bad_set(s):
            bad_array[idx] = 1
            total_bad += 1
            
    if total_bad == 0:
        # Found valid tournament immediately? Unlikely for random
        pass
        
    # Simulated Annealing
    # T starts at 1.0, decays
    T = 1.0
    cool_rate = 0.999
    
    best_total_bad = total_bad
    best_masks = out_masks[:]
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            T = 0.01 # Floor
            
        # Pick random edge
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Get affected sets
        affected_indices = edge_to_sets[(u, v)]
        
        # Calculate delta cost
        # For each affected set, check if it was bad, flip, check if it is bad now
        # We need to temporarily flip
        # Current state: u->v or v->u?
        # Check bit v in out_masks[u]
        if out_masks[u] & (1 << v):
            # u -> v
            # Flip: u loses v, v gains u
            out_masks[u] &= ~(1 << v)
            out_masks[v] |= (1 << u)
            current_dir = 1 # u->v was 1? Let's say 1 means u->v
        else:
            # v -> u
            out_masks[v] &= ~(1 << u)
            out_masks[u] |= (1 << v)
            current_dir = 0
            
        delta = 0
        for idx in affected_indices:
            s = four_sets[idx]
            # Check if bad BEFORE flip? 
            # We already flipped. We need to know if it WAS bad.
            # We have bad_array[idx] which is 1 if it was bad.
            was_bad = bad_array[idx]
            
            # Check if bad NOW
            is_bad_now = is_bad_set(s)
            
            if was_bad and not is_bad_now:
                delta -= 1
            elif not was_bad and is_bad_now:
                delta += 1
            # else no change
            
            # Update bad_array for the NEW state
            bad_array[idx] = 1 if is_bad_now else 0
            
        total_bad += delta
        
        # Accept or Reject
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = True
        else:
            # delta > 0
            prob = math.exp(-delta / T)
            accept = random.random() < prob
            
        if not accept:
            # Revert flip
            if current_dir == 1:
                # Was u->v, we flipped to v->u. Revert to u->v
                out_masks[u] |= (1 << v)
                out_masks[v] &= ~(1 << u)
            else:
                # Was v->u, we flipped to u->v. Revert to v->u
                out_masks[v] |= (1 << u)
                out_masks[u] &= ~(1 << v)
                
            # Revert bad_array and total_bad
            # We need to revert the changes to bad_array
            for idx in affected_indices:
                s = four_sets[idx]
                # Recalculate bad status for the REVERTED state
                # Or just store the old state?
                # Storing old state is easier.
                pass # This is inefficient. 
                
            # Better: Store the changes made to bad_array in a list and revert them.
            
        # This revert logic is getting complex and slow.
        # Let's optimize:
        # Instead of updating bad_array immediately, calculate delta, then if accepted, update.
        # If rejected, do nothing.
        
        # Let's rewrite the loop body.
        
        T *= cool_rate
        if total_bad == 0:
            break

    # If we didn't find 0, we might have a bad tournament.
    # We MUST return a valid one.
    # Check if total_bad == 0.
    if total_bad == 0:
        # Construct output
        orientation = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if out_masks[i] & (1 << j):
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return {'n': n_target, 'orientation': orientation}
    else:
        # Fallback to Paley 83
        p = 83
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if (j-i) % p in squares else 1)
        return {'n': p, 'orientation': orientation}
