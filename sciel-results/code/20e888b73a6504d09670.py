import time
import itertools
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    m = len(all_triples)
    
    # Map triple to index for faster lookup? 
    # Or just use the tuple itself.
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        ps = [
            tuple(sorted((t[0], t[1]))),
            tuple(sorted((t[0], t[2]))),
            tuple(sorted((t[1], t[2])))
        ]
        triple_pairs.append(ps)
        
    # Precompute triple masks
    triple_masks = []
    for t in all_triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
        
    # Precompute supersets for each triple mask
    # sup[triple_idx] = list of masks that contain this triple
    # There are 2^(16-3) = 8192 supersets for each triple
    # Total memory: 560 * 8192 ints. 560 * 8192 * 4 bytes ~ 18 MB. Acceptable.
    # But generating them takes time.
    # We can generate them on the fly or precompute.
    # Let's precompute a list of masks for each triple index.
    triple_supersets = []
    for tm in triple_masks:
        # Find all masks that are supersets of tm
        # A mask S is a superset of T if (S & T) == T
        # We can iterate over all masks 0..2^16-1? 560 * 65536 = 36M checks. Too slow for precomp.
        # Better: iterate over submasks of the complement?
        # Let C = (~tm) & ((1<<16)-1)
        # Supersets are tm | sub for sub in submasks(C)
        # Number of submasks is 2^(16-3) = 8192.
        # 560 * 8192 = 4.5M operations. Fast.
        comp = (~tm) & ((1 << n) - 1)
        subs = []
        sub = comp
        while True:
            subs.append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & comp
        # Convert to full masks
        sups = [tm | s for s in subs]
        triple_supersets.append(sups)
        
    # Precompute masks by size for alpha calculation?
    # Or just keep cover_count and check.
    # We will maintain cover_count array of size 2^16.
    
    best_triples = []
    best_alpha = n
    best_cover_count = None
    
    # Helper to calculate alpha from cover_count
    def calc_alpha(cc):
        max_size = 0
        # Iterate over all masks
        # Optimization: check larger sizes first?
        # We can just find the max popcount of any mask with cc[mask]==0
        # 65536 iterations.
        for mask in range(1 << n):
            if cc[mask] == 0:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
        return max_size
    
    # Greedy initialization
    def greedy_init():
        used_pairs = set()
        packing = []
        candidates = list(range(m))
        random.shuffle(candidates)
        for idx in candidates:
            ps = triple_pairs[idx]
            valid = True
            for p in ps:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                packing.append(idx)
                for p in ps:
                    used_pairs.add(p)
        return packing
    
    # Local Search / Simulated Annealing
    def local_search(initial_packing, time_limit):
        packing = initial_packing[:]
        used_pairs = set()
        for idx in packing:
            for p in triple_pairs[idx]:
                used_pairs.add(p)
                
        # Initialize cover_count
        cc = [0] * (1 << n)
        for idx in packing:
            sups = triple_supersets[idx]
            for s in sups:
                cc[s] += 1
                
        current_alpha = calc_alpha(cc)
        best_alpha_local = current_alpha
        best_packing_local = packing[:]
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        no_improve = 0
        
        while time.time() < time_limit and T > min_T:
            if len(packing) == 0:
                break
                
            # Pick a random triple to remove
            idx_remove = random.randint(0, len(packing) - 1)
            t_remove_idx = packing[idx_remove]
            
            # Remove it
            packing.pop(idx_remove)
            ps_rem = triple_pairs[t_remove_idx]
            for p in ps_rem:
                used_pairs.discard(p)
            sups_rem = triple_supersets[t_remove_idx]
            for s in sups_rem:
                cc[s] -= 1
                
            # Try to add a valid triple
            # Sample random triples until we find a valid one
            found = False
            for _ in range(50): # Limit attempts
                idx_add = random.randint(0, m - 1)
                if idx_add == t_remove_idx:
                    continue
                ps_add = triple_pairs[idx_add]
                valid = True
                for p in ps_add:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    # Check if it's the same triple (indices are unique, so no)
                    # Add it
                    packing.append(idx_add)
                    for p in ps_add:
                        used_pairs.add(p)
                    sups_add = triple_supersets[idx_add]
                    for s in sups_add:
                        cc[s] += 1
                    
                    new_alpha = calc_alpha(cc)
                    
                    if new_alpha < best_alpha_local:
                        best_alpha_local = new_alpha
                        best_packing_local = packing[:]
                        no_improve = 0
                        # Accept move
                        found = True
                        break
                    else:
                        # Simulated Annealing accept
                        delta = new_alpha - current_alpha
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            # Accept
                            current_alpha = new_alpha
                            found = True
                            break
                        else:
                            # Reject, revert
                            packing.pop()
                            for p in ps_add:
                                used_pairs.discard(p)
                            for s in sups_add:
                                cc[s] -= 1
                            
            if not found:
                # Revert removal
                packing.append(t_remove_idx)
                for p in ps_rem:
                    used_pairs.add(p)
                for s in sups_rem:
                    cc[s] += 1
                no_improve += 1
                if no_improve > 100:
                    break # Restart
                
            T *= cooling
            
        return best_packing_local, best_alpha_local

    # Main Loop
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        init_packing = greedy_init()
        
        # Calculate initial alpha
        # We can just pass to local search which does it
        
        # Allocate time for local search
        # Decrease time per restart if many restarts?
        # Or fixed time per restart?
        # Let's use a fixed budget per restart, decreasing as time goes on
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed
        if time_remaining < 10:
            break
            
        # Give each restart up to 5 seconds, or less if time is tight
        search_time = min(5.0, time_remaining - 5.0)
        
        try:
            improved_packing, improved_alpha = local_search(init_packing, time.time() + search_time)
        except Exception:
            continue
            
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            # Convert indices back to triples
            best_triples = [list(all_triples[i]) for i in improved_packing]
            
        if best_alpha <= 5:
            break

    return best_triples
