import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute pair masks for linearity check
    tpair = [0] * num_triples
    for i, (a, b, c) in enumerate(all_triples):
        pm = 0
        for x, y in ((a, b), (a, c), (b, c)):
            if x > y:
                x, y = y, x
            pm |= 1 << (x * n + y)
        tpair[i] = pm
        
    # Precompute popcount array for all 2^16 masks
    pop_arr = np.zeros(1 << n, dtype=np.int8)
    for i in range(1, 1 << n):
        pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        
    # Precompute subsets array for vectorized coverage check
    subsets = np.arange(1 << n, dtype=np.int32)
    
    # Precompute for each triple, the boolean array of which subsets contain it
    # This is expensive upfront but saves repeated computation
    # Actually, let's use a different approach: precompute the "supersets" list
    # For each triple mask, store the indices of subsets that contain it
    # But that's too much memory. Instead, let's use the vectorized approach but
    # optimize the inner loop.
    
    # Alternative: precompute a lookup: for each triple index, the list of subset
    # indices that contain it. This is 560 * (various sizes) which could be large.
    # Let's estimate: each triple has 2^13 = 8192 supersets. 560 * 8192 = 4.6M entries.
    # That's manageable as a list of arrays.
    
    triple_supersets = []
    for i in range(num_triples):
        t_mask = triple_masks[i]
        # Find all subsets that contain t_mask
        # A subset S contains T if (S & T) == T
        # We can generate this by: S = T | R where R is any subset of complement of T
        comp = ((1 << n) - 1) ^ t_mask
        supersets = []
        # Enumerate all submasks of comp
        r = comp
        while True:
            supersets.append(t_mask | r)
            if r == 0:
                break
            r = (r - 1) & comp
        triple_supersets.append(supersets)
    
    def compute_alpha_from_covered(covered):
        """Compute alpha from a precomputed covered boolean array."""
        max_alpha = int(np.max(pop_arr[~covered]))
        return max_alpha
    
    def make_covered(triple_indices):
        """Create the covered array from a list of triple indices."""
        covered = np.zeros(1 << n, dtype=bool)
        for idx in triple_indices:
            covered[triple_supersets[idx]] = True
        return covered
    
    def greedy_packing(order):
        """
        Greedily add triples in the given order if they are linear with existing ones.
        """
        sys_set = set()
        pair_union = 0
        for idx in order:
            if (tpair[idx] & pair_union) == 0:
                sys_set.add(idx)
                pair_union |= tpair[idx]
        return list(sys_set), pair_union
    
    # Initial solution: use multiple random restarts to find a good starting point
    best_packing = None
    best_alpha = n + 1
    best_covered = None
    
    for restart in range(20):
        order = list(range(num_triples))
        random.shuffle(order)
        packing, _ = greedy_packing(order)
        covered = make_covered(packing)
        alpha = compute_alpha_from_covered(covered)
        if alpha < best_alpha:
            best_alpha = alpha
            best_packing = list(packing)
            best_covered = covered
        if time.time() > deadline - 10:
            break
    
    if best_packing is None:
        # Fallback
        order = list(range(num_triples))
        random.shuffle(order)
        best_packing, _ = greedy_packing(order)
        best_covered = make_covered(best_packing)
        best_alpha = compute_alpha_from_covered(best_covered)
    
    # Local Search
    # State: current set of triple indices, pair_union mask, covered array, alpha
    current_packing = list(best_packing)
    current_set = set(current_packing)
    current_pair_union = 0
    for idx in current_packing:
        current_pair_union |= tpair[idx]
    current_covered = best_covered.copy()
    current_alpha = best_alpha
    
    # Track best
    global_best_packing = list(current_packing)
    global_best_alpha = current_alpha
    global_best_covered = current_covered.copy()
    
    def try_add_triple(cand):
        """Try to add a triple. Returns (success, new_covered, new_alpha) if successful."""
        if cand in current_set or (tpair[cand] & current_pair_union) != 0:
            return None
        
        # Compute new covered: OR in the supersets of cand
        new_covered = current_covered.copy()
        new_covered[triple_supersets[cand]] = True
        new_alpha = int(np.max(pop_arr[~new_covered]))
        
        if new_alpha <= current_alpha:
            return (new_covered, new_alpha)
        return None
    
    def try_remove_triple(rem):
        """Remove a triple and recompute covered from scratch (expensive but correct)."""
        new_packing = [x for x in current_packing if x != rem]
        new_covered = make_covered(new_packing)
        new_alpha = int(np.max(pop_arr[~new_covered]))
        return (new_packing, new_covered, new_alpha)
    
    step_count = 0
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.4:
            # Try to add a random valid triple
            cand = random.randrange(num_triples)
            if cand in current_set or (tpair[cand] & current_pair_union) != 0:
                continue
            
            result = try_add_triple(cand)
            if result is not None:
                new_covered, new_alpha = result
                current_packing.append(cand)
                current_set.add(cand)
                current_pair_union |= tpair[cand]
                current_covered = new_covered
                current_alpha = new_alpha
                if current_alpha < global_best_alpha:
                    global_best_alpha = current_alpha
                    global_best_packing = list(current_packing)
                    global_best_covered = current_covered.copy()
                    
        elif r < 0.8:
            # Swap: remove one, add one
            if not current_packing:
                continue
            rem = random.choice(current_packing)
            cand = random.randrange(num_triples)
            if cand in current_set or cand == rem:
                continue
            
            # Check linearity of new set: current_set - {rem} + {cand}
            conflict = False
            for other in current_set:
                if other == rem:
                    continue
                if (tpair[cand] & tpair[other]) != 0:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Remove rem, add cand
            # Recompute covered from scratch for correctness
            new_packing = [x for x in current_packing if x != rem] + [cand]
            new_covered = make_covered(new_packing)
            new_alpha = int(np.max(pop_arr[~new_covered]))
            
            if new_alpha <= current_alpha:
                current_packing = new_packing
                current_set.discard(rem)
                current_set.add(cand)
                current_pair_union = 0
                for idx in current_packing:
                    current_pair_union |= tpair[idx]
                current_covered = new_covered
                current_alpha = new_alpha
                if current_alpha < global_best_alpha:
                    global_best_alpha = current_alpha
                    global_best_packing = list(current_packing)
                    global_best_covered = current_covered.copy()
                    
        else:
            # Perturbation: Remove 1-3 random triples to escape local minima
            # Then greedily refill
            n_remove = random.randint(1, 3)
            removed = []
            for _ in range(n_remove):
                if current_packing:
                    rem = random.choice(current_packing)
                    current_packing.remove(rem)
                    current_set.discard(rem)
                    removed.append(rem)
            
            # Recompute pair_union
            current_pair_union = 0
            for idx in current_packing:
                current_pair_union |= tpair[idx]
            
            # Recompute covered
            current_covered = make_covered(current_packing)
            current_alpha = int(np.max(pop_arr[~current_covered]))
            
            # Greedily add back valid triples
            candidates = list(range(num_triples))
            random.shuffle(candidates)
            added_count = 0
            for cand in candidates:
                if cand in current_set or (tpair[cand] & current_pair_union) != 0:
                    continue
                
                # Try adding
                new_covered = current_covered.copy()
                new_covered[triple_supersets[cand]] = True
                new_alpha = int(np.max(pop_arr[~new_covered]))
                
                if new_alpha <= current_alpha:
                    current_packing.append(cand)
                    current_set.add(cand)
                    current_pair_union |= tpair[cand]
                    current_covered = new_covered
                    current_alpha = new_alpha
                    added_count += 1
                    if time.time() > deadline - 2:
                        break
            
            # After perturbation and refill, update best
            if current_alpha < global_best_alpha:
                global_best_alpha = current_alpha
                global_best_packing = list(current_packing)
                global_best_covered = current_covered.copy()
        
        step_count += 1
        if step_count % 500 == 0 and time.time() > deadline - 5:
            break

    # Convert best_packing (list of indices) to list of triples
    out = [list(all_triples[i]) for i in global_best_packing]
    return out

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Num triples: {len(res)}")
    # Verify linearity
    masks = set()
    for t in res:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        if m in masks:
            print("Duplicate triple found!")
        masks.add(m)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            mi = (1 << res[i][0]) | (1 << res[i][1]) | (1 << res[i][2])
            mj = (1 << res[j][0]) | (1 << res[j][1]) | (1 << res[j][2])
            inter = mi & mj
            count = bin(inter).count('1')
            if count > 1:
                print(f"Non-linear: {res[i]} and {res[j]} share {count} points")
    
    # Compute alpha
    def check_alpha(triples):
        n = 16
        pop_arr = np.zeros(1 << n, dtype=np.int8)
        for i in range(1, 1 << n):
            pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        covered = np.zeros(1 << n, dtype=bool)
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            # Mark all supersets
            comp = ((1 << n) - 1) ^ m
            r = comp
            while True:
                covered[m | r] = True
                if r == 0:
                    break
                r = (r - 1) & comp
        return int(np.max(pop_arr[~covered]))
    
    print(f"Alpha: {check_alpha(res)}")
