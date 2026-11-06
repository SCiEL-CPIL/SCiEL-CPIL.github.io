import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
    
    # Precompute compatibility: compat[i] is a set of triple indices compatible with i
    # Two triples are compatible if they share at most 1 point
    compat = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            if bin(mi & mj).count('1') <= 1:
                compat[i].add(j)
                compat[j].add(i)
    
    def compute_alpha(triples):
        """Compute independence number: largest subset of points containing no triple."""
        # A subset S is independent if no triple is entirely in S
        # Equivalently, for every triple (a,b,c), S is not a superset of {a,b,c}
        # We want max |S| such that for all triples t, t is not subset of S
        
        # Use the complement: S is independent iff for every triple, at least one point is NOT in S
        # Let's think of it as: we want to find the largest S where no triple is fully contained.
        
        # Brute force over all subsets is 2^16 = 65536, which is fast enough.
        # But let's be smarter: check from large to small.
        
        # Actually, 2^16 = 65536 subsets. For each subset, check if any triple is fully in it.
        # With ~20-30 triples, this is fast.
        
        triple_set = [frozenset(t) for t in triples]
        n = 16
        
        # Check subsets from size 16 down to 1
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                # Check if any triple is fully in s_mask
                has_triple = False
                for t_mask in triple_masks_for_subset:
                    if (t_mask & s_mask) == t_mask:
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 1
    
    # Better: precompute for speed
    def compute_alpha_fast(triple_list):
        """Compute independence number efficiently."""
        # Precompute masks for the given triples
        t_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triple_list]
        
        # Check from size 16 down
        for size in range(16, 0, -1):
            for subset in itertools.combinations(range(16), size):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                # Check if any triple is fully contained
                found = False
                for tm in t_masks:
                    if (tm & s_mask) == tm:
                        found = True
                        break
                if not found:
                    return size
        return 1
    
    def random_greedy_packing(rng):
        """Generate a random maximal linear triple packing."""
        available = set(range(n_triples))
        chosen = []
        used_points_mask = 0  # not strictly needed for linearity, just tracking
        
        # Random order
        order = list(range(n_triples))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            # Check compatibility with all chosen
            compatible = True
            for c_idx in chosen:
                if idx not in compat[c_idx]:
                    compatible = False
                    break
            if compatible:
                chosen.append(idx)
        
        return [list(all_triples[i]) for i in chosen]
    
    def try_swap(triples, rng):
        """Try to improve by swapping one triple for another."""
        n = len(triples)
        if n == 0:
            return None
        
        # Try removing one triple and adding a different one
        for remove_idx in range(n):
            remaining = [t for i, t in enumerate(triples) if i != remove_idx]
            remaining_indices = [i for i in range(n) if i != remove_idx]
            
            # Find all triples compatible with remaining
            candidates = []
            for idx in range(n_triples):
                if idx in remaining_indices:
                    continue
                compatible = True
                for r_idx in remaining_indices:
                    if idx not in compat[r_idx]:
                        compatible = False
                        break
                if compatible:
                    candidates.append(idx)
            
            if not candidates:
                continue
            
            # Try a random candidate
            new_idx = rng.choice(candidates)
            new_triples = remaining + [list(all_triples[new_idx])]
            return new_triples
        
        return None
    
    # Main search
    best_triples = None
    best_alpha = 16  # worst case
    
    # Initial random packings
    rng = random.Random(42)
    
    # First, get a good initial packing
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        packing = random_greedy_packing(rng)
        alpha = compute_alpha_fast(packing)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = packing
    
    if best_triples is None:
        # Fallback to seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Local search
    current = best_triples
    current_alpha = best_alpha
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        if no_improve_count >= max_no_improve:
            # Restart with a new random packing
            packing = random_greedy_packing(rng)
            alpha = compute_alpha_fast(packing)
            if alpha < current_alpha:
                current = packing
                current_alpha = alpha
                no_improve_count = 0
            else:
                no_improve_count = 0
            continue
        
        # Try a swap
        new_triples = try_swap(current, rng)
        if new_triples is None:
            no_improve_count += 1
            continue
        
        new_alpha = compute_alpha_fast(new_triples)
        if new_alpha < current_alpha:
            current = new_triples
            current_alpha = new_alpha
            no_improve_count = 0
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current
        else:
            no_improve_count += 1
    
    return best_triples
