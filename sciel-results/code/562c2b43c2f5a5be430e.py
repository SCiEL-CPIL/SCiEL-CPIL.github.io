import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all triples
    triple_masks = []
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
        
    num_triples = len(all_triples)
    
    # Precompute compatibility: compat[i] is a bitmask of triples compatible with triple i
    compat = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        mask = 0
        for j in range(num_triples):
            if i == j:
                continue
            mj = triple_masks[j]
            if (mi & mj).bit_count() <= 1:
                mask |= (1 << j)
        compat[i] = mask
    
    # Precompute for each triple, the set of point pairs it covers
    # Pair represented as (min, max) tuple -> we'll use a 16x16 boolean matrix
    # pair_covered[p][q] = True if some triple in system contains both p and q
    # But we need to track which triples cover which pairs
    
    # For each triple index, compute the 3 pairs it covers
    triple_pairs = []
    for i, t in enumerate(all_triples):
        pairs = [(min(t[0], t[1]), max(t[0], t[1])),
                 (min(t[0], t[2]), max(t[0], t[2])),
                 (min(t[1], t[2]), max(t[1], t[2]))]
        triple_pairs.append(pairs)
    
    # Exact alpha computation using subset enumeration from top down
    def get_alpha_exact(system_indices):
        """Compute the independence number exactly."""
        if not system_indices:
            return 16
        
        sys_masks = [triple_masks[i] for i in system_indices]
        
        for k in range(16, 0, -1):
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for sm in sys_masks:
                    if (mask & sm) == sm:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 0
    
    # Pair-coverage greedy: build system by always adding the triple that covers
    # the most currently uncovered point pairs while maintaining linearity
    def pair_coverage_greedy():
        """Greedy construction maximizing pair coverage."""
        system = []
        system_set = set()
        
        # Track which pairs are covered
        pair_covered = [[False] * 16 for _ in range(16)]
        
        while True:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_new_pairs = -1
            best_system_size = -1
            
            # Try adding each candidate triple
            for idx in range(num_triples):
                if idx in system_set:
                    continue
                
                # Check linearity with existing system
                compatible = True
                for existing in system:
                    if not (compat[existing] & (1 << idx)):
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Count new pairs covered
                new_pairs = 0
                for p1, p2 in triple_pairs[idx]:
                    if not pair_covered[p1][p2]:
                        new_pairs += 1
                
                # Prefer triples that cover more new pairs
                # Also prefer to keep system small (fewer triples = less chance of large independent sets)
                # Actually, more triples should help reduce alpha
                if new_pairs > best_new_pairs or (new_pairs == best_new_pairs and len(system) < best_system_size):
                    best_new_pairs = new_pairs
                    best_idx = idx
                    best_system_size = len(system)
            
            if best_idx == -1:
                break
            
            # Add the best triple
            system.append(best_idx)
            system_set.add(best_idx)
            for p1, p2 in triple_pairs[best_idx]:
                pair_covered[p1][p2] = True
                pair_covered[p2][p1] = True
        
        return system
    
    # Local search: try to reduce alpha by swapping or adding/removing triples
    def local_search(system, max_iterations=200):
        """Improve system to reduce independence number."""
        if not system:
            return system
        
        current_alpha = get_alpha_exact(system)
        current = system[:]
        best_alpha = current_alpha
        best_system = current[:]
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try different moves:
            # 1. Add a new triple (if compatible)
            # 2. Remove a triple
            # 3. Swap a triple
            
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Find a triple compatible with all in current
                candidates = []
                for j in range(num_triples):
                    if j in current:
                        continue
                    ok = True
                    for o in current:
                        if not (compat[o] & (1 << j)):
                            ok = False
                            break
                    if ok:
                        candidates.append(j)
                
                if candidates:
                    new_idx = random.choice(candidates)
                    new_system = current + [new_idx]
                    new_alpha = get_alpha_exact(new_system)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_system = new_system
                        current = new_system
                    elif new_alpha == best_alpha and random.random() < 0.3:
                        current = new_system  # Accept equal for exploration
            
            elif move_type == 'remove' and len(current) > 3:
                # Remove a random triple
                remove_pos = random.randint(0, len(current) - 1)
                new_system = current[:remove_pos] + current[remove_pos+1:]
                new_alpha = get_alpha_exact(new_system)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_system = new_system
                    current = new_system
            
            elif move_type == 'swap' and len(current) > 0:
                # Swap a triple
                remove_pos = random.randint(0, len(current) - 1)
                remove_idx = current[remove_pos]
                others = [x for i, x in enumerate(current) if i != remove_pos]
                
                candidates = []
                for j in range(num_triples):
                    if j == remove_idx or j in others:
                        continue
                    ok = True
                    for o in others:
                        if not (compat[o] & (1 << j)):
                            ok = False
                            break
                    if ok:
                        candidates.append(j)
                
                if candidates:
                    new_idx = random.choice(candidates)
                    new_system = current[:]
                    new_system[remove_pos] = new_idx
                    new_alpha = get_alpha_exact(new_system)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_system = new_system
                        current = new_system
                    elif new_alpha == best_alpha and random.random() < 0.2:
                        current = new_system
        
        return best_system
    
    best_system = None
    best_alpha = 17
    
    # Phase 1: Build initial systems using pair-coverage greedy
    for _ in range(5):
        if time.time() > deadline:
            break
        
        system = pair_coverage_greedy()
        if system:
            alpha = get_alpha_exact(system)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = system[:]
            
            # Local search refinement
            refined = local_search(system, max_iterations=100)
            alpha = get_alpha_exact(refined)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = refined[:]
    
    # Phase 2: More aggressive local search from best solution
    if best_system is not None:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            refined = local_search(best_system, max_iterations=200)
            alpha = get_alpha_exact(refined)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = refined[:]
    
    # Phase 3: Random restarts with local search if time remains
    if time.time() < deadline - 10:
        while time.time() < deadline - 2:
            # Random greedy build
            indices = list(range(num_triples))
            random.shuffle(indices)
            system = []
            for idx in indices:
                compatible = True
                for existing in system:
                    if not (compat[existing] & (1 << idx)):
                        compatible = False
                        break
                if compatible:
                    system.append(idx)
            
            if system:
                alpha = get_alpha_exact(system)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_system = system[:]
                
                # Quick local search
                if time.time() < deadline - 5:
                    refined = local_search(system, max_iterations=50)
                    alpha = get_alpha_exact(refined)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_system = refined[:]
    
    if best_system is None:
        return []
        
    # Convert indices back to triples
    return [list(all_triples[i]) for i in best_system]
