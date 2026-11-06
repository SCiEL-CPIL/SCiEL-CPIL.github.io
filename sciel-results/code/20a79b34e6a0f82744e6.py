import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
        
    # Precompute compatibility
    compat = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            if (mi & mj).bit_count() <= 1:
                compat[i].add(j)
                compat[j].add(i)
    
    def get_alpha(T_indices):
        """Compute independence number efficiently using bitmask DP."""
        if not T_indices:
            return 16
        
        # A set S is independent if no triple in T is fully contained in S
        # alpha = max |S| such that for all t in T, not (mask(t) subset of mask(S))
        
        # Use the fact that we can check subsets from largest to smallest
        # But use a smarter approach: build up valid sets
        
        # For each point, track which triples it "blocks"
        # Actually, let's use a different approach:
        # The independence number is the size of the largest subset with no triple fully inside it.
        
        # Efficient method: iterate over all 2^16 subsets is too many (65536) but manageable
        # Actually 2^16 = 65536, and for each we check against |T| triples
        # This is 65536 * |T| which for |T|~30 is ~2M ops - fast enough!
        
        best = 0
        sys_masks = [triple_masks[i] for i in T_indices]
        
        for mask in range(1 << 16):
            # Check if mask is independent (no triple fully contained)
            independent = True
            for sm in sys_masks:
                if (mask & sm) == sm:
                    independent = False
                    break
            if independent:
                cnt = mask.bit_count()
                if cnt > best:
                    best = cnt
                    # Early exit if we can't do better
                    if best >= 16 - len(sys_masks):  # rough upper bound
                        break
        
        return best
    
    def greedy_packing(order):
        """Build a maximal packing following the given order."""
        T = []
        for idx in order:
            compatible = True
            for existing in T:
                if idx not in compat[existing]:
                    compatible = False
                    break
            if compatible:
                T.append(idx)
        return T
    
    def local_search(T, time_limit):
        """Improve T via add/remove/swap moves with simulated annealing."""
        current_T = T[:]
        current_alpha = get_alpha(current_T)
        best_T = current_T[:]
        best_alpha = current_alpha
        
        temp = 2.0  # Start temperature
        cooling = 0.995
        
        while time.time() < time_limit:
            temp *= cooling
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try adding a triple
                if len(current_T) < 50:
                    idx = random.randint(0, n_triples - 1)
                    if idx not in current_T:
                        ok = all(idx in compat[t] for t in current_T)
                        if ok:
                            new_T = current_T + [idx]
                            new_alpha = get_alpha(new_T)
                            delta = new_alpha - current_alpha
                            if delta <= 0 or random.random() < (1.0 / (1.0 + temp)):
                                current_T = new_T
                                current_alpha = new_alpha
                                if new_alpha < best_alpha:
                                    best_alpha = new_alpha
                                    best_T = new_T[:]
                
            elif move_type < 0.5:
                # Try removing a triple
                if len(current_T) > 3:
                    idx = random.choice(current_T)
                    new_T = [t for t in current_T if t != idx]
                    new_alpha = get_alpha(new_T)
                    delta = new_alpha - current_alpha
                    if delta <= 0 or random.random() < (1.0 / (1.0 + temp)):
                        current_T = new_T
                        current_alpha = new_alpha
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_T = new_T[:]
                        
            else:
                # Try swapping: remove one, add one
                if len(current_T) > 0:
                    remove_idx = random.choice(current_T)
                    add_idx = random.randint(0, n_triples - 1)
                    if add_idx != remove_idx and add_idx not in current_T:
                        ok = all(add_idx in compat[t] for t in current_T if t != remove_idx)
                        if ok:
                            new_T = [t for t in current_T if t != remove_idx]
                            new_T.append(add_idx)
                            new_alpha = get_alpha(new_T)
                            delta = new_alpha - current_alpha
                            if delta <= 0 or random.random() < (1.0 / (1.0 + temp)):
                                current_T = new_T
                                current_alpha = new_alpha
                                if new_alpha < best_alpha:
                                    best_alpha = new_alpha
                                    best_T = new_T[:]
            
            # Occasionally restart with fresh greedy
            if random.random() < 0.005:
                order = list(range(n_triples))
                random.shuffle(order)
                fresh = greedy_packing(order)
                fresh_alpha = get_alpha(fresh)
                if fresh_alpha < best_alpha:
                    best_alpha = fresh_alpha
                    best_T = fresh[:]
                    current_T = fresh[:]
                    current_alpha = fresh_alpha
                    temp = 2.0
        
        return best_T, best_alpha
    
    best_alpha = 16
    best_system = []
    
    # Multiple restarts with local search
    while time.time() < deadline:
        order = list(range(n_triples))
        random.shuffle(order)
        
        T = greedy_packing(order)
        alpha = get_alpha(T)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = T[:]
        
        # If we've found a decent solution, do local search
        if alpha <= 14 and time.time() < deadline - 15:
            T, alpha = local_search(T, deadline - 5)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = T[:]
        
        # Early termination
        if best_alpha <= 7:
            break
    
    if not best_system:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return [list(all_triples[i]) for i in best_system]
