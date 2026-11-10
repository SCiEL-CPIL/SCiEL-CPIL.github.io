import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all triples as masks and indices
    triple_list = []
    mask_to_idx = {}
    idx_to_mask = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(triple_list)
                triple_list.append(mask)
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                
    num_triples = len(triple_list)
    
    # Precompute conflicts
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = triple_list[i]
        for j in range(i + 1, num_triples):
            mj = triple_list[j]
            if bin(mi & mj).count('1') >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute masks by size for alpha calculation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        k = bin(mask).count('1')
        if k > 0:
            masks_by_size[k].append(mask)
    
    def is_independent(mask, edges_set):
        for e_idx in edges_set:
            e_mask = idx_to_mask[e_idx]
            if (mask & e_mask) == e_mask:
                return False
        return True
        
    def calc_alpha(edges_set):
        if not edges_set:
            return n
        for m in masks_by_size[5]:
            if is_independent(m, edges_set):
                return 5
        for m in masks_by_size[4]:
            if is_independent(m, edges_set):
                return 4
        for m in masks_by_size[3]:
            if is_independent(m, edges_set):
                return 3
        for m in masks_by_size[2]:
            if is_independent(m, edges_set):
                return 2
        return 1

    def is_valid_add(triple_idx, edges_set):
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True

    def greedy_pack(seed_offset=0):
        current_triples = set()
        available = list(range(num_triples))
        random.seed(seed_offset)
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
        
        return current_triples

    def two_opt_swap(current_triples):
        """
        Perform a 2-opt move: remove two edges and add two new valid edges.
        This allows escaping local optima that single-edge swaps can't.
        """
        if len(current_triples) < 2:
            return None
            
        # Pick two edges to remove
        edges_list = list(current_triples)
        e1, e2 = random.sample(edges_list, 2)
        
        new_triples = set(current_triples)
        new_triples.discard(e1)
        new_triples.discard(e2)
        
        # Try to add two new edges
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        added = 0
        for idx in candidates:
            if idx in new_triples:
                continue
            if is_valid_add(idx, new_triples):
                new_triples.add(idx)
                added += 1
                if added == 2:
                    break
        
        if added == 0:
            return None
            
        return new_triples

    def targeted_two_opt(current_triples, witness_mask):
        """
        Targeted 2-opt: try to break a witness independent set by adding edges inside it,
        potentially removing conflicting edges.
        """
        points = []
        for i in range(n):
            if witness_mask & (1 << i):
                points.append(i)
                
        # Generate all triples from these points
        local_triples = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    mask = (1 << points[i]) | (1 << points[j]) | (1 << points[k])
                    idx = mask_to_idx.get(mask)
                    if idx is not None:
                        local_triples.append(idx)
                        
        if not local_triples:
            return None
            
        random.shuffle(local_triples)
        
        # Try adding one or two edges from local_triples
        for num_add in [1, 2]:
            for combo in itertools.combinations(local_triples, num_add):
                new_triples = set(current_triples)
                
                # Check if we can add these edges, possibly removing conflicts
                can_add = True
                for t_idx in combo:
                    if t_idx in new_triples:
                        continue
                    if not is_valid_add(t_idx, new_triples):
                        # Find conflicts and try to remove one
                        conflicts_in_set = [c for c in conflicts[t_idx] if c in new_triples]
                        if not conflicts_in_set:
                            can_add = False
                            break
                        # Remove one conflict
                        new_triples.discard(random.choice(conflicts_in_set))
                    
                if can_add:
                    for t_idx in combo:
                        if t_idx not in new_triples and is_valid_add(t_idx, new_triples):
                            new_triples.add(t_idx)
                    return new_triples
                    
        return None

    def point_coverage_heuristic(current_triples):
        """
        Score based on how well points are covered.
        Returns a dict of point -> number of edges containing it.
        """
        coverage = [0] * n
        for e_idx in current_triples:
            mask = idx_to_mask[e_idx]
            for i in range(n):
                if mask & (1 << i):
                    coverage[i] += 1
        return coverage

    best_alpha = n
    best_triples = None
    
    # Multiple restarts with different initializations
    for restart in range(8):
        if time.time() > deadline:
            break
            
        # Different greedy initializations
        current_triples = greedy_pack(restart * 1000)
        current_alpha = calc_alpha(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current_triples)
            
        # Simulated annealing with 2-opt moves
        T = 1.0
        cooling_rate = 0.997
        min_T = 0.03
        no_improve_steps = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_steps < max_no_improve:
            current_alpha = calc_alpha(current_triples)
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = set(current_triples)
                no_improve_steps = 0
                T = max(T, 0.5)
                
                if best_alpha <= 4:
                    # Found good solution, do more local search
                    no_improve_steps = -100  # Allow more search
            else:
                no_improve_steps += 1
                
            # Generate move
            move_type = random.random()
            new_triples = None
            
            if current_alpha > 4 and move_type < 0.6:
                # Targeted move on a witness
                target_size = 5 if current_alpha >= 5 else current_alpha
                witness = None
                candidates = random.sample(masks_by_size[target_size], min(30, len(masks_by_size[target_size])))
                for m in candidates:
                    if is_independent(m, current_triples):
                        witness = m
                        break
                
                if witness:
                    new_triples = targeted_two_opt(current_triples, witness)
            
            if new_triples is None and move_type < 0.85:
                # 2-opt move
                new_triples = two_opt_swap(current_triples)
            
            if new_triples is None:
                # Single edge swap as fallback
                if current_triples:
                    to_remove = random.choice(list(current_triples))
                    new_triples = set(current_triples)
                    new_triples.remove(to_remove)
                    
                    candidates = list(range(num_triples))
                    random.shuffle(candidates)
                    
                    for idx in candidates:
                        if idx in new_triples:
                            continue
                        if is_valid_add(idx, new_triples):
                            new_triples.add(idx)
                            break
                    else:
                        new_triples = None
                
            if new_triples is None:
                # Perturb: remove 2-3 random edges
                for _ in range(random.randint(2, 3)):
                    if current_triples:
                        current_triples.remove(random.choice(list(current_triples)))
                continue
                
            new_alpha = calc_alpha(new_triples)
            
            # Acceptance
            delta = new_alpha - current_alpha
            if delta <= 0:
                current_triples = new_triples
                current_alpha = new_alpha
                if delta < 0:
                    T = max(T, 0.5)
            else:
                if random.random() < (2.71828 ** (-delta / T)):
                    current_triples = new_triples
                    current_alpha = new_alpha
                    
            T *= cooling_rate
            if T < min_T:
                T = min_T
                if random.random() < 0.15:
                    for _ in range(2):
                        if current_triples:
                            current_triples.remove(random.choice(list(current_triples)))

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    # Convert indices back to points
    result = []
    for idx in best_triples:
        mask = idx_to_mask[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
