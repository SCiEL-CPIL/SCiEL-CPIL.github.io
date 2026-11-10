import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all triples as masks
    triple_list = []
    mask_to_idx = {}
    idx_to_mask = {}
    idx_to_pts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(triple_list)
                triple_list.append(mask)
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                idx_to_pts[idx] = (i, j, k)
                
    num_triples = len(triple_list)
    
    # Precompute conflicts: two triples conflict if they share >= 2 points
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = triple_list[i]
        for j in range(i + 1, num_triples):
            mj = triple_list[j]
            common = mi & mj
            if common & (common - 1):  # Has at least 2 bits set
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute for each triple, which 5-sets contain it
    # A 5-set contains a triple if the triple mask is a subset of the 5-set mask
    triple_to_5sets = [[] for _ in range(num_triples)]
    
    # Precompute all 5-sets and their triple subsets
    masks_size_5 = []
    mask_5_to_triples = {}
    for mask in range(1 << n):
        if bin(mask).count('1') == 5:
            masks_size_5.append(mask)
            triples_in_mask = []
            for t_idx in range(num_triples):
                t_mask = triple_list[t_idx]
                if (t_mask & mask) == t_mask:
                    triples_in_mask.append(t_idx)
                    triple_to_5sets[t_idx].append(len(masks_size_5) - 1)
            mask_5_to_triples[mask] = triples_in_mask
            
    num_5sets = len(masks_size_5)
    
    # Precompute for each triple, which 4-sets contain it
    masks_size_4 = []
    mask_4_to_triples = {}
    triple_to_4sets = [[] for _ in range(num_triples)]
    for mask in range(1 << n):
        if bin(mask).count('1') == 4:
            masks_size_4.append(mask)
            triples_in_mask = []
            for t_idx in range(num_triples):
                t_mask = triple_list[t_idx]
                if (t_mask & mask) == t_mask:
                    triples_in_mask.append(t_idx)
                    triple_to_4sets[t_idx].append(len(masks_size_4) - 1)
            mask_4_to_triples[mask] = triples_in_mask
            
    num_4sets = len(masks_size_4)
    
    # State: maintain coverage counts for 5-sets and 4-sets
    # coverage_5[i] = number of edges in current system that are subsets of 5-set i
    # A 5-set is independent iff coverage_5[i] == 0
    
    def init_coverage(edges_set):
        """Initialize coverage arrays from a set of edge indices."""
        cov5 = [0] * num_5sets
        cov4 = [0] * num_4sets
        for t_idx in edges_set:
            for s_idx in triple_to_5sets[t_idx]:
                cov5[s_idx] += 1
            for s_idx in triple_to_4sets[t_idx]:
                cov4[s_idx] += 1
        return cov5, cov4
    
    def add_edge(t_idx, cov5, cov4):
        """Add edge t_idx, updating coverage arrays."""
        for s_idx in triple_to_5sets[t_idx]:
            cov5[s_idx] += 1
        for s_idx in triple_to_4sets[t_idx]:
            cov4[s_idx] += 1
    
    def remove_edge(t_idx, cov5, cov4):
        """Remove edge t_idx, updating coverage arrays."""
        for s_idx in triple_to_5sets[t_idx]:
            cov5[s_idx] -= 1
        for s_idx in triple_to_4sets[t_idx]:
            cov4[s_idx] -= 1
    
    def calc_alpha(cov5, cov4, edges_set):
        """
        Calculate independence number using coverage arrays.
        Returns the maximum size of a subset with no edge fully contained.
        """
        # Check size 5: is there a 5-set with coverage 0?
        for i in range(num_5sets):
            if cov5[i] == 0:
                return 5
                
        # Check size 4: is there a 4-set with coverage 0?
        for i in range(num_4sets):
            if cov4[i] == 0:
                return 4
                
        # Check size 3: is there a triple not in edges_set?
        if len(edges_set) < num_triples:
            return 3
            
        return 2
    
    def is_valid_add(triple_idx, edges_set):
        """Check if triple_idx can be added to edges_set without conflict."""
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True
        
    def greedy_pack():
        """Greedy construction of a linear triple system."""
        current_triples = set()
        available = list(range(num_triples))
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
                
        return current_triples
    
    best_alpha = n
    best_triples = None
    best_cov5 = None
    best_cov4 = None
    
    # Try multiple greedy starts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        current_triples = greedy_pack()
        cov5, cov4 = init_coverage(current_triples)
        current_alpha = calc_alpha(cov5, cov4, current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current_triples)
            best_cov5 = list(cov5)
            best_cov4 = list(cov4)
            if best_alpha <= 4:
                break
    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Local search with targeted moves
    current_triples = set(best_triples)
    cov5 = list(best_cov5)
    cov4 = list(best_cov4)
    current_alpha = best_alpha
    
    T = 1.0
    cooling_rate = 0.997
    min_T = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        new_triples = set(current_triples)
        new_cov5 = list(cov5)
        new_cov4 = list(cov4)
        move_valid = False
        
        if current_alpha >= 5 and move_type < 0.6:
            # Targeted: find an independent 5-set and try to add an edge covering it
            # Find a random independent 5-set
            independent_5sets = [i for i in range(num_5sets) if new_cov5[i] == 0]
            if independent_5sets:
                target_5set_idx = random.choice(independent_5sets)
                target_5set_mask = masks_size_5[target_5set_idx]
                target_triples = mask_5_to_triples[target_5set_mask]
                random.shuffle(target_triples)
                
                for t_idx in target_triples:
                    if t_idx in new_triples:
                        continue
                    if is_valid_add(t_idx, new_triples):
                        # Add this edge
                        new_triples.add(t_idx)
                        add_edge(t_idx, new_cov5, new_cov4)
                        move_valid = True
                        break
                    else:
                        # Try to remove a conflicting edge and add this one
                        conflicts_in_set = [c for c in conflicts[t_idx] if c in new_triples]
                        if conflicts_in_set:
                            c_idx = random.choice(conflicts_in_set)
                            new_triples.remove(c_idx)
                            remove_edge(c_idx, new_cov5, new_cov4)
                            if is_valid_add(t_idx, new_triples):
                                new_triples.add(t_idx)
                                add_edge(t_idx, new_cov5, new_cov4)
                                move_valid = True
                                break
        elif current_alpha == 4 and move_type < 0.7:
            # Try to eliminate independent 4-sets
            independent_4sets = [i for i in range(num_4sets) if new_cov4[i] == 0]
            if independent_4sets:
                target_4set_idx = random.choice(independent_4sets)
                target_4set_mask = masks_size_4[target_4set_idx]
                target_triples = mask_4_to_triples[target_4set_mask]
                random.shuffle(target_triples)
                
                for t_idx in target_triples:
                    if t_idx in new_triples:
                        continue
                    if is_valid_add(t_idx, new_triples):
                        new_triples.add(t_idx)
                        add_edge(t_idx, new_cov5, new_cov4)
                        move_valid = True
                        break
                    else:
                        conflicts_in_set = [c for c in conflicts[t_idx] if c in new_triples]
                        if conflicts_in_set:
                            c_idx = random.choice(conflicts_in_set)
                            new_triples.remove(c_idx)
                            remove_edge(c_idx, new_cov5, new_cov4)
                            if is_valid_add(t_idx, new_triples):
                                new_triples.add(t_idx)
                                add_edge(t_idx, new_cov5, new_cov4)
                                move_valid = True
                                break
        else:
            # Random move: add, remove, or swap
            if move_type < 0.4:
                # Try to add a random valid edge
                valid_adds = []
                for idx in range(num_triples):
                    if idx not in new_triples and is_valid_add(idx, new_triples):
                        valid_adds.append(idx)
                if valid_adds:
                    t_idx = random.choice(valid_adds)
                    new_triples.add(t_idx)
                    add_edge(t_idx, new_cov5, new_cov4)
                    move_valid = True
            elif move_type < 0.7:
                # Remove a random edge
                if new_triples:
                    t_idx = random.choice(list(new_triples))
                    new_triples.remove(t_idx)
                    remove_edge(t_idx, new_cov5, new_cov4)
                    move_valid = True
            else:
                # Swap: remove one, try to add another
                if new_triples:
                    to_remove = random.choice(list(new_triples))
                    new_triples.remove(to_remove)
                    remove_edge(to_remove, new_cov5, new_cov4)
                    
                    valid_adds = []
                    for idx in range(num_triples):
                        if idx not in new_triples and is_valid_add(idx, new_triples):
                            valid_adds.append(idx)
                    if valid_adds:
                        t_idx = random.choice(valid_adds)
                        new_triples.add(t_idx)
                        add_edge(t_idx, new_cov5, new_cov4)
                    move_valid = True
        
        if not move_valid:
            continue
        
        new_alpha = calc_alpha(new_cov5, new_cov4, new_triples)
        
        # Acceptance
        delta = new_alpha - current_alpha
        if delta <= 0:
            current_triples = new_triples
            cov5 = new_cov5
            cov4 = new_cov4
            current_alpha = new_alpha
            no_improve_count = 0
            if delta < 0:
                T = max(T, 0.5)
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = set(current_triples)
                    best_cov5 = list(cov5)
                    best_cov4 = list(cov4)
        else:
            if random.random() < (2.71828 ** (-delta / T)):
                current_triples = new_triples
                cov5 = new_cov5
                cov4 = new_cov4
                current_alpha = new_alpha
                no_improve_count += 1
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            if random.random() < 0.15:
                # Perturb: remove 2-3 random edges
                for _ in range(random.randint(2, 3)):
                    if current_triples:
                        t_idx = random.choice(list(current_triples))
                        current_triples.remove(t_idx)
                        remove_edge(t_idx, cov5, cov4)
                current_alpha = calc_alpha(cov5, cov4, current_triples)
    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    result = []
    for idx in best_triples:
        pts = list(idx_to_pts[idx])
        result.append(pts)
        
    return result
