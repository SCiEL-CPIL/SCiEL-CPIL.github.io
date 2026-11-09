# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)
    best_family = []
    best_size = 0
    rng = random.Random(12345)

    def check_conflict(m, F, inter_maps):
        """Check if m conflicts with any pair in F."""
        for i in range(len(F)):
            s = m & F[i]
            lst = inter_maps[i].get(s)
            if lst:
                for j in lst:
                    if (m & F[j]) == s:
                        return True
        return False

    def greedy_build(order, current_best_size, time_limit=None):
        """
        Greedily build a family from the given order of set indices.
        Returns the family as a list of masks.
        """
        if time_limit and time.time() > time_limit:
            return []
            
        F = []
        inter_maps = []
        
        for idx in order:
            if time_limit and time.time() > time_limit:
                break
            m = all_sets[idx]
            conflict = check_conflict(m, F, inter_maps)
            
            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})
                
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)
                
                if len(F) >= max_target:
                    break

        return F

    def local_refine(F, inter_maps, time_limit):
        """
        Try to add more sets by finding sets that conflict with few existing sets.
        For each candidate set, count how many pairs it would create sunflowers with.
        If removing a few sets allows adding more, do it.
        """
        if not F:
            return F, inter_maps
        
        # Build set of masks in F for quick lookup
        F_set = set(F)
        
        # For each candidate not in F, compute conflict count
        best_addition = None
        best_conflict_count = float('inf')
        best_remove_set = None
        
        # Sample a subset of candidates to keep this fast
        sample_size = min(500, total - len(F))
        if sample_size <= 0:
            return F, inter_maps
        
        candidates = [i for i in range(total) if all_sets[i] not in F_set]
        rng.shuffle(candidates)
        candidates = candidates[:sample_size]
        
        for cand_idx in candidates:
            if time.time() > time_limit:
                break
            m = all_sets[cand_idx]
            
            # Find all pairs (i,j) in F such that {m, F[i], F[j]} is a sunflower
            conflicting_pairs = set()
            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflicting_pairs.add((min(i,j), max(i,j)))
            
            if len(conflicting_pairs) == 0:
                # Can add directly
                F.append(m)
                idx_new = len(F) - 1
                inter_maps.append({})
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)
                return F, inter_maps
            
            # Find minimum set of elements to remove to resolve all conflicts
            # Use greedy: repeatedly remove the element involved in most conflicts
            remaining_pairs = set(conflicting_pairs)
            removed = set()
            
            while remaining_pairs and len(removed) < 5:  # Limit removals to 5
                # Count how many conflicts each element is in
                elem_count = {}
                for (i, j) in remaining_pairs:
                    elem_count[i] = elem_count.get(i, 0) + 1
                    elem_count[j] = elem_count.get(j, 0) + 1
                
                if not elem_count:
                    break
                    
                # Remove the most conflicted element
                max_elem = max(elem_count, key=elem_count.get)
                removed.add(max_elem)
                
                # Remove all pairs involving max_elem
                remaining_pairs = {(i, j) for (i, j) in remaining_pairs if i != max_elem and j != max_elem}
            
            if not remaining_pairs and len(removed) > 0:
                # Check if this is better than current best
                net_gain = 1 - len(removed)
                if best_addition is None or len(removed) < best_conflict_count:
                    best_addition = cand_idx
                    best_conflict_count = len(removed)
                    best_remove_set = removed
        
        # Apply the best improvement if found
        if best_addition is not None and best_remove_set is not None:
            # Remove the specified elements
            remove_list = sorted(best_remove_set, reverse=True)
            for idx in remove_list:
                # Rebuild inter_maps is complex, so we'll just rebuild from scratch
                pass
            
            # For simplicity, rebuild the family without the removed elements
            keep_indices = [i for i in range(len(F)) if i not in best_remove_set]
            new_F = [F[i] for i in keep_indices]
            new_inter_maps = []
            
            # Rebuild inter_maps for new_F
            for i in range(len(new_F)):
                new_inter_maps.append({})
            
            for i in range(len(new_F)):
                for j in range(i + 1, len(new_F)):
                    s = new_F[i] & new_F[j]
                    if s not in new_inter_maps[i]:
                        new_inter_maps[i][s] = []
                    new_inter_maps[i][s].append(j)
                    if s not in new_inter_maps[j]:
                        new_inter_maps[j][s] = []
                    new_inter_maps[j][s].append(i)
            
            # Add the new element
            new_F.append(all_sets[best_addition])
            idx_new = len(new_F) - 1
            new_inter_maps.append({})
            for i in range(idx_new):
                s = new_F[idx_new] & new_F[i]
                if s not in new_inter_maps[i]:
                    new_inter_maps[i][s] = []
                new_inter_maps[i][s].append(idx_new)
                if s not in new_inter_maps[idx_new]:
                    new_inter_maps[idx_new][s] = []
                new_inter_maps[idx_new][s].append(i)
            
            F = new_F
            inter_maps = new_inter_maps
        
        return F, inter_maps

    # 1. Deterministic pass (natural order)
    if time.time() < deadline:
        order = list(range(total))
        F = greedy_build(order, best_size)
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
    
    # 2. Random restarts with different seeds/orders
    restart_count = 0
    while time.time() < deadline - 5.0:  # Leave 5s for final refinement
        restart_count += 1
        seed = 12345 + restart_count
        rng = random.Random(seed)
        
        order = list(range(total))
        rng.shuffle(order)
        
        F = greedy_build(order, best_size)
        
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
            if best_size >= max_target:
                break
    
    # 3. Final refinement pass on the best family found
    if best_family and time.time() < deadline - 1.0:
        # Rebuild inter_maps for best_family
        F = list(best_family)
        inter_maps = []
        for i in range(len(F)):
            inter_maps.append({})
        
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                s = F[i] & F[j]
                if s not in inter_maps[i]:
                    inter_maps[i][s] = []
                inter_maps[i][s].append(j)
                if s not in inter_maps[j]:
                    inter_maps[j][s] = []
                inter_maps[j][s].append(i)
        
        # Try local refinement
        refined_F, refined_inter_maps = local_refine(F, inter_maps, deadline - 0.5)
        if len(refined_F) > best_size:
            best_family = refined_F
            best_size = len(refined_F)
    
    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
