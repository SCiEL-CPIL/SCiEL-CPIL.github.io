import random
import itertools
import time

def construct():
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks
    triple_masks = []
    for t in all_triples:
        m = 0
        for v in t:
            m |= (1 << v)
        triple_masks.append(m)
        
    # Conflicts
    conflict = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        for j in range(i+1, num_triples):
            if (triple_masks[i] & triple_masks[j]).bit_count() > 1:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # Target Alpha = 4 => Cover all subsets of size 5
    subsets_5 = list(itertools.combinations(points, 5))
    num_subsets = len(subsets_5)
    
    subset_masks = []
    for s in subsets_5:
        m = 0
        for v in s:
            m |= (1 << v)
        subset_masks.append(m)
        
    # Map triple -> list of subset indices it covers
    triple_to_subsets = [[] for _ in range(num_triples)]
    # Map subset -> list of triple indices that cover it
    subset_to_triples = [[] for _ in range(num_subsets)]
    
    for i, t_mask in enumerate(triple_masks):
        for j, s_mask in enumerate(subset_masks):
            if (t_mask & s_mask) == t_mask:
                triple_to_subsets[i].append(j)
                subset_to_triples[j].append(i)
                
    def find_solution_for_alpha(alpha_target):
        # For alpha=4, cover size 5
        # For alpha=3, cover size 4
        if alpha_target == 4:
            target_subsets = subsets_5
            target_num = num_subsets
            t_to_s = triple_to_subsets
            s_to_t = subset_to_triples
        elif alpha_target == 3:
            # Generate subsets of size 4
            target_subsets = list(itertools.combinations(points, 4))
            target_num = len(target_subsets)
            t_masks = triple_masks
            s_masks = []
            for s in target_subsets:
                m = 0
                for v in s:
                    m |= (1 << v)
                s_masks.append(m)
            t_to_s = [[] for _ in range(num_triples)]
            s_to_t = [[] for _ in range(target_num)]
            for i, t_mask in enumerate(t_masks):
                for j, s_mask in enumerate(s_masks):
                    if (t_mask & s_mask) == t_mask:
                        t_to_s[i].append(j)
                        s_to_t[j].append(i)
        else:
            return None
            
        start_time = time.time()
        timeout = 60
        
        def backtrack(selected, uncovered, available):
            if time.time() - start_time > timeout:
                return None
                
            if not uncovered:
                return selected[:]
                
            # Find the most constrained subset (fewest available coverers)
            min_count = float('inf')
            best_subset = -1
            best_coverers = None
            
            for s_idx in uncovered:
                coverers = s_to_t[s_idx]
                avail_coverers = [t for t in coverers if t in available]
                if len(avail_coverers) < min_count:
                    min_count = len(avail_coverers)
                    best_subset = s_idx
                    best_coverers = avail_coverers
                    if min_count == 1:
                        break
                        
            if best_subset == -1 or min_count == 0:
                return None
                
            # Randomize coverers
            random.shuffle(best_coverers)
            
            for t in best_coverers:
                new_selected = selected + [t]
                new_uncovered = uncovered.copy()
                for s_idx in t_to_s[t]:
                    new_uncovered.discard(s_idx)
                    
                new_available = available.copy()
                new_available.discard(t)
                for c in conflict[t]:
                    new_available.discard(c)
                    
                result = backtrack(new_selected, new_uncovered, new_available)
                if result is not None:
                    return result
                    
            return None
            
        # Greedy pre-selection to reduce search space
        def greedy_preselect(available, uncovered):
            selected = []
            while uncovered and time.time() - start_time < timeout:
                best_t = -1
                max_cov = -1
                for t in available:
                    cov = 0
                    for s_idx in t_to_s[t]:
                        if s_idx in uncovered:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_t = t
                        if max_cov == len(t_to_s[t]):
                            break
                            
                if best_t == -1:
                    break
                
                selected.append(best_t)
                for s_idx in t_to_s[best_t]:
                    uncovered.discard(s_idx)
                    
                new_available = available.copy()
                new_available.discard(best_t)
                for c in conflict[best_t]:
                    new_available.discard(c)
                available = new_available
                
            return selected, uncovered, available
            
        # Try a few restarts with randomized greedy
        for restart in range(10):
            init_available = set(range(num_triples))
            init_uncovered = set(range(target_num))
            
            # Randomize the order of triples in greedy? 
            # The greedy picks the one with max coverage.
            # To diversify, we can slightly perturb the scores or just rely on the backtracking.
            
            sel, unc, avail = greedy_preselect(init_available, init_uncovered)
            
            if not unc:
                return [all_triples[t] for t in sel]
                
            result = backtrack(sel, unc, avail)
            if result:
                return [all_triples[t] for t in result]
                
        return None

    # Try alpha=4
    sol = find_solution_for_alpha(4)
    if sol:
        return sol
        
    # Fallback to alpha=5 greedy
    E = []
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    for edge in shuffled:
        edge_set = set(edge)
        ok = True
        for e in E:
            if len(edge_set & set(e)) > 1:
                ok = False
                break
        if ok:
            E.append(edge)
            
    return [list(e) for e in E]
