"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    n_triples = len(all_triples)  # 560
    
    # Precompute conflict graph: two triples conflict if they share >= 2 points
    conflict = [0] * n_triples
    for i in range(n_triples):
        a, b, c = all_triples[i]
        for j in range(i + 1, n_triples):
            d, e, f = all_triples[j]
            shared = len({a, b, c} & {d, e, f})
            if shared >= 2:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Precompute point masks for each triple
    triple_points_mask = [0] * n_triples
    for i, (a, b, c) in enumerate(all_triples):
        triple_points_mask[i] = (1 << a) | (1 << b) | (1 << c)
    
    # Precompute all subset masks grouped by size
    subsets_by_size = {}
    for size in range(1, n + 1):
        subsets_by_size[size] = []
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            subsets_by_size[size].append(mask)
    
    def independence_number_fast(triple_indices):
        """Find largest subset of points containing no complete triple from triple_indices."""
        if not triple_indices:
            return n
        
        triple_set = set(triple_indices)
        
        for size in range(n, 0, -1):
            for mask in subsets_by_size[size]:
                is_indep = True
                for ti in triple_set:
                    if (mask & triple_points_mask[ti]) == triple_points_mask[ti]:
                        is_indep = False
                        break
                if is_indep:
                    return size
        return 0
    
    def greedy_dense_packing(seed):
        """Greedy packing using conflict graph to maximize number of triples."""
        random.seed(seed)
        
        selected = set()
        remaining = list(range(n_triples))
        
        while remaining:
            compatible = []
            for ti in remaining:
                is_compatible = True
                for si in selected:
                    if conflict[ti] & (1 << si):
                        is_compatible = False
                        break
                if is_compatible:
                    compatible.append(ti)
            
            if not compatible:
                break
            
            covered_points = set()
            for si in selected:
                a, b, c = all_triples[si]
                covered_points.add(a)
                covered_points.add(b)
                covered_points.add(c)
            
            scored = []
            for ti in compatible:
                a, b, c = all_triples[ti]
                new_points = sum(1 for p in (a, b, c) if p not in covered_points)
                scored.append((new_points, ti))
            
            max_new = max(s[0] for s in scored)
            best_candidates = [ti for new_pts, ti in scored if new_pts == max_new]
            chosen = random.choice(best_candidates)
            
            selected.add(chosen)
            remaining.remove(chosen)
            
            to_remove = set()
            for ti in remaining:
                if conflict[ti] & (1 << chosen):
                    to_remove.add(ti)
            remaining = [ti for ti in remaining if ti not in to_remove]
        
        return list(selected)
    
    def local_search_improve(triple_indices, max_iters=200):
        """Improve the independence number by swapping triples."""
        triples = list(triple_indices)
        current_indep = independence_number_fast(triples)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            if not triples:
                break
            
            idx = random.randrange(len(triples))
            old_ti = triples[idx]
            triples.pop(idx)
            
            base_conflict_mask = 0
            for ti in triples:
                base_conflict_mask |= conflict[ti]
            
            candidates = []
            for ti in range(n_triples):
                if ti == old_ti:
                    continue
                if base_conflict_mask & (1 << ti):
                    continue
                candidates.append(ti)
            
            if not candidates:
                triples.append(old_ti)
                continue
            
            random.shuffle(candidates)
            sample_size = min(30, len(candidates))
            
            best_new_indep = current_indep
            best_candidate = None
            
            for ti in candidates[:sample_size]:
                test_triples = triples + [ti]
                test_indep = independence_number_fast(test_triples)
                if test_indep < best_new_indep:
                    best_new_indep = test_indep
                    best_candidate = ti
            
            if best_candidate is not None:
                triples.append(best_candidate)
                current_indep = best_new_indep
                improved = True
            else:
                triples.append(old_ti)
            
            if not improved:
                break
        
        return triples, current_indep
    
    # Main search loop
    best_triples = None
    best_indep = n + 1
    
    for seed in range(300):
        if time.time() > deadline - 15:
            break
        
        triple_indices = greedy_dense_packing(seed)
        
        if not triple_indices:
            continue
        
        indep = independence_number_fast(triple_indices)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = list(triple_indices)
        
        if indep <= 8 and time.time() < deadline - 30:
            improved_triples, improved_indep = local_search_improve(triple_indices, max_iters=100)
            if improved_indep < best_indep:
                best_indep = improved_indep
                best_triples = improved_triples
        
        if best_indep <= 6:
            break
    
    if best_triples is not None:
        result = [list(all_triples[ti]) for ti in best_triples]
    else:
        result = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return result
