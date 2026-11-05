import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 16
    
    # Precompute all triples as bitmasks and constituent points
    all_triples = []
    all_triple_masks = []
    for comb in itertools.combinations(range(N), 3):
        mask = (1 << comb[0]) | (1 << comb[1]) | (1 << comb[2])
        all_triples.append(comb)
        all_triple_masks.append(mask)
    
    m = len(all_triples)
    
    # Precompute all pairs
    all_pairs = list(itertools.combinations(range(N), 2))
    p = len(all_pairs)
    pair_index = {pair: i for i, pair in enumerate(all_pairs)}
    
    # Build constraint matrix for linearity: each pair appears in at most one triple
    A = np.zeros((p, m), dtype=np.int8)
    for j, tri in enumerate(all_triples):
        A[pair_index[(tri[0], tri[1])], j] = 1
        A[pair_index[(tri[0], tri[2])], j] = 1
        A[pair_index[(tri[1], tri[2])], j] = 1
    
    def calc_alpha(triple_masks):
        """Calculate independence number using bitmask check over all subsets."""
        if not triple_masks:
            return N
        
        max_size = 0
        # Check subsets from size N down to 1, early exit if we find large independent set
        # But we want the MAXIMUM, so we need to check all
        # Optimization: iterate all 2^16 masks, track max popcount of independent ones
        for mask in range(1, 1 << N):
            size = mask.bit_count()
            if size <= max_size:
                continue
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                max_size = size
        return max_size
    
    def count_independent_k_sets(triple_masks, k):
        """Count number of k-subsets that contain no triple."""
        count = 0
        for comb in itertools.combinations(range(N), k):
            mask = 0
            for pnt in comb:
                mask |= (1 << pnt)
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                count += 1
        return count
    
    # Phase 1: MILP to find maximum linear packing
    best_triples = []
    best_alpha = N
    
    try:
        c = np.full(m, -1.0)  # Maximize sum(x)
        constraints = LinearConstraint(A, lb=-np.inf, ub=1.0)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(m)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if res.status == 0 and res.x is not None:
            x = res.x
            selected = np.where(x > 0.5)[0]
            best_triples = [list(all_triples[i]) for i in selected]
            best_masks = [all_triple_masks[i] for i in selected]
            best_alpha = calc_alpha(best_masks)
            
    except Exception:
        # Fallback: greedy random packing
        best_triples = []
        best_masks = []
        used_pairs = set()
        candidates = list(range(m))
        random.shuffle(candidates)
        for idx in candidates:
            tri = all_triples[idx]
            pairs = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
            if all(p not in used_pairs for p in pairs):
                best_triples.append(list(tri))
                best_masks.append(all_triple_masks[idx])
                for p in pairs:
                    used_pairs.add(p)
        best_alpha = calc_alpha(best_masks)
    
    if time.time() > deadline - 10:
        return best_triples
    
    # Phase 2: Local search / Simulated Annealing to minimize alpha
    current_triples = [t[:] for t in best_triples]
    current_masks = best_masks[:]
    current_alpha = best_alpha
    
    temp = 1.0
    cooling_rate = 0.995
    
    # Precompute for fast linearity check: for each triple, store which other triples conflict
    # A triple conflicts if they share 2+ points
    # We'll maintain a set of used pairs for O(1) linearity check
    
    def get_used_pairs(triples):
        pairs = set()
        for t in triples:
            pairs.add((t[0], t[1]))
            pairs.add((t[0], t[2]))
            pairs.add((t[1], t[2]))
        return pairs
    
    used_pairs = get_used_pairs(current_triples)
    
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        
        if not current_masks:
            break
        
        remove_idx = random.randint(0, len(current_masks) - 1)
        removed_mask = current_masks[remove_idx]
        removed_triple = current_triples[remove_idx]
        
        # Remove from used_pairs
        for i in range(3):
            for j in range(i + 1, 3):
                used_pairs.discard((removed_triple[i], removed_triple[j]))
        
        # Find a compatible replacement
        candidate_mask = None
        candidate_triple = None
        for _ in range(30):
            cand_idx = random.randint(0, m - 1)
            cand_mask = all_triple_masks[cand_idx]
            if cand_mask == removed_mask:
                continue
            cand_triple = all_triples[cand_idx]
            # Check linearity with remaining triples
            compatible = True
            for i in range(3):
                for j in range(i + 1, 3):
                    if (cand_triple[i], cand_triple[j]) in used_pairs:
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                candidate_mask = cand_mask
                candidate_triple = list(cand_triple)
                break
        
        if candidate_mask is None:
            # Revert
            for i in range(3):
                for j in range(i + 1, 3):
                    used_pairs.add((removed_triple[i], removed_triple[j]))
            continue
        
        # Perform swap
        old_triple = current_triples[remove_idx]
        current_triples[remove_idx] = candidate_triple
        current_masks[remove_idx] = candidate_mask
        
        for i in range(3):
            for j in range(i + 1, 3):
                used_pairs.add((candidate_triple[i], candidate_triple[j]))
        
        new_alpha = calc_alpha(current_masks)
        
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_triples = [t[:] for t in current_triples]
            if best_alpha <= 7:
                return best_triples
        
        delta = new_alpha - current_alpha
        if delta < 0:
            current_alpha = new_alpha
        else:
            if random.random() < pow(temp, -delta):
                current_alpha = new_alpha
            else:
                # Revert
                current_triples[remove_idx] = old_triple
                current_masks[remove_idx] = removed_mask
                for i in range(3):
                    for j in range(i + 1, 3):
                        used_pairs.discard((candidate_triple[i], candidate_triple[j]))
                for i in range(3):
                    for j in range(i + 1, 3):
                        used_pairs.add((old_triple[i], old_triple[j]))
        
        temp *= cooling_rate
        if temp < 0.01:
            temp = 0.01
    
    return best_triples
