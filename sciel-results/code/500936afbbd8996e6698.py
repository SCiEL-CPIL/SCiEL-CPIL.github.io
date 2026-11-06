import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Pre-compute all transversals (forbidden configurations)
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = []
        remaining = [1, 2, 3, 4, 5]
        for j in remaining:
            rest = [x for x in remaining if x != j]
            for k in range(3):
                if k == 0:
                    p2, p3 = (rest[0], rest[1]), (rest[2], rest[3])
                elif k == 1:
                    p2, p3 = (rest[0], rest[2]), (rest[1], rest[3])
                else:
                    p2, p3 = (rest[0], rest[3]), (rest[1], rest[2])
                matching = [(sp[0], sp[j]), (sp[p2[0]], sp[p2[1]]), (sp[p3[0]], sp[p3[1]])]
                matchings.append(matching)
        
        for matching in matchings:
            (a, b), (c, d), (e, f) = matching
            t1 = tuple(sorted((a, c, e)))
            t2 = tuple(sorted((a, c, f)))
            t3 = tuple(sorted((a, d, e)))
            t4 = tuple(sorted((a, d, f)))
            t5 = tuple(sorted((b, c, e)))
            t6 = tuple(sorted((b, c, f)))
            t7 = tuple(sorted((b, d, e)))
            t8 = tuple(sorted((b, d, f)))
            idxs = tuple(triple_to_idx[t] for t in (t1, t2, t3, t4, t5, t6, t7, t8))
            transversals.append(idxs)
    
    num_transversals = len(transversals)
    
    # For each triple, precompute which transversals it belongs to
    triple_transversals = [[] for _ in range(num_triples)]
    for ti, trans in enumerate(transversals):
        for idx in trans:
            triple_transversals[idx].append(ti)
    
    # Precompute degree of each triple
    triple_degree = np.array([len(tt) for tt in triple_transversals], dtype=np.int32)
    
    # Spectral initialization: Build adjacency tensor and compute local density
    # For each triple, compute a "density score" based on how many of its transversals
    # have high total degree (indicating conflict-prone regions)
    def spectral_init():
        # Compute for each triple: sum of degrees of co-transversal triples
        # This identifies triples that are in "dense" conflict regions
        density_score = np.zeros(num_triples, dtype=np.float64)
        for i in range(num_triples):
            for ti in triple_transversals[i]:
                trans = transversals[ti]
                for j in trans:
                    if j != i:
                        density_score[i] += triple_degree[j]
        
        # Normalize and sort by density score (descending)
        # Triples with lower density are "safer" to include
        order = np.argsort(density_score)  # ascending: low density first
        return order
    
    # Fast conflict check using bitsets
    def resolve_conflicts(present, order=None):
        """Iteratively remove triples to eliminate all K(2,2,2) configurations."""
        if order is None:
            order = list(range(num_transversals))
        
        # Use a set for fast lookup of present transversals
        # Track for each transversal how many of its 8 triples are present
        trans_count = np.zeros(num_transversals, dtype=np.int32)
        for i in range(num_triples):
            if present[i]:
                for ti in triple_transversals[i]:
                    trans_count[ti] += 1
        
        # Find all transversals with count == 8
        conflict_trans = np.where(trans_count == 8)[0]
        
        # Process conflicts
        changed = True
        while changed:
            if time.time() > deadline - 10:
                break
            changed = False
            for ti in conflict_trans:
                if time.time() > deadline - 10:
                    break
                trans = transversals[ti]
                # Check if still all 8 present
                if all(present[idx] for idx in trans):
                    # Remove the triple with highest degree
                    best_idx = max(trans, key=lambda x: triple_degree[x])
                    present[best_idx] = 0
                    # Update counts for all transversals containing best_idx
                    for ti2 in triple_transversals[best_idx]:
                        trans_count[ti2] -= 1
                    changed = True
                    # Re-find conflicts
                    conflict_trans = np.where(trans_count == 8)[0]
        
        return present
    
    # Local search: try to add back removed triples
    def local_search_add(present):
        """Try to add back removed triples if possible."""
        removed_indices = [i for i in range(num_triples) if not present[i]]
        rng = random.Random(999)
        rng.shuffle(removed_indices)
        
        added = 0
        for idx in removed_indices:
            if time.time() > deadline - 10:
                break
            # Check if adding this triple creates any conflict
            can_add = True
            for ti in triple_transversals[idx]:
                trans = transversals[ti]
                # Check if all other 7 are present
                all_others_present = True
                for other_idx in trans:
                    if other_idx != idx and not present[other_idx]:
                        all_others_present = False
                        break
                if all_others_present:
                    can_add = False
                    break
            
            if can_add:
                present[idx] = 1
                added += 1
        
        return present, added
    
    # Multi-start local search with spectral initialization
    best_count = 0
    best_present = None
    
    # Get spectral ordering
    spec_order = spectral_init()
    
    # Try multiple strategies
    seeds = [42, 123, 456, 789, 1000, 2024, 7, 13, 99, 314, 777, 888, 555, 666, 111]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(seed)
        present = bytearray(num_triples)
        for i in range(num_triples):
            present[i] = 1
        
        # Use spectral ordering to guide removal
        # Start from low-density triples and work up
        trans_order = list(range(num_transversals))
        rng.shuffle(trans_order)
        
        present = resolve_conflicts(present, trans_order)
        present, added = local_search_add(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = bytearray(present)
    
    # Additional local search passes on the best solution
    if best_present is not None:
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            best_present, added = local_search_add(best_present)
            if added == 0:
                break
    
    # Final attempt: try removing and re-adding to escape local optima
    if best_present is not None and time.time() < deadline - 20:
        rng = random.Random(2024)
        removed = [i for i in range(num_triples) if best_present[i]]
        rng.shuffle(removed)
        
        # Try removing a few random triples and see if we can add back more
        for num_remove in [1, 2, 3]:
            if time.time() > deadline - 15:
                break
            test_present = bytearray(best_present)
            for idx in removed[:num_remove]:
                test_present[idx] = 0
            
            test_present, added = local_search_add(test_present)
            test_count = sum(test_present)
            
            if test_count > best_count:
                best_count = test_count
                best_present = test_present
    
    # Build result
    if best_present is not None:
        result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
    else:
        result = [list(t) for t in all_triples]
    
    return result
