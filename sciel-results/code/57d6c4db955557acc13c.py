import itertools
import random
import time
import numpy as np
from numpy.random import default_rng

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    points = list(range(16))
    n_points = 16

    # Precompute all triples
    all_triples = []
    triple_to_idx = {}
    for a, b, c in itertools.combinations(points, 3):
        t = (a, b, c)
        idx = len(all_triples)
        all_triples.append(t)
        triple_to_idx[t] = idx

    n_triples = len(all_triples)

    # Precompute transversals: each is a set of 8 triple indices
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate all 15 perfect matchings on these 6 points
        matchings = set()
        for perm in itertools.permutations(range(6)):
            m = frozenset([
                tuple(sorted((sp[perm[0]], sp[perm[1]]))),
                tuple(sorted((sp[perm[2]], sp[perm[3]]))),
                tuple(sorted((sp[perm[4]], sp[perm[5]])))
            ])
            m = tuple(sorted(m))
            matchings.add(m)

        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            t_indices = []
            for x in (a, b):
                for y in (c, d):
                    for z in (e, f):
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(triple_to_idx[t])
            t_indices = tuple(sorted(t_indices))
            transversals.append(t_indices)

    n_transversals = len(transversals)

    # Build mapping from triple index to list of transversal indices
    triple_to_transversals = [[] for _ in range(n_triples)]
    for ti, trans in enumerate(transversals):
        for t_idx in trans:
            triple_to_transversals[t_idx].append(ti)

    # Convert transversals to numpy array for fast operations
    trans_arr = np.array(transversals, dtype=np.int32)
    
    # Precompute degree of each triple (number of transversals it belongs to)
    triple_degree = np.array([len(triple_to_transversals[i]) for i in range(n_triples)], dtype=np.int32)

    def greedy_remove(seed=42):
        """Greedy removal: start with all triples, remove to resolve conflicts."""
        rng = random.Random(seed)
        present = np.ones(n_triples, dtype=np.uint8)
        
        # Shuffle transversals for randomization
        trans_order = list(range(n_transversals))
        rng.shuffle(trans_order)
        
        # Count conflicts and remove
        for ti in trans_order:
            if time.time() > deadline - 5:
                break
            trans = trans_arr[ti]
            # Check if all 8 are present
            if np.all(present[trans]):
                # Find the triple with highest degree among the 8
                degrees = triple_degree[trans]
                best_local = np.argmax(degrees)
                best_idx = trans[best_local]
                present[best_idx] = 0
        
        return present

    def local_search_swap(present, seed=0, max_iter=5000):
        """Local search: try to swap a removed triple with a present one to increase count."""
        rng = random.Random(seed)
        present = present.copy()
        current_count = np.sum(present)
        
        removed = np.where(present == 0)[0]
        if len(removed) == 0:
            return present, current_count
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            if len(removed) == 0:
                break
            
            # Pick a random removed triple to try to add back
            idx = removed[rng.randrange(len(removed))]
            
            # Check if adding idx creates a conflict
            can_add = True
            for ti in triple_to_transversals[idx]:
                trans = trans_arr[ti]
                # Check if all other 7 are present
                others = np.setdiff1d(trans, idx)
                if np.all(present[others]):
                    can_add = False
                    break
            
            if can_add:
                present[idx] = 1
                current_count += 1
                removed = np.where(present == 0)[0]
            else:
                # Try to find a present triple in one of the conflicting transversals to remove
                # to make room for idx
                for ti in triple_to_transversals[idx]:
                    trans = trans_arr[ti]
                    others = np.setdiff1d(trans, idx)
                    if np.all(present[others]):
                        # This transversal is the conflict source
                        # Pick one to remove - prefer low degree
                        degrees = triple_degree[others]
                        # Among those that are present
                        present_mask = present[others]
                        if not np.any(present_mask):
                            continue
                        # Get indices of present ones
                        present_others = others[present_mask]
                        if len(present_others) == 0:
                            continue
                        # Pick the one with lowest degree
                        min_deg_idx = np.argmin(triple_degree[present_others])
                        remove_idx = present_others[min_deg_idx]
                        
                        # Perform swap: remove remove_idx, add idx
                        present[remove_idx] = 0
                        present[idx] = 1
                        # Update removed list
                        removed = np.where(present == 0)[0]
                        break
        
        return present, current_count

    def local_search_add_back(present, seed=0):
        """Try to add back removed triples that don't create conflicts."""
        rng = random.Random(seed)
        present = present.copy()
        
        removed = np.where(present == 0)[0]
        rng.shuffle(removed)
        
        for idx in removed:
            if time.time() > deadline - 5:
                break
            # Check if adding idx creates a conflict
            can_add = True
            for ti in triple_to_transversals[idx]:
                trans = trans_arr[ti]
                others = np.setdiff1d(trans, idx)
                if np.all(present[others]):
                    can_add = False
                    break
            
            if can_add:
                present[idx] = 1
        
        return present

    # Try multiple greedy seeds
    best_present = None
    best_count = 0
    
    seeds = [42, 123, 456, 789, 1000, 2024, 7, 13, 23, 37]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        present = greedy_remove(seed)
        count = np.sum(present)
        
        # Try to improve with add-back
        present2 = local_search_add_back(present, seed + 1000)
        count2 = np.sum(present2)
        
        if count2 > best_count:
            best_count = count2
            best_present = present2.copy()
    
    # Try local search swap from the best solution
    if best_present is not None and time.time() < deadline - 15:
        # Try several swap-based local searches
        for swap_seed in [100, 200, 300, 400, 500]:
            if time.time() > deadline - 10:
                break
            present_swapped, count_swapped = local_search_swap(best_present, swap_seed, max_iter=3000)
            if count_swapped > best_count:
                best_count = count_swapped
                best_present = present_swapped.copy()
                # Try add-back again after swap
                present_add = local_search_add_back(best_present, swap_seed + 2000)
                count_add = np.sum(present_add)
                if count_add > best_count:
                    best_count = count_add
                    best_present = present_add.copy()
    
    # Final add-back pass
    if best_present is not None and time.time() < deadline - 5:
        best_present = local_search_add_back(best_present, 9999)
        best_count = np.sum(best_present)

    if best_present is not None:
        result = [list(all_triples[i]) for i in range(n_triples) if best_present[i]]
    else:
        # Fallback
        result = [list(t) for t in all_triples]
    
    return result
