import itertools
import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    all_triples_set = set(all_triples)
    
    # Map each triple to an index for faster lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute transversals: each is a tuple of 8 triple indices
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate 15 perfect matchings using a standard method
        # Fix first element's partner, then second, etc.
        # Use a set to avoid duplicates
        matchings = set()
        for i in range(6):
            for j in range(i+1, 6):
                if i == 0 and j != 1:
                    continue  # Fix first pair to (0,1) to avoid duplicates
                # Actually, let's use a simpler approach:
                # Generate all 15 matchings by a known method
                pass
        
        # Simpler: use the permutation approach but deduplicate
        seen = set()
        for perm in itertools.permutations(range(6)):
            # Create canonical form: sort pairs, then sort list of pairs
            pairs = [tuple(sorted((sp[perm[0]], sp[perm[1]]))),
                     tuple(sorted((sp[perm[2]], sp[perm[3]]))),
                     tuple(sorted((sp[perm[4]], sp[perm[5]])))]
            pairs.sort()
            key = tuple(pairs)
            if key not in seen:
                seen.add(key)
                (p1, p2, p3) = key
                a, b = p1
                c, d = p2
                e, f = p3
                t1 = tuple(sorted((a, c, e)))
                t2 = tuple(sorted((a, c, f)))
                t3 = tuple(sorted((a, d, e)))
                t4 = tuple(sorted((a, d, f)))
                t5 = tuple(sorted((b, c, e)))
                t6 = tuple(sorted((b, c, f)))
                t7 = tuple(sorted((b, d, e)))
                t8 = tuple(sorted((b, d, f)))
                # Convert to indices
                idxs = tuple(triple_to_idx[t] for t in (t1, t2, t3, t4, t5, t6, t7, t8))
                transversals.append(idxs)
    
    num_transversals = len(transversals)
    
    # For each triple, precompute which transversals it belongs to
    triple_transversals = [[] for _ in range(num_triples)]
    for ti, trans in enumerate(transversals):
        for idx in trans:
            triple_transversals[idx].append(ti)
    
    # Greedy removal with smart strategy: remove the triple in the conflict that has the highest degree
    # (i.e., appears in the most other transversals that are currently "at risk")
    
    def greedy_remove(seed=42):
        rng = random.Random(seed)
        present = bytearray(num_triples)  # 1 if present, 0 if removed
        for i in range(num_triples):
            present[i] = 1
        
        # Shuffle transversals for randomization
        trans_order = list(range(num_transversals))
        rng.shuffle(trans_order)
        
        # Count how many transversals are currently conflicting (all 8 present)
        # We'll do a pass, and for each conflict, remove the triple with max degree among the 8
        
        progress = True
        while progress:
            if time.time() > deadline:
                break
            progress = False
            for ti in trans_order:
                trans = transversals[ti]
                # Check if all 8 are present
                if all(present[idx] for idx in trans):
                    # Conflict found
                    # Find the triple in this conflict that appears in the most other transversals
                    # (to minimize future conflicts)
                    best_idx = -1
                    best_degree = -1
                    for idx in trans:
                        degree = len(triple_transversals[idx])
                        if degree > best_degree:
                            best_degree = degree
                            best_idx = idx
                    # Remove it
                    present[best_idx] = 0
                    progress = True
                    # Continue checking (don't break, to find more conflicts in same pass)
        
        return sum(present)
    
    # Try multiple seeds and keep the best
    best_count = 0
    best_present = None
    
    # First, try a few seeds
    for seed in [42, 123, 456, 789, 1000, 2024]:
        if time.time() > deadline - 5:
            break
        count = greedy_remove(seed)
        if count > best_count:
            best_count = count
            # Re-run to get the actual set
            rng = random.Random(seed)
            present = bytearray(num_triples)
            for i in range(num_triples):
                present[i] = 1
            trans_order = list(range(num_transversals))
            rng.shuffle(trans_order)
            progress = True
            while progress:
                if time.time() > deadline:
                    break
                progress = False
                for ti in trans_order:
                    trans = transversals[ti]
                    if all(present[idx] for idx in trans):
                        best_idx = -1
                        best_degree = -1
                        for idx in trans:
                            degree = len(triple_transversals[idx])
                            if degree > best_degree:
                                best_degree = degree
                                best_idx = idx
                        present[best_idx] = 0
                        progress = True
            best_present = present
    
    # Now try to add back removed triples (local search)
    if best_present is not None:
        if time.time() < deadline - 10:
            # Build the current set
            current_set = set()
            for i in range(num_triples):
                if best_present[i]:
                    current_set.add(all_triples[i])
            
            # Try to add back each removed triple
            removed_indices = [i for i in range(num_triples) if not best_present[i]]
            rng = random.Random(999)
            rng.shuffle(removed_indices)
            
            for idx in removed_indices:
                if time.time() > deadline - 5:
                    break
                triple = all_triples[idx]
                # Check if adding this triple creates any conflict
                # A conflict is created if there exists a transversal containing this triple
                # where the other 7 triples are all present
                can_add = True
                for ti in triple_transversals[idx]:
                    trans = transversals[ti]
                    # Check if all other 7 are present
                    all_others_present = True
                    for other_idx in trans:
                        if other_idx != idx and not best_present[other_idx]:
                            all_others_present = False
                            break
                    if all_others_present:
                        can_add = False
                        break
                
                if can_add:
                    best_present[idx] = 1
                    current_set.add(triple)
    
    if best_present is not None:
        result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
    else:
        # Fallback: return all triples (shouldn't happen, but safe)
        result = [list(t) for t in all_triples]
    
    return result

# EVOLVE-BLOCK-END
