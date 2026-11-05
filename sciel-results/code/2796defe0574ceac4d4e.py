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
    
    # Precompute all triples as bit-encoded integers for fast hashing
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
        # Fix the first point, pair it with each of the other 5, etc.
        # Actually, let's just enumerate all ways to partition 6 elements into 3 pairs.
        # There are 15 such partitions.
        # Method: generate all permutations and take canonical form, dedupe.
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
            # Convert to sorted tuple for consistent representation
            t_indices = tuple(sorted(t_indices))
            transversals.append(t_indices)
    
    n_transversals = len(transversals)
    
    # Build a mapping from each triple index to the list of transversal indices it belongs to
    triple_to_transversals = [[] for _ in range(n_triples)]
    for ti, trans in enumerate(transversals):
        for t_idx in trans:
            triple_to_transversals[t_idx].append(ti)
    
    # Convert to numpy arrays for faster operations
    # Represent current state as a boolean numpy array
    current_set = np.ones(n_triples, dtype=bool)
    
    # For each transversal, track how many of its 8 triples are currently present
    # conflict_count[ti] = number of triples in transversal ti that are in current_set
    conflict_count = np.array([8] * n_transversals, dtype=np.int32)
    
    def is_valid():
        return np.all(conflict_count < 8)
    
    # Greedy removal: remove triples to eliminate all conflicts
    # We use a priority-based approach: repeatedly find a conflicting transversal,
    # and remove the triple from it that appears in the most other conflicting transversals.
    
    def find_best_removal():
        """Find the triple to remove that breaks the most conflicts."""
        # Find all conflicting transversals
        conflicting = np.where(conflict_count == 8)[0]
        if len(conflicting) == 0:
            return None
        
        # For each conflicting transversal, consider removing each of its 8 triples.
        # Score = number of other conflicting transversals that also contain that triple.
        best_score = -1
        best_triple = None
        best_trans = None
        
        # Sample a subset of conflicting transversals if there are too many
        if len(conflicting) > 200:
            sample = np.random.choice(conflicting, size=200, replace=False)
        else:
            sample = conflicting
        
        for trans_idx in sample:
            trans = transversals[trans_idx]
            for t_idx in trans:
                # Count how many other conflicting transversals contain t_idx
                # We can approximate by checking triple_to_transversals[t_idx]
                # and counting how many are in the conflicting set
                # But that's expensive. Let's just use a heuristic:
                # Remove the triple that appears in the most transversals overall
                # (or in the most conflicting ones - approximate with overall)
                score = len(triple_to_transversals[t_idx])
                if score > best_score:
                    best_score = score
                    best_triple = t_idx
                    best_trans = trans_idx
        
        return best_triple, best_trans
    
    # Initial phase: aggressively remove to get a valid configuration
    iteration = 0
    while time.time() < deadline - 60:  # Leave time for local search
        if is_valid():
            break
        
        removal = find_best_removal()
        if removal is None:
            break
        
        t_idx, trans_idx = removal
        # Remove the triple
        current_set[t_idx] = False
        # Update conflict counts for all transversals containing t_idx
        for trans_j in triple_to_transversals[t_idx]:
            conflict_count[trans_j] -= 1
        
        iteration += 1
    
    # Local search: try to add back triples while maintaining validity
    # Use simulated annealing or simple hill climbing
    
    rng = np.random.default_rng(42)
    
    # Get list of absent triples
    absent_triples = np.where(~current_set)[0]
    rng.shuffle(absent_triples)
    
    best_count = np.sum(current_set)
    best_state = current_set.copy()
    
    # Try adding triples one by one, in random order
    for t_idx in absent_triples:
        if time.time() > deadline - 10:
            break
        
        # Check if adding this triple would create any conflict
        # A conflict is created if any transversal containing t_idx already has 7 other triples present
        can_add = True
        for trans_j in triple_to_transversals[t_idx]:
            if conflict_count[trans_j] == 7:
                can_add = False
                break
        
        if can_add:
            # Add it
            current_set[t_idx] = True
            for trans_j in triple_to_transversals[t_idx]:
                conflict_count[trans_j] += 1
            
            # Try to also add other triples that might now be addable?
            # Actually, adding only makes it harder, so we just continue.
    
    # Local search: try swapping - remove one triple, add another
    for _ in range(500):
        if time.time() > deadline - 10:
            break
        
        if not is_valid():
            # Fix any conflicts first
            removal = find_best_removal()
            if removal is None:
                break
            t_idx, _ = removal
            current_set[t_idx] = False
            for trans_j in triple_to_transversals[t_idx]:
                conflict_count[trans_j] -= 1
            continue
        
        # Try a random swap: remove one present triple, try to add two absent ones
        present = np.where(current_set)[0]
        absent = np.where(~current_set)[0]
        
        if len(present) == 0 or len(absent) < 2:
            break
        
        remove_idx = rng.choice(present)
        
        # Save state
        old_state = current_set.copy()
        old_conflict = conflict_count.copy()
        
        # Remove
        current_set[remove_idx] = False
        for trans_j in triple_to_transversals[remove_idx]:
            conflict_count[trans_j] -= 1
        
        added = 0
        for add_idx in rng.choice(absent, size=min(len(absent), 50), replace=False):
            can_add = True
            for trans_j in triple_to_transversals[add_idx]:
                if conflict_count[trans_j] == 7:
                    can_add = False
                    break
            if can_add:
                current_set[add_idx] = True
                for trans_j in triple_to_transversals[add_idx]:
                    conflict_count[trans_j] += 1
                added += 1
        
        new_count = np.sum(current_set)
        if new_count > best_count:
            best_count = new_count
            best_state = current_set.copy()
        else:
            # Revert
            current_set = old_state
            conflict_count = old_conflict
    
    # Use the best state found
    final_set = best_state
    
    # Convert back to list of triples
    result = [list(all_triples[i]) for i in range(n_triples) if final_set[i]]
    
    return result
