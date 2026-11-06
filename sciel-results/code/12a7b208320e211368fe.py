"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is formed by three disjoint pairs (a,a'), (b,b'), (c,c')
    # where a,a',b,b',c,c' are all distinct
    # The 8 transversal triples are all combinations choosing one from each pair
    # We need to find all such configurations
    
    # For efficiency, precompute for each pair of points, which triples contain both
    # Actually, let's think differently:
    # A K(2,2,2) consists of 6 distinct points partitioned into 3 pairs
    # and 8 triples (one from each "side" of the 3 pairs)
    # We need to detect if all 8 triples are present
    
    # Precompute: for each triple, store its index
    # For each set of 6 points, check if it forms a K(2,2,2)
    
    # Better approach: precompute all K(2,2,2) patterns
    # Each K(2,2,2) is defined by 3 disjoint pairs
    # Number of ways to choose 6 points from 16: C(16,6) = 8008
    # For each 6-set, number of ways to partition into 3 pairs: 15
    # So total K(2,2,2) patterns: 8008 * 15 = 120120
    
    # But we can be smarter. Let's precompute for each 6-point subset
    # whether it contains a K(2,2,2) configuration, and which triples are needed
    
    # Actually, let's use a different approach:
    # For each triple, precompute which K(2,2,2) patterns it belongs to
    # This is expensive but let's try
    
    # Alternative: use bitmask representation
    # Represent each triple as a bitmask of 3 bits in a 16-bit number
    # Use a dictionary to map triple -> index
    
    # Let's precompute all K(2,2,2) configurations
    # A K(2,2,2) is determined by 6 distinct points and a partition into 3 pairs
    # For 6 points p0..p5, partitions into 3 pairs:
    # (p0,p1)(p2,p3)(p4,p5), (p0,p1)(p2,p4)(p3,p5), etc.
    # There are 15 such partitions
    
    k222_patterns = []  # list of (tuple_of_8_triples, frozenset_of_6_points)
    
    def get_partitions_into_pairs(six_points):
        """Generate all ways to partition 6 points into 3 pairs"""
        p0, p1, p2, p3, p4, p5 = six_points
        # Fix p0's partner, then partition remaining 4
        yield (p0, p1), (p2, p3), (p4, p5)
        yield (p0, p1), (p2, p4), (p3, p5)
        yield (p0, p1), (p2, p5), (p3, p4)
        yield (p0, p2), (p1, p3), (p4, p5)
        yield (p0, p2), (p1, p4), (p3, p5)
        yield (p0, p2), (p1, p5), (p3, p4)
        yield (p0, p3), (p1, p2), (p4, p5)
        yield (p0, p3), (1, p4), (p3, p5)  # bug, fix below
        yield (p0, p3), (p1, p5), (p2, p4)
        yield (p0, p4), (p1, p2), (p3, p5)
        yield (p0, p4), (p1, p3), (p2, p5)
        yield (p0, p4), (p1, p5), (p2, p3)
        yield (p0, p5), (p1, p2), (p3, p4)
        yield (p0, p5), (p1, p3), (p2, p4)
        yield (p0, p5), (p1, p4), (p2, p3)
    
    # Fix the bug in the generator
    def get_partitions_into_pairs_fixed(six_points):
        p0, p1, p2, p3, p4, p5 = six_points
        yield ((p0,p1), (p2,p3), (p4,p5))
        yield ((p0,p1), (p2,p4), (p3,p5))
        yield ((p0,p1), (p2,p5), (p3,p4))
        yield ((p0,p2), (p1,p3), (p4,p5))
        yield ((p0,p2), (p1,p4), (p3,p5))
        yield ((p0,p2), (p1,p5), (p3,p4))
        yield ((p0,p3), (p1,p2), (p4,p5))
        yield ((p0,p3), (p1,p4), (p2,p5))
        yield ((p0,p3), (p1,p5), (p2,p4))
        yield ((p0,p4), (p1,p2), (p3,p5))
        yield ((p0,p4), (p1,p3), (p2,p5))
        yield ((p0,p4), (p1,p5), (p2,p3))
        yield ((p0,p5), (p1,p2), (p3,p4))
        yield ((p0,p5), (p1,p3), (p2,p4))
        yield ((p0,p5), (p1,p4), (p2,p3))
    
    # Precompute all K(2,2,2) patterns
    # For each 6-point subset, for each partition into 3 pairs,
    # compute the 8 transversal triples
    
    triple_to_k222s = defaultdict(list)  # triple_idx -> list of k222_indices
    
    k222_triple_sets = []  # each element is a frozenset of 8 triple indices
    
    for six_points in combinations(range(n), 6):
        for pairs in get_partitions_into_pairs_fixed(six_points):
            # Compute the 8 transversal triples
            # pairs = ((a,a'), (b,b'), (c,c'))
            (a, a_p), (b, b_p), (c, c_p) = pairs
            transversal_triples = [
                (a, b, c), (a, b, c_p), (a, b_p, c), (a, b_p, c_p),
                (a_p, b, c), (a_p, b, c_p), (a_p, b_p, c), (a_p, b_p, c_p)
            ]
            # Sort each triple
            transversal_triples = [tuple(sorted(t)) for t in transversal_triples]
            # Get indices
            triple_indices = []
            valid = True
            for t in transversal_triples:
                if t in triple_to_idx:
                    triple_indices.append(triple_to_idx[t])
                else:
                    valid = False
                    break
            if valid:
                k222_triple_sets.append(frozenset(triple_indices))
                for idx in triple_indices:
                    triple_to_k222s[idx].append(len(k222_triple_sets) - 1)
    
    num_k222s = len(k222_triple_sets)
    
    # Now we have all K(2,2,2) patterns
    # Use greedy + local search to maximize the number of triples
    
    # Greedy seed: start with all triples, remove those that create K(2,2,2)
    # Actually, start empty and add triples greedily
    
    # Better: start with a random subset and use local search
    
    # Let's use simulated annealing
    
    # State: boolean array of length num_triples
    # Energy: -count + penalty * num_violations
    
    # Use bit-parallel conflict tracking
    
    # For efficiency, track for each K(2,2,2) how many of its 8 triples are present
    k222_counts = [0] * num_k222s  # number of triples from each k222 in current set
    
    def is_valid_add(triple_idx, current_set):
        """Check if adding triple_idx creates a K(2,2,2) violation"""
        for k222_idx in triple_to_k222s.get(triple_idx, []):
            if k222_counts[k222_idx] == 7:  # all other 7 are present
                return False
        return True
    
    def add_triple(triple_idx):
        current_set[triple_idx] = True
        for k222_idx in triple_to_k222s.get(triple_idx, []):
            k222_counts[k222_idx] += 1
    
    def remove_triple(triple_idx):
        current_set[triple_idx] = False
        for k222_idx in triple_to_k222s.get(triple_idx, []):
            k222_counts[k222_idx] -= 1
    
    # Greedy initialization
    current_set = [False] * num_triples
    triple_order = list(range(num_triples))
    random.shuffle(triple_order)
    
    for idx in triple_order:
        if is_valid_add(idx, current_set):
            add_triple(idx)
    
    best_set = current_set[:]
    best_count = sum(current_set)
    
    # Local search with simulated annealing
    # Moves: add one triple, remove one triple, or swap (remove one, add another)
    
    current_count = best_count
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.01
    
    rng = random.Random(42)
    
    while time.time() < deadline and temperature > min_temperature:
        # Try a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove a random triple
            if current_count == 0:
                temperature *= cooling_rate
                continue
            idx = rng.randrange(num_triples)
            if not current_set[idx]:
                continue
            # Remove it
            remove_triple(idx)
            new_count = current_count - 1
            delta_energy = -1  # count decreased by 1
            # Check if this removes any K(2,2,2) violations
            # Actually, removing can only help
            accept = True
            if accept:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_set = current_set[:]
            else:
                add_triple(idx)  # revert
        elif move_type < 0.7:
            # Add a random triple
            idx = rng.randrange(num_triples)
            if current_set[idx]:
                continue
            if not is_valid_add(idx, current_set):
                # Would create a violation, so don't add
                # But in SA, we might still try with penalty
                # For now, skip invalid adds
                continue
            add_triple(idx)
            new_count = current_count + 1
            accept = True
            if accept:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_set = current_set[:]
            else:
                remove_triple(idx)  # revert
        else:
            # Swap: remove one, add another
            if current_count == 0:
                temperature *= cooling_rate
                continue
            idx_remove = rng.randrange(num_triples)
            if not current_set[idx_remove]:
                continue
            idx_add = rng.randrange(num_triples)
            if current_set[idx_add]:
                continue
            # Check if removing idx_remove and adding idx_add is valid
            # First, tentatively remove idx_remove
            remove_triple(idx_remove)
            # Now check if we can add idx_add
            if is_valid_add(idx_add, current_set):
                add_triple(idx_add)
                # Count stays the same
                # This is a neutral move, always accept
                if current_count > best_count:
                    best_count = current_count
                    best_set = current_set[:]
            else:
                # Revert
                add_triple(idx_remove)
        
        temperature *= cooling_rate
    
    # Convert best_set back to list of triples
    result = []
    for i, included in enumerate(best_set):
        if included:
            result.append(list(all_triples[i]))
    
    return result

# EVOLVE-BLOCK-END
