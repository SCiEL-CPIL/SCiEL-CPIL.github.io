"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
import numpy as np

def construct():
    n = 16
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    
    # Precompute for each triple, which K(2,2,2) configurations it participates in
    # A K(2,2,2) is defined by 3 disjoint pairs: {a1,a2}, {b1,b2}, {c1,c2}
    # The 8 transversal triples are all (xi, yi, zi)
    # For a given triple (a,b,c), it belongs to K(2,2,2) configs where
    # a is in one pair, b in another, c in the third (or any permutation)
    
    # Better approach: for each pair of points, track which triples contain them
    # A K(2,2,2) violation occurs when for 3 disjoint pairs P1,P2,P3,
    # all 8 combinations are present
    
    # Key insight: for 3 disjoint pairs, there are C(16,2)*C(14,2)*C(12,2)/6 possible 
    # ways to choose 3 disjoint pairs, but we can enumerate more efficiently
    
    # Let's precompute: for each unordered pair of pairs, which points are left
    # Actually, let's use a different approach:
    # For each triple (a,b,c), the K(2,2,2) configs it's in are determined by
    # choosing a partner for a (not b,c), partner for b (not a,c), partner for c (not a,b)
    # such that all three partners are distinct and different from a,b,c
    
    # Precompute: for each triple index, list of (partner_a, partner_b, partner_c) 
    # where {a,partner_a}, {b,partner_b}, {c,partner_c} are disjoint pairs
    # Then a K(2,2,2) is "complete" when all 8 corresponding triples are present
    
    # More efficient: enumerate all sets of 3 disjoint pairs
    # Number of ways: C(16,2)*C(14,2)*C(12,2)/6 = 120*91*66/6 = 117,930
    # For each such set, the 8 transversal triples are determined
    
    # Precompute all K(2,2,2) configurations and which triples they contain
    k222_configs = []
    triple_to_configs = [[] for _ in range(num_triples)]
    
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all 3-disjoint-pair configurations
    # Choose 6 distinct points, partition into 3 pairs
    # Number: C(16,6) * 15 = 8008 * 15 = 120,120 (each set of 6 has 15 pairings)
    for six_points in itertools.combinations(range(n), 6):
        a, b, c, d, e, f = six_points
        # 15 ways to pair 6 points into 3 pairs
        pairings = [
            (a,b),(c,d),(e,f),
            (a,b),(c,e),(d,f),
            (a,b),(c,f),(d,e),
            (a,c),(b,d),(e,f),
            (a,c),(b,e),(d,f),
            (a,c),(b,f),(d,e),
            (a,d),(b,c),(e,f),
            (a,d),(b,e),(c,f),
            (a,d),(b,f),(c,e),
            (a,e),(b,c),(d,f),
            (a,e),(b,d),(c,f),
            (a,e),(b,f),(c,d),
            (a,f),(b,c),(d,e),
            (a,f),(b,d),(c,e),
            (a,f),(b,e),(c,d),
        ]
        for p1, p2, p3 in pairings:
            # The 8 transversal triples
            x1, x2 = p1
            y1, y2 = p2
            z1, z2 = p3
            config_triples = set()
            for x in (x1, x2):
                for y in (y1, y2):
                    for z in (z1, z2):
                        t = tuple(sorted((x, y, z)))
                        config_triples.add(t)
            
            config_idx = len(k222_configs)
            k222_configs.append(config_triples)
            for t in config_triples:
                if t in triple_index:
                    triple_to_configs[triple_index[t]].append(config_idx)
    
    num_configs = len(k222_configs)
    
    # For each config, track how many of its 8 triples are present
    config_count = np.zeros(num_configs, dtype=np.int32)
    
    def add_triple(t_idx):
        """Add triple t_idx, update config counts. Returns False if would create violation."""
        if config_count[triple_to_configs[t_idx]] >= 8:
            pass  # already complete, adding more doesn't hurt (but we check below)
        for cfg_idx in triple_to_configs[t_idx]:
            if config_count[cfg_idx] == 7:
                # Adding this would make it 8, creating a violation
                return False
            config_count[cfg_idx] += 1
        return True
    
    def remove_triple(t_idx):
        """Remove triple t_idx, update config counts."""
        for cfg_idx in triple_to_configs[t_idx]:
            config_count[cfg_idx] -= 1
    
    def try_add(t_idx, present_set):
        """Try to add triple, return True if successful."""
        if t_idx in present_set:
            return False
        # Check if any config would become complete
        for cfg_idx in triple_to_configs[t_idx]:
            if config_count[cfg_idx] == 7:
                return False
        # Add it
        present_set.add(t_idx)
        for cfg_idx in triple_to_configs[t_idx]:
            config_count[cfg_idx] += 1
        return True
    
    def try_remove(t_idx, present_set):
        """Remove triple, return True if was present."""
        if t_idx not in present_set:
            return False
        present_set.discard(t_idx)
        for cfg_idx in triple_to_configs[t_idx]:
            config_count[cfg_idx] -= 1
        return True
    
    best_size = 0
    best_set = set()
    
    # WalkSAT-style local search with restarts
    num_restarts = 50
    max_steps = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random subset
        # Start with a random fraction of triples
        init_fraction = 0.3 + 0.4 * random.random()
        present = set()
        random.shuffle(all_triples)
        for t in all_triples:
            if random.random() < init_fraction:
                if try_add(triple_index[t], present):
                    pass  # already added
        
        current_size = len(present)
        
        # Local search: WalkSAT style
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_size > best_size:
                best_size = current_size
                best_set = set(present)
            
            # Try to add a random absent triple
            if random.random() < 0.7:
                # Try addition
                candidates = [t for t in all_triples if t not in present]
                if not candidates:
                    break
                t = random.choice(candidates)
                t_idx = triple_index[t]
                if try_add(t_idx, present):
                    current_size += 1
                else:
                    # Failed to add, try a different move
                    if random.random() < 0.3 and present:
                        # Remove a random present triple to enable future additions
                        t_remove = random.choice(list(present))
                        try_remove(triple_index[t_remove], present)
                        current_size -= 1
            else:
                # Try a swap: remove one, add another
                if not present:
                    continue
                t_remove = random.choice(list(present))
                candidates = [t for t in all_triples if t not in present]
                if not candidates:
                    break
                t_add = random.choice(candidates)
                # Try the swap
                t_remove_idx = triple_index[t_remove]
                t_add_idx = triple_index[t_add]
                
                # Check if removing t_remove and adding t_add is valid
                # First check if t_add can be added (ignoring t_remove for now)
                # We need to check: after removing t_remove, can we add t_add?
                # This is complex, so let's just try:
                if try_remove(t_remove_idx, present):
                    if try_add(t_add_idx, present):
                        # Net size change: -1 + 1 = 0, but we might have freed up configs
                        pass
                    else:
                        # Put it back
                        try_add(t_remove_idx, present)
        
        # After local search, try greedy completion
        if time.time() > deadline:
            break
        for t in all_triples:
            if t not in present:
                if try_add(triple_index[t], present):
                    current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(present)
    
    # Convert back to list of triples
    result = [list(all_triples[i]) for i in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
