import random
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in all_triples ]
    
    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Sort masks by popcount descending
    all_masks = list(range(1 << 16))
    all_masks.sort(key=lambda x: popcount[x], reverse=True)
    
    def get_alpha(edge_masks):
        if not edge_masks:
            return 16
        for m in all_masks:
            is_indep = True
            for e in edge_masks:
                if (m & e) == e:
                    is_indep = False
                    break
            if is_indep:
                return popcount[m]
        return 0

    def is_linear(new_mask, edge_masks):
        for e in edge_masks:
            inter = new_mask & e
            if inter and (inter & (inter - 1)):
                return False
        return True

    # Greedy initialization
    def greedy_init():
        edges = []
        shuffled_indices = list(range(len(all_triples)))
        random.shuffle(shuffled_indices)
        for idx in shuffled_indices:
            m = triple_masks[idx]
            if is_linear(m, edges):
                edges.append(m)
        return edges

    best_edges = greedy_init()
    best_alpha = get_alpha(best_edges)
    
    # Local Search
    current_edges = best_edges[:]
    current_alpha = best_alpha
    
    T = 1.0
    T_min = 0.01
    
    while time.time() < deadline:
        # Generate neighbor
        r = random.random()
        new_edges = current_edges[:]
        
        if r < 0.7:
            # Swap
            if current_edges:
                # Remove random
                idx_rm = random.randint(0, len(current_edges)-1)
                removed = current_edges[idx_rm]
                new_edges.pop(idx_rm)
                
                # Add random valid
                added = False
                for _ in range(50): # Try 50 times to find a valid one
                    cand_idx = random.randint(0, len(all_triples)-1)
                    cand_mask = triple_masks[cand_idx]
                    if cand_mask != removed and is_linear(cand_mask, new_edges):
                        new_edges.append(cand_mask)
                        added = True
                        break
                if not added:
                    # Revert if couldn't add
                    new_edges.insert(idx_rm, removed)
        elif r < 0.85:
            # Add
            added = False
            for _ in range(20):
                cand_idx = random.randint(0, len(all_triples)-1)
                cand_mask = triple_masks[cand_idx]
                if is_linear(cand_mask, new_edges):
                    new_edges.append(cand_mask)
                    added = True
                    break
            if not added:
                continue # Skip this iteration if no valid add found
        else:
            # Remove
            if current_edges:
                idx_rm = random.randint(0, len(current_edges)-1)
                new_edges.pop(idx_rm)
            else:
                continue

        new_alpha = get_alpha(new_edges)
        
        if new_alpha < current_alpha:
            current_edges = new_edges
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = current_edges[:]
                # If we hit 5, it's likely optimal, but keep searching briefly or stop?
                # Given the lower bound is 5, 5 is the best possible.
                if best_alpha == 5:
                    # We can break early to save time, or continue to ensure robustness?
                    # The task says "Minimize". 5 is the min.
                    # Breaking is safe.
                    break
        else:
            # Simulated Annealing accept
            delta = new_alpha - current_alpha
            if delta > 0:
                accept_prob = 1.0 / (1.0 + delta) # Simple decay
                if random.random() < accept_prob * T:
                    current_edges = new_edges
                    current_alpha = new_alpha
            # Decay T
            T *= 0.999
            
    # Convert back to list of triples
    best_triples = []
    for m in best_edges:
        # Find triple corresponding to mask
        # We can precompute a map mask -> triple
        # Or just iterate
        pass
        
    # Precompute mask to triple map for output
    mask_to_triple = {}
    for t, m in zip(all_triples, triple_masks):
        mask_to_triple[m] = list(t)
        
    for m in best_edges:
        best_triples.append(mask_to_triple[m])
        
    return best_triples
