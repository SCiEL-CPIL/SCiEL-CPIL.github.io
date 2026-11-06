"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

V = 16
N_SUBSETS = 1 << V

# Precompute subset masks and sizes
SUBSET_MASKS = np.arange(N_SUBSETS, dtype=np.int32)
SUBSET_SIZES_ARR = np.array([bin(i).count('1') for i in range(N_SUBSETS)], dtype=np.int8)

# Precompute all triples and their masks
ALL_TRIPLES = []
TRIPLE_MASKS = []
for t in itertools.combinations(range(V), 3):
    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    ALL_TRIPLES.append(t)
    TRIPLE_MASKS.append(mask)

# Precompute pairs for each triple
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
    norm_pairs = tuple(sorted(p) for p in pairs)
    TRIPLE_PAIRS.append(norm_pairs)

def compute_alpha(edge_masks):
    """Compute independence number using NumPy vectorization."""
    if not edge_masks:
        return V
    independent = np.ones(N_SUBSETS, dtype=bool)
    for e in edge_masks:
        independent &= ((SUBSET_MASKS & e) != e)
    if not np.any(independent):
        return 0
    return int(np.max(SUBSET_SIZES_ARR[independent]))

def compute_alpha_and_witness(edge_masks):
    """Compute alpha and return the witness set (largest independent set)."""
    if not edge_masks:
        return V, (1 << V) - 1
    independent = np.ones(N_SUBSETS, dtype=bool)
    for e in edge_masks:
        independent &= ((SUBSET_MASKS & e) != e)
    if not np.any(independent):
        return 0, 0
    max_size = int(np.max(SUBSET_SIZES_ARR[independent]))
    # Find one witness with max size
    candidates = np.where(independent & (SUBSET_SIZES_ARR == max_size))[0]
    if len(candidates) > 0:
        witness = int(candidates[0])
    else:
        witness = 0
    return max_size, witness

def greedy_construct(seed=42):
    """Greedy construction of a linear triple system."""
    rng = random.Random(seed)
    used_pairs = set()
    edges = []
    edge_masks = []
    
    # Shuffle triples
    indices = list(range(len(ALL_TRIPLES)))
    rng.shuffle(indices)
    
    for idx in indices:
        t = ALL_TRIPLES[idx]
        m = TRIPLE_MASKS[idx]
        pairs = TRIPLE_PAIRS[idx]
        
        valid = True
        for p in pairs:
            if p in used_pairs:
                valid = False
                break
        if valid:
            edges.append(t)
            edge_masks.append(m)
            for p in pairs:
                used_pairs.add(p)
    
    return edges, edge_masks, used_pairs

def repair_step(edges, edge_masks, used_pairs, rng):
    """
    Try to reduce alpha by adding triples that hit the current largest independent set.
    Returns new (edges, edge_masks, used_pairs, improved)
    """
    alpha, witness = compute_alpha_and_witness(edge_masks)
    
    if alpha <= 3:
        return edges, edge_masks, used_pairs, False
    
    # Get the points in the witness set
    witness_points = [i for i in range(V) if (witness >> i) & 1]
    
    # Find triples that are subsets of the witness and not already used
    # and don't conflict with existing pairs
    candidate_triples = []
    for idx, t in enumerate(ALL_TRIPLES):
        m = TRIPLE_MASKS[idx]
        # Check if triple is subset of witness
        if (m & witness) == m:
            pairs = TRIPLE_PAIRS[idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                candidate_triples.append(idx)
    
    if not candidate_triples:
        # No direct fix; try a swap: remove a random edge and try to add two
        if len(edges) > 5:
            rem_idx = rng.randrange(len(edges))
            rem_t = edges[rem_idx]
            rem_m = edge_masks[rem_idx]
            rem_pairs = TRIPLE_PAIRS[ALL_TRIPLES.index(rem_t)]
            
            # Remove it
            new_edges = edges[:]
            new_masks = edge_masks[:]
            new_used = used_pairs.copy()
            
            new_edges.pop(rem_idx)
            new_masks.pop(rem_idx)
            for p in rem_pairs:
                new_used.discard(p)
            
            # Try to add two new triples
            added = 0
            for idx in rng.sample(range(len(ALL_TRIPLES)), min(50, len(ALL_TRIPLES))):
                if added >= 2:
                    break
                t = ALL_TRIPLES[idx]
                m = TRIPLE_MASKS[idx]
                pairs = TRIPLE_PAIRS[idx]
                valid = True
                for p in pairs:
                    if p in new_used:
                        valid = False
                        break
                if valid:
                    new_edges.append(t)
                    new_masks.append(m)
                    for p in pairs:
                        new_used.add(p)
                    added += 1
            
            new_alpha, _ = compute_alpha(new_masks)
            if new_alpha < alpha:
                return new_edges, new_masks, new_used, True
        
        return edges, edge_masks, used_pairs, False
    
    # Add the best candidate (randomly from candidates)
    chosen = rng.choice(candidate_triples)
    t = ALL_TRIPLES[chosen]
    m = TRIPLE_MASKS[chosen]
    pairs = TRIPLE_PAIRS[chosen]
    
    new_edges = edges + [t]
    new_masks = edge_masks + [m]
    new_used = used_pairs.copy()
    for p in pairs:
        new_used.add(p)
    
    new_alpha, _ = compute_alpha(new_masks)
    if new_alpha < alpha:
        return new_edges, new_masks, new_used, True
    
    return edges, edge_masks, used_pairs, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_alpha = V
    best_edges = None
    best_masks = None
    best_used = None
    
    # Try multiple seeds for greedy construction
    for seed in range(100):
        if time.time() > deadline:
            break
        
        edges, edge_masks, used_pairs = greedy_construct(seed)
        alpha, _ = compute_alpha(edge_masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
            best_masks = edge_masks[:]
            best_used = used_pairs.copy()
        
        if best_alpha <= 3:
            break
    
    # If we found something good, try to improve further with repair
    if best_edges is not None and best_alpha > 3:
        rng = random.Random(12345)
        edges = best_edges[:]
        edge_masks = best_masks[:]
        used_pairs = best_used.copy()
        current_alpha = best_alpha
        no_improve_count = 0
        
        while time.time() < deadline and no_improve_count < 200:
            new_edges, new_masks, new_used, improved = repair_step(
                edges, edge_masks, used_pairs, rng
            )
            
            if improved:
                edges = new_edges
                edge_masks = new_masks
                used_pairs = new_used
                current_alpha, _ = compute_alpha(edge_masks)
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_edges = edges[:]
                    best_masks = edge_masks[:]
                    best_used = used_pairs.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if best_alpha <= 3:
                break
    
    # Final fallback: if still not good, try a different strategy
    # Use a more aggressive greedy with priority on covering large subsets
    if best_alpha > 4 and time.time() < deadline:
        for seed in range(100):
            if time.time() > deadline:
                break
            
            # Priority greedy: always pick triple that hits the most uncovered pairs
            rng = random.Random(seed + 10000)
            used_pairs = set()
            edges = []
            edge_masks = []
            
            available = list(range(len(ALL_TRIPLES)))
            rng.shuffle(available)
            
            for idx in available:
                t = ALL_TRIPLES[idx]
                m = TRIPLE_MASKS[idx]
                pairs = TRIPLE_PAIRS[idx]
                
                valid = True
                for p in pairs:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    edges.append(t)
                    edge_masks.append(m)
                    for p in pairs:
                        used_pairs.add(p)
            
            alpha, _ = compute_alpha(edge_masks)
            if alpha < best_alpha:
                best_alpha = alpha
                best_edges = edges[:]
                best_masks = edge_masks[:]
                best_used = used_pairs.copy()
            
            if best_alpha <= 3:
                break
    
    # Return the best result
    if best_edges is None:
        # Absolute fallback: just 5 disjoint triples
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_edges
# EVOLVE-BLOCK-END
