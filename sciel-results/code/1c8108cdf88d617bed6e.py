import time
import random
import numpy as np
from itertools import combinations

def compute_alpha_complement(triples, n=16):
    """
    Compute independence number using complement/hitting set approach.
    Returns the size of the largest subset containing no triple.
    """
    if not triples:
        return n
    
    triple_set = set(frozenset(t) for t in triples)
    
    # Use iterative deepening: check if there exists an independent set of size k
    # Start from high k and go down
    for k in range(n, 0, -1):
        # Check if there exists a subset of size k with no triple
        found = _exists_independent_set(triples, k, n)
        if found:
            return k
    return 0

def _exists_independent_set(triples, k, n=16, time_limit=0.5):
    """
    Check if there exists an independent set of size k.
    Uses backtracking with pruning.
    """
    start_time = time.time()
    
    # Precompute: for each point, which triples contain it
    point_to_triples = {i: [] for i in range(n)}
    for t in triples:
        for p in t:
            point_to_triples[p].append(frozenset(t))
    
    # Use a greedy + backtracking hybrid
    def backtrack(available, chosen, depth):
        if time.time() - start_time > time_limit:
            return False
        
        if len(chosen) == k:
            return True
        
        # Pruning: not enough available points
        if len(available) < k - len(chosen):
            return False
        
        # Try adding each available point
        for point in list(available):
            # Check if adding this point would create a triple with chosen points
            # A triple is formed if all 3 points of some triple are in chosen ∪ {point}
            new_chosen = chosen | {point}
            valid = True
            
            # Check all triples: is any triple fully contained in new_chosen?
            for t in point_to_triples[point]:
                if t.issubset(new_chosen):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also check: does choosing this point create a triple with other chosen points?
            # Already handled above by checking all triples containing point
            
            # Recurse
            available_new = available - {point}
            if backtrack(available_new, new_chosen, depth + 1):
                return True
        
        return False
    
    initial_available = set(range(n))
    return backtrack(initial_available, frozenset(), 0)

def compute_alpha_fast_v2(triples, n=16):
    """
    Fast independence number calculation using bitset operations.
    This is a hybrid: use numpy for speed, with fallback.
    """
    if not triples:
        return n
    
    # Build adjacency: for each point, which triples contain it
    # Use bitmask approach
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    triple_masks_np = np.array(triple_masks, dtype=np.int32)
    num_triples = len(triple_masks_np)
    
    # Generate all subset masks
    all_masks = np.arange(1 << n, dtype=np.int32)
    
    # Popcount
    if not hasattr(compute_alpha_fast_v2, "popcount_cache"):
        pc = np.zeros(1 << n, dtype=np.int8)
        for i in range(1, 1 << n):
            pc[i] = pc[i >> 1] + (i & 1)
        compute_alpha_fast_v2.popcount_cache = pc
    popcount = compute_alpha_fast_v2.popcount_cache
    
    # For each mask, check if any triple is fully contained
    # (mask & triple_mask) == triple_mask means triple is subset of mask
    # Vectorized: shape (65536, num_triples)
    if num_triples > 50:
        # Too many triples, use different approach
        return _compute_alpha_subset_check(triples, n)
    
    intersects = (all_masks[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    has_triple = intersects.any(axis=1)
    independent = ~has_triple
    
    if not independent.any():
        return 0
    
    max_alpha = popcount[independent].max()
    return int(max_alpha)

def _compute_alpha_subset_check(triples, n=16):
    """Fallback alpha calculation for many triples."""
    triple_set = set(frozenset(t) for t in triples)
    
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            subset_set = frozenset(subset)
            # Check if any triple is fully contained in subset
            contained = False
            for t in triple_set:
                if t.issubset(subset_set):
                    contained = True
                    break
            if not contained:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    
    def get_used_pairs(triples):
        used = set()
        for a, b, c in triples:
            used.add((a, b))
            used.add((a, c))
            used.add((b, c))
        return used
    
    def is_linear(triples):
        used = get_used_pairs(triples)
        return len(used) == 3 * len(triples)
    
    def try_structured_construction(seed):
        """
        Structured construction using group-based approach.
        Divide 16 points into 4 groups of 4, then systematically create triples.
        """
        rng = random.Random(seed)
        
        # Group points: 4 groups of 4
        groups = [list(range(4*i, 4*i+4)) for i in range(4)]
        
        # Shuffle within groups and between groups
        for g in groups:
            rng.shuffle(g)
        rng.shuffle(groups)
        
        triples = []
        used_pairs = set()
        
        def can_add(t):
            a, b, c = t
            return ((a, b) not in used_pairs and 
                    (a, c) not in used_pairs and 
                    (b, c) not in used_pairs)
        
        def add_triple(t):
            a, b, c = t
            triples.append(t)
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))
        
        # Phase 1: Add "cross-group" triples (one point from each of 3 different groups)
        # This ensures good mixing
        group_indices = list(range(4))
        rng.shuffle(group_indices)
        
        for i, j, k in [(group_indices[0], group_indices[1], group_indices[2]),
                         (group_indices[0], group_indices[1], group_indices[3]),
                         (group_indices[0], group_indices[2], group_indices[3]),
                         (group_indices[1], group_indices[2], group_indices[3])]:
            # Try all combinations of one point from each group
            candidates = list(combinations(range(4), 3))
            rng.shuffle(candidates)
            for a_idx, b_idx, c_idx in candidates:
                a = groups[i][a_idx]
                b = groups[j][b_idx]
                c = groups[k][c_idx]
                if can_add((a, b, c)):
                    add_triple((a, b, c))
        
        # Phase 2: Add "within-group" triples if possible
        for g in groups:
            for t in combinations(g, 3):
                if can_add(t):
                    add_triple(t)
        
        return triples
    
    def try_greedy_dense(seed):
        """
        Greedy construction that prioritizes dense packing.
        """
        rng = random.Random(seed)
        candidates = all_triples[:]
        rng.shuffle(candidates)
        
        used_pairs = set()
        triples = []
        
        def can_add(t):
            a, b, c = t
            return ((a, b) not in used_pairs and 
                    (a, c) not in used_pairs and 
                    (b, c) not in used_pairs)
        
        # First pass: add triples that cover many new points
        point_coverage = [0] * n
        for t in candidates:
            if can_add(t):
                a, b, c = t
                score = point_coverage[a] + point_coverage[b] + point_coverage[c]
                # We want to cover uncovered points
                if score < 3:  # At least one new point
                    triples.append(t)
                    used_pairs.add((a, b))
                    used_pairs.add((a, c))
                    used_pairs.add((b, c))
                    point_coverage[a] += 1
                    point_coverage[b] += 1
                    point_coverage[c] += 1
        
        # Second pass: fill in remaining
        for t in candidates:
            if t not in triples and can_add(t):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
        
        return triples
    
    best_triples = []
    best_alpha = n + 1
    
    # Phase 1: Try multiple construction strategies
    num_structured = 500
    num_greedy = 500
    
    for i in range(num_structured):
        if time.time() > deadline - 30:
            break
        triples = try_structured_construction(i)
        if triples:
            alpha = compute_alpha_fast_v2(triples, n)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
    
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        triples = try_greedy_dense(i + 1000)
        if triples:
            alpha = compute_alpha_fast_v2(triples, n)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
    
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Phase 2: Local search with insertion/deletion
    cur_triples = list(best_triples)
    cur_alpha = best_alpha
    used_pairs = get_used_pairs(cur_triples)
    
    def try_improve_insert_delete(max_steps=2000):
        nonlocal cur_triples, cur_alpha, used_pairs
        nonlocal best_triples, best_alpha
        
        steps = 0
        no_improve_count = 0
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            action = random.random()
            
            if action < 0.4:
                # Try to insert a new triple
                t = random.choice(all_triples)
                a, b, c = t
                if ((a, b) not in used_pairs and 
                    (a, c) not in used_pairs and 
                    (b, c) not in used_pairs):
                    # Insert
                    cur_triples.append(t)
                    used_pairs.add((a, b))
                    used_pairs.add((a, c))
                    used_pairs.add((b, c))
                    new_alpha = compute_alpha_fast_v2(cur_triples, n)
                    if new_alpha < cur_alpha:
                        cur_alpha = new_alpha
                        no_improve_count = 0
                        if cur_alpha < best_alpha:
                            best_alpha = cur_alpha
                            best_triples = list(cur_triples)
                    else:
                        # Revert
                        cur_triples.pop()
                        used_pairs.discard((a, b))
                        used_pairs.discard((a, c))
                        used_pairs.discard((b, c))
                        no_improve_count += 1
            else:
                # Try to delete a triple and replace with a better one
                idx = random.randint(0, len(cur_triples) - 1)
                t_old = cur_triples[idx]
                a, b, c = t_old
                old_pairs = [(a, b), (a, c), (b, c)]
                for p in old_pairs:
                    used_pairs.discard(p)
                
                # Try to find a replacement
                improved = False
                for _ in range(50):
                    t_new = random.choice(all_triples)
                    x, y, z = t_new
                    new_pairs = [(x, y), (x, z), (y, z)]
                    
                    if all(p not in used_pairs for p in new_pairs):
                        # Try this replacement
                        cur_triples[idx] = t_new
                        for p in new_pairs:
                            used_pairs.add(p)
                        
                        new_alpha = compute_alpha_fast_v2(cur_triples, n)
                        if new_alpha < cur_alpha:
                            cur_alpha = new_alpha
                            improved = True
                            no_improve_count = 0
                            if cur_alpha < best_alpha:
                                best_alpha = cur_alpha
                                best_triples = list(cur_triples)
                            break
                        else:
                            # Revert
                            cur_triples[idx] = t_old
                            for p in new_pairs:
                                used_pairs.discard(p)
                
                if not improved:
                    # Restore old triple
                    cur_triples[idx] = t_old
                    for p in old_pairs:
                        used_pairs.add(p)
                    no_improve_count += 1
            
            if no_improve_count > 100:
                break
        
        return cur_alpha
    
    # Run local search for a while
    while time.time() < deadline - 10:
        try_improve_insert_delete(1000)
        if best_alpha <= 5:
            break
    
    # Final check
    if best_triples and is_linear(best_triples):
        return [list(t) for t in best_triples]
    
    # Fallback
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
