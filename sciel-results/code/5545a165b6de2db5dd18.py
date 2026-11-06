import time
import itertools
import random
import sys

def _fast_independence_number(triples, n=16):
    """
    Compute the independence number using bitmasks and branch-and-bound.
    A set is independent if it contains no triple entirely.
    """
    if not triples:
        return n
    
    # Build mask for each triple
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    # For each point, precompute which triples contain it
    # Actually, we need: for a candidate set S (as bitmask), check if any triple is subset of S
    
    # Branch and bound: try to find the largest independent set
    # We'll use a recursive approach with pruning
    
    all_points_mask = (1 << n) - 1
    
    def has_independent_set_of_size(target_size, available_mask, current_size=0):
        """Check if there's an independent set of at least target_size within available_mask."""
        if current_size + bin(available_mask).count('0' if False else '1') < target_size:
            # Not enough points left
            if bin(available_mask).count('1') + current_size < target_size:
                return False
        
        if current_size >= target_size:
            return True
        
        # If remaining points can't reach target
        remaining_count = bin(available_mask).count('1')
        if current_size + remaining_count < target_size:
            return False
        
        # Check if available_mask itself is independent (no triple inside)
        # If yes, we found an independent set of size current_size + remaining_count
        is_independent = True
        for tm in triple_masks:
            if (available_mask & tm) == tm:
                is_independent = False
                break
        if is_independent:
            return True  # The whole available set is independent
        
        # Try adding points one by one
        # Pick a point to branch on
        # Heuristic: pick the point that appears in the most triples within available_mask
        best_point = -1
        best_count = -1
        for p in range(n):
            if (available_mask >> p) & 1:
                # Count how many triples contain p and are within available_mask
                cnt = 0
                for tm in triple_masks:
                    if (tm >> p) & 1 and (available_mask & tm) == tm:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_point = p
        
        if best_point == -1:
            return current_size >= target_size
        
        # Branch 1: include best_point
        new_mask = available_mask & ~(1 << best_point)
        if has_independent_set_of_size(target_size, new_mask, current_size + 1):
            return True
        
        # Branch 2: exclude best_point
        if has_independent_set_of_size(target_size, new_mask, current_size):
            return True
        
        return False
    
    # Binary search or linear search for the largest independent set size
    # Try from high to low
    for size in range(n, 2, -1):
        if has_independent_set_of_size(size, all_points_mask, 0):
            return size
    
    # Check size 3
    if has_independent_set_of_size(3, all_points_mask, 0):
        return 3
    
    # Check size 2
    return 2

def _has_independent_set_size_fast(triples_masks, size, n=16):
    """
    Returns True if there exists an independent set of the given size.
    Uses bitset operations for speed.
    """
    all_points_mask = (1 << n) - 1
    
    def check(available_mask, current_size):
        if current_size >= size:
            return True
        
        remaining = bin(available_mask).count('1')
        if current_size + remaining < size:
            return False
        
        # Check if available_mask is independent
        for tm in triples_masks:
            if (available_mask & tm) == tm:
                break
        else:
            # No triple fully contained, so available_mask is independent
            return True
        
        # Branch on a point
        # Pick point with most constraints
        best_p = -1
        best_cnt = -1
        for p in range(n):
            if (available_mask >> p) & 1:
                cnt = 0
                for tm in triples_masks:
                    if (tm >> p) & 1 and (available_mask & tm) == tm:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_p = p
        
        if best_p == -1:
            return current_size >= size
        
        new_mask = available_mask & ~(1 << best_p)
        
        # Include
        if check(new_mask, current_size + 1):
            return True
        # Exclude
        if check(new_mask, current_size):
            return True
        
        return False
    
    return check(all_points_mask, 0)

def _independence_number(triples, n=16):
    """Compute the independence number of a linear triple system."""
    if not triples:
        return n
    
    masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        masks.append(m)
    
    # Try from high to low
    for r in range(n, 2, -1):
        if _has_independent_set_size_fast(masks, r, n):
            return r
    return 2

def _is_linear(triples):
    """Check if the triple system is linear (any two triples share at most one point)."""
    used_pairs = set()
    for t in triples:
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair in used_pairs:
                return False
            used_pairs.add(pair)
    return True

def _greedy_construct(seed, n=16):
    """Greedy construction of a linear triple system."""
    random.seed(seed)
    V = list(range(n))
    all_triples = list(itertools.combinations(V, 3))
    random.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    
    for t in all_triples:
        ok = True
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair in used_pairs:
                ok = False
                break
        if ok:
            triples.append(list(t))
            for a, b in itertools.combinations(t, 2):
                used_pairs.add((min(a, b), max(a, b)))
    
    return triples

def _structured_construct(n=16):
    """
    Structured construction using F2^4-inspired approach.
    Points are 0..15 (4-bit vectors).
    Triples: {a, b, a^b} for all distinct nonzero a, b with a^b != 0.
    This gives a linear triple system since each pair determines a unique third point.
    """
    triples = []
    used_pairs = set()
    
    for a in range(1, n):
        for b in range(a+1, n):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted([a, b, c]))
            # Check linearity
            ok = True
            for x, y in itertools.combinations(t, 2):
                pair = (min(x, y), max(x, y))
                if pair in used_pairs:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
                for x, y in itertools.combinations(t, 2):
                    pair = (min(x, y), max(x, y))
                    used_pairs.add(pair)
    
    return triples

def _local_search_refine(sys_candidate, best_alpha, deadline, n=16):
    """
    Local search to try to reduce the independence number.
    Tries adding triples that hit large independent sets.
    """
    start = time.time()
    current_alpha = _independence_number(sys_candidate, n)
    current_sys = sys_candidate[:]
    
    used_pairs = set()
    for t in current_sys:
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            used_pairs.add(pair)
    
    # Find all valid triples that can be added
    all_triples = list(itertools.combinations(range(n), 3))
    
    improved = True
    while improved and (time.time() - start < deadline - 2):
        improved = False
        
        # Recompute valid additions
        valid_adds = []
        for t in all_triples:
            t_tuple = tuple(t)
            can_add = True
            for a, b in itertools.combinations(t_tuple, 2):
                pair = (min(a, b), max(a, b))
                if pair in used_pairs:
                    can_add = False
                    break
            if can_add:
                valid_adds.append(t_tuple)
        
        if not valid_adds:
            break
        
        # Try random valid additions
        random.shuffle(valid_adds)
        for t in valid_adds[:50]:
            if time.time() - start > deadline - 2:
                break
            
            new_sys = current_sys[:] + [list(t)]
            new_alpha = _independence_number(new_sys, n)
            
            if new_alpha < current_alpha:
                current_sys = new_sys
                current_alpha = new_alpha
                improved = True
                
                # Update used_pairs
                for a, b in itertools.combinations(t, 2):
                    pair = (min(a, b), max(a, b))
                    used_pairs.add(pair)
                
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                
                break
    
    return current_sys, current_alpha, best_alpha

def _backtrack_construct(deadline, n=16, max_depth=25):
    """
    Backtracking construction with forward checking.
    Tries to build a dense linear triple system.
    """
    start = time.time()
    random.seed(int(time.time() * 1000) % 10000)
    
    all_triples = list(itertools.combinations(range(n), 3))
    random.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    
    def backtrack(remaining, depth=0):
        nonlocal triples, used_pairs
        
        if time.time() > deadline:
            return False
        
        if depth >= max_depth:
            return False
        
        if not remaining:
            return True
        
        # Filter to valid triples
        valid = []
        for t in remaining:
            can_add = True
            for a, b in itertools.combinations(t, 2):
                pair = (min(a, b), max(a, b))
                if pair in used_pairs:
                    can_add = False
                    break
            if can_add:
                valid.append(t)
        
        if not valid:
            return len(triples) > 0
        
        # Heuristic: pick triple that covers most "central" pairs
        # For simplicity, pick randomly from first few valid
        idx = random.randint(0, min(len(valid)-1, 10))
        chosen = valid[idx]
        
        # Add
        triples.append(chosen)
        for a, b in itertools.combinations(chosen, 2):
            used_pairs.add((min(a, b), max(a, b)))
        
        remaining_new = [t for t in remaining if t != chosen]
        
        if backtrack(remaining_new, depth + 1):
            return True
        
        # Backtrack
        triples.pop()
        for a, b in itertools.combinations(chosen, 2):
            used_pairs.discard((min(a, b), max(a, b)))
        
        return False
    
    backtrack(all_triples)
    return triples[:]

def construct():
    start = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    n = 16
    
    best_alpha = 16
    best_sys = []
    
    # 1. Try structured construction first (fast, deterministic)
    sys_candidate = _structured_construct(n)
    if sys_candidate and _is_linear(sys_candidate):
        alpha = _independence_number(sys_candidate, n)
        best_alpha = alpha
        best_sys = sys_candidate[:]
    
    # 2. Try greedy with different seeds
    for seed in range(20):
        if time.time() - start > DEADLINE - 10:
            break
        sys_candidate = _greedy_construct(seed, n)
        if sys_candidate and _is_linear(sys_candidate):
            alpha = _independence_number(sys_candidate, n)
            if alpha < best_alpha:
                best_alpha = alpha
                best_sys = sys_candidate[:]
    
    # 3. Local search refinement on best system
    if best_alpha > 5:
        refined_sys, refined_alpha, best_alpha = _local_search_refine(
            best_sys, best_alpha, DEADLINE - 5, n
        )
        if refined_alpha < best_alpha:
            best_alpha = refined_alpha
            best_sys = refined_sys
    
    # 4. Try backtracking construction for denser systems
    if time.time() - start < DEADLINE - 20:
        for attempt in range(5):
            if time.time() - start > DEADLINE - 15:
                break
            sys_candidate = _backtrack_construct(DEADLINE - 10, n, max_depth=30)
            if sys_candidate and _is_linear(sys_candidate):
                alpha = _independence_number(sys_candidate, n)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_sys = sys_candidate[:]
                
                # Refine
                if best_alpha > 5 and time.time() - start < DEADLINE - 10:
                    refined_sys, refined_alpha, best_alpha = _local_search_refine(
                        best_sys, best_alpha, DEADLINE - 5, n
                    )
                    if refined_alpha < best_alpha:
                        best_alpha = refined_alpha
                        best_sys = refined_sys
    
    # 5. Final refinement pass
    if best_alpha > 5 and time.time() - start < DEADLINE - 3:
        refined_sys, refined_alpha, best_alpha = _local_search_refine(
            best_sys, best_alpha, DEADLINE - 1, n
        )
        if refined_alpha < best_alpha:
            best_alpha = refined_alpha
            best_sys = refined_sys
    
    return best_sys
