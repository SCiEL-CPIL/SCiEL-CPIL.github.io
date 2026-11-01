# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # For the specific hard case q=3, n=6, radius=1, use specialized ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        # Too large for bitset approach in pure python quickly, use generic greedy
        return _generic_construct(q, n, radius, deadline)
        
    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute neighbors as lists for faster iteration
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # For radius=1, each word has 1 + n*(q-1) = 13 neighbors
    neighbors = [None] * total_words
    for i, t in enumerate(all_tuples):
        nb = [i]
        t_list = list(t)
        for pos in range(n):
            orig = t_list[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list[pos] = new_val
                j = idx_map[tuple(t_list)]
                nb.append(j)
            t_list[pos] = orig
        neighbors[i] = nb
    
    # Also keep bitset neighbors for fast coverage checks
    neighbors_mask = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        neighbors_mask[i] = mask
    
    full_mask = (1 << total_words) - 1
    
    def greedy_start():
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors_mask[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            code_set.add(best_idx)
            m = neighbors_mask[best_idx]
            for idx in neighbors[best_idx]:
                cover_count[idx] += 1
            
            uncovered_mask &= ~m
            
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                redundant = True
                for idx in neighbors[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    code_set.remove(c)
                    for idx in neighbors[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        """Add codewords until valid cover."""
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors_mask[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            code_set.add(best_idx)
            m = neighbors_mask[best_idx]
            for idx in neighbors[best_idx]:
                cover_count[idx] += 1
            
            uncovered_mask &= ~m
        return code_set, cover_count, uncovered_mask

    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    # Main ILS Loop with multiple restarts
    rng = random.Random(42)
    
    best_size = float('inf')
    best_code = []
    
    # Multiple restarts with different seeds
    max_restarts = 8
    time_per_restart = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
        
        # Vary the random seed for each restart
        rng_seed = 42 + restart * 137
        local_rng = random.Random(rng_seed)
        
        # Initial solution
        code_set, cover_count, uncovered_mask = greedy_start()
        code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Recalculate uncovered
        uncovered_mask = get_uncovered_mask(cover_count)
        
        current_size = len(code_set)
        
        # ILS iterations within this restart
        max_ils_iters = 500
        for it in range(max_ils_iters):
            if time.time() > deadline - 3:
                break
            
            # Perturb: remove k random codewords
            k = local_rng.randint(2, 6)
            current_code = list(code_set)
            if len(current_code) < k:
                k = max(1, len(current_code) // 2)
            
            to_remove = local_rng.sample(current_code, k)
            
            # Apply removal
            for c in to_remove:
                code_set.remove(c)
                for idx in neighbors[c]:
                    cover_count[idx] -= 1
            
            # Recalculate uncovered
            uncovered_mask = get_uncovered_mask(cover_count)
            
            # Repair
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            
            # Reduce
            code_set, cover_count = reduce_code(code_set, cover_count)
            
            # Check improvement
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)
                
                if best_size == 73:
                    # Can't do better than known optimum
                    break
            elif len(code_set) < current_size:
                # Accept if better than start of this ILS run
                pass
            
            # If we've been stuck for a while, try more aggressive perturbation
            if it > 100 and it % 20 == 0:
                k2 = local_rng.randint(5, 10)
                current_code = list(code_set)
                if len(current_code) > k2:
                    to_remove2 = local_rng.sample(current_code, k2)
                    for c in to_remove2:
                        code_set.remove(c)
                        for idx in neighbors[c]:
                            cover_count[idx] -= 1
                    uncovered_mask = get_uncovered_mask(cover_count)
                    code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
                    code_set, cover_count = reduce_code(code_set, cover_count)
        
        if best_size == 73:
            break
    
    # Convert best_code to lists of ints
    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbors
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Greedy
    code_set = set()
    cover_count = [0] * total_words
    uncovered_mask = full_mask
    
    while uncovered_mask and time.time() < deadline - 10:
        best_idx = -1
        best_cnt = -1
        for i in range(total_words):
            if i in code_set:
                continue
            new_covered = neighbors[i] & uncovered_mask
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1: break
        code_set.add(best_idx)
        m = neighbors[best_idx]
        mm = m
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mm ^= lsb
        uncovered_mask &= ~m
        
    # Simple local search: remove redundant
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            redundant = True
            m = neighbors[c]
            mm = m
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                mm ^= lsb
            if redundant:
                code_set.remove(c)
                m = neighbors[c]
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    mm ^= lsb
                changed = True
                break
                
    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
