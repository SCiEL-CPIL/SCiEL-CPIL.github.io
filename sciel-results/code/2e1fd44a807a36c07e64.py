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
    
    # Precompute neighbors as lists of indices
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # For radius=1, each word has 1 + n*(q-1) = 1 + 6*2 = 13 neighbors
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
    
    # Convert to numpy arrays for fast operations
    # neighbors_np[i] is a sorted array of neighbor indices
    neighbors_np = np.array([np.array(nb) for nb in neighbors], dtype=np.int32)
    
    def greedy_start():
        code_set = set()
        cover_count = np.zeros(total_words, dtype=np.int32)
        uncovered = np.ones(total_words, dtype=bool)
        
        while uncovered.any():
            # Vectorized gain computation
            # gain[i] = number of uncovered neighbors of word i
            # neighbors_np is (729, 13) array
            # For each i, gain = sum(uncovered[neighbors_np[i]])
            # We can do this with fancy indexing
            all_nb = neighbors_np.ravel()  # (729*13,)
            gains = np.bincount(all_nb, weights=uncovered[all_nb].astype(np.int32), minlength=total_words)
            # Wait, that's not right. Let me think again.
            # neighbors_np[i] gives the 13 neighbors of word i.
            # gain[i] = sum of uncovered[j] for j in neighbors_np[i]
            # We can use: for each neighbor position p (0..12), add uncovered[neighbors_np[:,p]] to gains
            gains = np.zeros(total_words, dtype=np.int32)
            for p in range(neighbors_np.shape[1]):
                gains += uncovered[neighbors_np[:, p]].astype(np.int32)
            
            # Mask out words already in code
            for c in code_set:
                gains[c] = 0
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            code_set.add(best_idx)
            # Update cover_count and uncovered
            nb = neighbors_np[best_idx]
            cover_count[nb] += 1
            uncovered[nb] = False
        
        return code_set, cover_count

    def reduce_code(code_set, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                nb = neighbors_np[c]
                if (cover_count[nb] > 1).all():
                    code_set.remove(c)
                    cover_count[nb] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count):
        """Add codewords until valid cover."""
        uncovered = (cover_count == 0)
        while uncovered.any():
            gains = np.zeros(total_words, dtype=np.int32)
            for p in range(neighbors_np.shape[1]):
                gains += uncovered[neighbors_np[:, p]].astype(np.int32)
            
            for c in code_set:
                gains[c] = 0
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            code_set.add(best_idx)
            nb = neighbors_np[best_idx]
            cover_count[nb] += 1
            uncovered = (cover_count == 0)
        
        return code_set, cover_count

    # Main ILS Loop
    rng = random.Random(42)
    
    # Initial solution
    code_set, cover_count = greedy_start()
    code_set, cover_count = reduce_code(code_set, cover_count)
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    # ILS
    iter_count = 0
    while time.time() < deadline - 2:
        iter_count += 1
        # Perturb: remove k random codewords
        k = rng.randint(2, 6)
        current_code = list(code_set)
        if len(current_code) < k:
            k = max(1, len(current_code) // 2)
        
        to_remove = rng.sample(current_code, k)
        
        for c in to_remove:
            code_set.remove(c)
            cover_count[neighbors_np[c]] -= 1
        
        # Repair
        code_set, cover_count = repair(code_set, cover_count)
        
        # Reduce
        code_set, cover_count = reduce_code(code_set, cover_count)
        
        # 2-opt local search: try swapping one codeword for another
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_list = list(code_set)
            for i, c in enumerate(current_list):
                # Try removing c
                cover_count[neighbors_np[c]] -= 1
                uncovered = (cover_count == 0)
                if not uncovered.any():
                    # c is redundant, remove it
                    code_set.discard(c)
                    improved = True
                    break
                # Try adding each other word to see if we can compensate
                for j, d in enumerate(current_list):
                    if i == j:
                        continue
                    nb_d = neighbors_np[d]
                    # Check if adding d covers all uncovered
                    if (uncovered[nb_d]).all():
                        # Swap: remove c, add d
                        code_set.discard(c)
                        cover_count[neighbors_np[c]] += 1  # restore
                        code_set.add(d)
                        cover_count[nb_d] += 1
                        improved = True
                        break
                if improved:
                    break
                # Restore c
                cover_count[neighbors_np[c]] += 1
            
            if improved:
                code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
            if best_size <= 73:
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
