# EVOLVE-BLOCK-START
import time
import random
import itertools

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
    
    # For the specific hard case q=3, n=6, radius=1, use specialized group-structured ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)
        
    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute neighbors as bitsets
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor lists and bitset masks
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        mask |= (1 << i)
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Precompute perturbation vectors in Z_3^6 for structured swaps
    # Small perturbations: single-coordinate changes and 2-coordinate changes
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):  # +1 and -1 mod 3
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)
    
    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    def greedy_start(seed=42):
        rng = random.Random(seed)
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            # Use random tie-breaking for diversity
            candidates = []
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    candidates = [i]
                elif cnt == best_cnt:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = rng.choice(candidates)
            code_set.add(best_idx)
            m = neighbors[best_idx]
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~m
            
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count, max_passes=10):
        changed = True
        passes = 0
        while changed and passes < max_passes and time.time() < deadline - 3:
            changed = False
            passes += 1
            for c in list(code_set):
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        while uncovered_mask:
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
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    def check_valid(cover_count):
        """Check if cover_count is valid (no zeros)"""
        for i in range(total_words):
            if cover_count[i] == 0:
                return False
        return True

    def try_1exchange(code_set, cover_count, c, new_idx):
        """Try replacing c with new_idx. Returns (success, new_cover_count)"""
        if new_idx in code_set:
            return False, cover_count
        
        # Save old cover_count state for this codeword's neighbors
        old_counts = {}
        for idx in neighbor_indices[c]:
            old_counts[idx] = cover_count[idx]
        
        # Remove c
        for idx in neighbor_indices[c]:
            cover_count[idx] -= 1
        code_set.remove(c)
        
        # Add new_idx
        code_set.add(new_idx)
        for idx in neighbor_indices[new_idx]:
            cover_count[idx] += 1
        
        # Check validity
        valid = True
        for i in range(total_words):
            if cover_count[i] == 0:
                valid = False
                break
        
        if valid:
            return True, cover_count
        else:
            # Revert
            code_set.remove(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] -= 1
            code_set.add(c)
            for idx, cnt in old_counts.items():
                cover_count[idx] = cnt
            return False, cover_count

    def local_search_1exchange(code_set, cover_count, rng, max_iters=100):
        """
        Try 1-exchange moves: replace codeword c with c+d for small d.
        Accept if the code remains valid and size doesn't increase.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx in code_set:
                continue
            
            success, cover_count = try_1exchange(code_set, cover_count, c, new_idx)
            if success:
                improved = True
        
        return code_set, cover_count, improved

    def local_search_2exchange(code_set, cover_count, rng, max_iters=50):
        """
        Try 2-exchange moves: replace two codewords c1, c2 with two new codewords.
        This can escape local optima that 1-exchange cannot.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(code_set) < 2:
                break
                
            # Pick two codewords to remove
            c1, c2 = rng.sample(list(code_set), 2)
            
            # Generate candidate replacements
            c1_tuple = all_tuples[c1]
            c2_tuple = all_tuples[c2]
            
            # Try a few random perturbations for each
            found = False
            for attempt in range(5):
                d1 = rng.choice(perturbations)
                d2 = rng.choice(perturbations)
                new1_tuple = tuple((c1_tuple[i] + d1[i]) % q for i in range(n))
                new2_tuple = tuple((c2_tuple[i] + d2[i]) % q for i in range(n))
                new1_idx = idx_map[new1_tuple]
                new2_idx = idx_map[new2_tuple]
                
                # Check no conflicts
                if new1_idx in code_set or new2_idx in code_set:
                    continue
                if new1_idx == new2_idx:
                    continue
                
                # Save state
                old_counts = {}
                for idx in neighbor_indices[c1]:
                    old_counts[idx] = cover_count[idx]
                for idx in neighbor_indices[c2]:
                    if idx not in old_counts:
                        old_counts[idx] = cover_count[idx]
                
                # Remove c1 and c2
                for idx in neighbor_indices[c1]:
                    cover_count[idx] -= 1
                for idx in neighbor_indices[c2]:
                    cover_count[idx] -= 1
                code_set.remove(c1)
                code_set.remove(c2)
                
                # Add new1 and new2
                code_set.add(new1_idx)
                code_set.add(new2_idx)
                for idx in neighbor_indices[new1_idx]:
                    cover_count[idx] += 1
                for idx in neighbor_indices[new2_idx]:
                    cover_count[idx] += 1
                
                # Check validity
                valid = True
                for i in range(total_words):
                    if cover_count[i] == 0:
                        valid = False
                        break
                
                if valid:
                    improved = True
                    found = True
                    break
                else:
                    # Revert
                    code_set.remove(new1_idx)
                    code_set.remove(new2_idx)
                    for idx in neighbor_indices[new1_idx]:
                        cover_count[idx] -= 1
                    for idx in neighbor_indices[new2_idx]:
                        cover_count[idx] -= 1
                    code_set.add(c1)
                    code_set.add(c2)
                    for idx, cnt in old_counts.items():
                        cover_count[idx] = cnt
            
            if found:
                continue
        
        return code_set, cover_count, improved

    def full_local_search(code_set, cover_count, rng):
        """Run both 1-exchange and 2-exchange local search until no improvement."""
        any_improved = True
        while any_improved and time.time() < deadline - 3:
            any_improved = False
            code_set, cover_count, imp1 = local_search_1exchange(code_set, cover_count, rng, max_iters=80)
            if imp1:
                any_improved = True
            code_set, cover_count, imp2 = local_search_2exchange(code_set, cover_count, rng, max_iters=30)
            if imp2:
                any_improved = True
            code_set, cover_count = reduce_code(code_set, cover_count)
        return code_set, cover_count

    # Main ILS Loop
    rng = random.Random(42)
    
    # Multi-start: try several different initial solutions
    best_size = float('inf')
    best_code = None
    
    num_starts = 3
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
        seed = 42 + s * 1000
        code_set, cover_count, uncovered_mask = greedy_start(seed)
        code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Local search
        code_set, cover_count = full_local_search(code_set, cover_count, rng)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
        
        if best_size <= 73:
            break
    
    # ILS with structured perturbations
    iteration = 0
    while time.time() < deadline - 8:
        iteration += 1
        
        # Perturb: structured swap or random removal
        r = rng.random()
        if r < 0.4:
            # Structured 1-exchange perturbation
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx not in code_set:
                # Save state
                old_counts = {}
                for idx in neighbor_indices[c]:
                    old_counts[idx] = cover_count[idx]
                
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
                code_set.remove(c)
                code_set.add(new_idx)
                for idx in neighbor_indices[new_idx]:
                    cover_count[idx] += 1
                
                # Check if valid, if not revert
                valid = all(cover_count[i] > 0 for i in range(total_words))
                if not valid:
                    code_set.remove(new_idx)
                    for idx in neighbor_indices[new_idx]:
                        cover_count[idx] -= 1
                    code_set.add(c)
                    for idx, cnt in old_counts.items():
                        cover_count[idx] = cnt
        elif r < 0.7:
            # Random removal of k codewords
            k = rng.randint(2, 4)
            current_code = list(code_set)
            if len(current_code) < k:
                k = max(1, len(current_code) // 2)
            to_remove = rng.sample(current_code, k)
            for c in to_remove:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
            uncovered_mask = get_uncovered_mask(cover_count)
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)
        else:
            # 2-exchange perturbation: remove 2, add 2 random nearby
            if len(code_set) >= 2:
                c1, c2 = rng.sample(list(code_set), 2)
                c1_tuple = all_tuples[c1]
                c2_tuple = all_tuples[c2]
                d1 = rng.choice(perturbations)
                d2 = rng.choice(perturbations)
                new1_tuple = tuple((c1_tuple[i] + d1[i]) % q for i in range(n))
                new2_tuple = tuple((c2_tuple[i] + d2[i]) % q for i in range(n))
                new1_idx = idx_map[new1_tuple]
                new2_idx = idx_map[new2_tuple]
                
                if new1_idx not in code_set and new2_idx not in code_set and new1_idx != new2_idx:
                    old_counts = {}
                    for idx in neighbor_indices[c1]:
                        old_counts[idx] = cover_count[idx]
                    for idx in neighbor_indices[c2]:
                        if idx not in old_counts:
                            old_counts[idx] = cover_count[idx]
                    
                    for idx in neighbor_indices[c1]:
                        cover_count[idx] -= 1
                    for idx in neighbor_indices[c2]:
                        cover_count[idx] -= 1
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(new1_idx)
                    code_set.add(new2_idx)
                    for idx in neighbor_indices[new1_idx]:
                        cover_count[idx] += 1
                    for idx in neighbor_indices[new2_idx]:
                        cover_count[idx] += 1
                    
                    valid = all(cover_count[i] > 0 for i in range(total_words))
                    if not valid:
                        code_set.remove(new1_idx)
                        code_set.remove(new2_idx)
                        for idx in neighbor_indices[new1_idx]:
                            cover_count[idx] -= 1
                        for idx in neighbor_indices[new2_idx]:
                            cover_count[idx] -= 1
                        code_set.add(c1)
                        code_set.add(c2)
                        for idx, cnt in old_counts.items():
                            cover_count[idx] = cnt
        
        # Local search after perturbation
        code_set, cover_count = full_local_search(code_set, cover_count, rng)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

    # Final polish: dedicated high-precision pass
    if best_code is not None and time.time() < deadline - 5:
        code_set = set(best_code)
        cover_count = [0] * total_words
        for c in code_set:
            for idx in neighbor_indices[c]:
                cover_count[idx] += 1
        
        # Run more aggressive local search
        polish_rng = random.Random(12345)
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            code_set, cover_count, imp = local_search_1exchange(code_set, cover_count, polish_rng, max_iters=150)
            if imp:
                code_set, cover_count = reduce_code(code_set, cover_count)
            code_set, cover_count, imp = local_search_2exchange(code_set, cover_count, polish_rng, max_iters=50)
            if imp:
                code_set, cover_count = reduce_code(code_set, cover_count)
            if not imp:
                break
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
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
        for idx in neighbor_indices[best_idx]:
            cover_count[idx] += 1
        uncovered_mask &= ~neighbors[best_idx]
        
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            redundant = True
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
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
