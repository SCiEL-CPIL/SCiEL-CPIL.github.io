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
    
    def greedy_start():
        code_set = set()
        cover_count = [0] * total_words
        zero_count = total_words
        uncovered_mask = full_mask

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
                if cover_count[idx] == 0:
                    zero_count -= 1
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]

        return code_set, cover_count, zero_count, uncovered_mask

    def reduce_code(code_set, cover_count, zero_count):
        changed = True
        while changed:
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
                        if cover_count[idx] == 1:
                            zero_count += 1
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count, zero_count

    def repair(code_set, cover_count, zero_count, uncovered_mask):
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
                if cover_count[idx] == 0:
                    zero_count -= 1
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, zero_count, uncovered_mask

    def try_add(code_set, cover_count, zero_count, idx):
        code_set.add(idx)
        for j in neighbor_indices[idx]:
            if cover_count[j] == 0:
                zero_count -= 1
            cover_count[j] += 1
        return zero_count

    def try_remove(code_set, cover_count, zero_count, idx):
        code_set.remove(idx)
        for j in neighbor_indices[idx]:
            if cover_count[j] == 1:
                zero_count += 1
            cover_count[j] -= 1
        return zero_count

    def local_search_1exchange(code_set, cover_count, zero_count, rng, max_iters=200):
        """
        Try 1-exchange moves: replace codeword c with c+d for small d.
        Accept if the code remains valid (zero_count == 0).
        """
        improved = False
        code_list = list(code_set)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(code_list)
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]

            if new_idx in code_set:
                continue

            # Try swap: remove c, add new_idx
            zero_count = try_remove(code_set, cover_count, zero_count, c)
            zero_count = try_add(code_set, cover_count, zero_count, new_idx)

            if zero_count == 0:
                improved = True
                code_list[code_list.index(c)] = new_idx
            else:
                # Revert
                zero_count = try_remove(code_set, cover_count, zero_count, new_idx)
                zero_count = try_add(code_set, cover_count, zero_count, c)

        return code_set, cover_count, zero_count, improved

    def local_search_2exchange(code_set, cover_count, zero_count, rng, max_iters=60):
        """
        Try 2-exchange moves: remove two codewords c1, c2 and add two new ones n1, n2.
        Accept if the code remains valid (zero_count == 0).
        """
        improved = False
        code_list = list(code_set)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(code_list) < 2:
                break
            # Pick two distinct codewords to remove
            i1 = rng.randrange(len(code_list))
            i2 = rng.randrange(len(code_list) - 1)
            if i2 >= i1:
                i2 += 1
            c1 = code_list[i1]
            c2 = code_list[i2]

            # Generate two new candidate positions
            d1 = rng.choice(perturbations)
            d2 = rng.choice(perturbations)
            t1 = all_tuples[c1]
            t2 = all_tuples[c2]
            n1 = idx_map[tuple((t1[i] + d1[i]) % q for i in range(n))]
            n2 = idx_map[tuple((t2[i] + d2[i]) % q for i in range(n))]

            if n1 == n2:
                continue
            if n1 in code_set or n2 in code_set:
                continue

            # Apply: remove c1, c2; add n1, n2
            zero_count = try_remove(code_set, cover_count, zero_count, c1)
            zero_count = try_remove(code_set, cover_count, zero_count, c2)
            zero_count = try_add(code_set, cover_count, zero_count, n1)
            zero_count = try_add(code_set, cover_count, zero_count, n2)

            if zero_count == 0:
                improved = True
                code_list[i1] = n1
                code_list[i2] = n2
            else:
                # Revert
                zero_count = try_remove(code_set, cover_count, zero_count, n1)
                zero_count = try_remove(code_set, cover_count, zero_count, n2)
                zero_count = try_add(code_set, cover_count, zero_count, c1)
                zero_count = try_add(code_set, cover_count, zero_count, c2)

        return code_set, cover_count, zero_count, improved

    # Main ILS Loop
    rng = random.Random(42)

    # Initial solution
    code_set, cover_count, zero_count, uncovered_mask = greedy_start()
    code_set, cover_count, zero_count = reduce_code(code_set, cover_count, zero_count)
    code_set, cover_count, zero_count, _ = local_search_1exchange(code_set, cover_count, zero_count, rng, max_iters=100)

    best_size = len(code_set)
    best_code = list(code_set)

    # ILS with structured perturbations
    iteration = 0
    while time.time() < deadline - 2:
        iteration += 1

        # Perturb: structured swap or random removal
        if rng.random() < 0.5:
            # Structured swap: remove one, add a nearby one
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_idx = idx_map[tuple((c_tuple[i] + d[i]) % q for i in range(n))]
            if new_idx not in code_set:
                zero_count = try_remove(code_set, cover_count, zero_count, c)
                zero_count = try_add(code_set, cover_count, zero_count, new_idx)
                # Local search after perturbation
                code_set, cover_count, zero_count, _ = local_search_1exchange(code_set, cover_count, zero_count, rng, max_iters=80)
        else:
            # Random removal of k codewords
            k = rng.randint(2, 5)
            current_code = list(code_set)
            if len(current_code) < k:
                k = len(current_code)
            to_remove = rng.sample(current_code, k)
            for c in to_remove:
                zero_count = try_remove(code_set, cover_count, zero_count, c)
            uncovered_mask = 0
            for i in range(total_words):
                if cover_count[i] == 0:
                    uncovered_mask |= (1 << i)
            code_set, cover_count, zero_count, uncovered_mask = repair(code_set, cover_count, zero_count, uncovered_mask)
            code_set, cover_count, zero_count = reduce_code(code_set, cover_count, zero_count)
            code_set, cover_count, zero_count, _ = local_search_1exchange(code_set, cover_count, zero_count, rng, max_iters=80)

        # Occasionally try a 2-exchange pass to escape 1-exchange local optima
        if iteration % 3 == 0:
            code_set, cover_count, zero_count, _ = local_search_2exchange(code_set, cover_count, zero_count, rng, max_iters=40)

        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

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
