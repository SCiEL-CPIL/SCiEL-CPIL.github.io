import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    n_max = 180

    # Precompute forbidden triples: (a,b,c) with a<b<c and 1/a = 1/b + 1/c
    # Equivalent: (b-a)(c-a) = a^2
    forbidden_triples = []
    for a in range(1, n_max + 1):
        a_sq = a * a
        for u in range(1, int(a_sq ** 0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u >= v:
                    continue
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    forbidden_triples.append((a, b, c))

    # conflict_mask[i] (0-indexed): bitmask of all vertices that appear in a forbidden triple with i
    # pair_conflicts[i] (0-indexed): list of (j, k) pairs such that {i, j+1, k+1} is forbidden
    conflict_mask = [0] * n_max
    pair_conflicts = [[] for _ in range(n_max)]

    for a, b, c in forbidden_triples:
        ai, bi, ci = a - 1, b - 1, c - 1
        conflict_mask[ai] |= (1 << bi) | (1 << ci)
        conflict_mask[bi] |= (1 << ai) | (1 << ci)
        conflict_mask[ci] |= (1 << ai) | (1 << bi)
        pair_conflicts[ai].append((bi, ci))
        pair_conflicts[bi].append((ai, ci))
        pair_conflicts[ci].append((ai, bi))

    # triple_count for degree heuristic
    triple_count = [0] * n_max
    for a, b, c in forbidden_triples:
        triple_count[a - 1] += 1
        triple_count[b - 1] += 1
        triple_count[c - 1] += 1

    def is_valid_addition(S, idx):
        """Check if vertex idx can be added to set S."""
        for j, k in pair_conflicts[idx]:
            if (S >> j) & 1 and (S >> k) & 1:
                return False
        return True

    def greedy(order):
        """Greedy construction with given order."""
        S = 0
        for idx in order:
            if (S >> idx) & 1:
                continue
            if is_valid_addition(S, idx):
                S |= (1 << idx)
        return S

    # Multiple greedy orders
    order_deg = list(range(n_max))
    order_deg.sort(key=lambda x: triple_count[x])
    order_asc = list(range(n_max))
    order_desc = list(range(n_max - 1, -1, -1))

    best_S = 0
    best_size = 0

    for order in [order_deg, order_asc, order_desc]:
        S = greedy(order)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Random greedy orders
    for _ in range(200):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(n_max))
        random.shuffle(order_rand)
        S = greedy(order_rand)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Fast local search: only 1-swap moves (remove one, add one)
    def local_search(S, max_iter=200):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() - start_time > deadline:
                break
            # Try adding any vertex
            for idx in range(n_max):
                if (S >> idx) & 1:
                    continue
                if is_valid_addition(S, idx):
                    S |= (1 << idx)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove one, add one
            for r in range(n_max):
                if not ((S >> r) & 1):
                    continue
                S2 = S & ~(1 << r)
                for a in range(n_max):
                    if (S2 >> a) & 1:
                        continue
                    if is_valid_addition(S2, a):
                        S3 = S2 | (1 << a)
                        if S3.bit_count() > S.bit_count():
                            S = S3
                            improved = True
                            break
                if improved:
                    break
        return S

    S_cur = local_search(best_S, max_iter=300)
    best_size = S_cur.bit_count()
    best_S = S_cur

    # Random restarts with perturbation
    for restart in range(100):
        if time.time() - start_time > deadline:
            break
        S_pert = best_S
        S_list = [i for i in range(n_max) if (S_pert >> i) & 1]
        num_remove = random.randint(3, 8)
        if len(S_list) > num_remove:
            remove_set = random.sample(S_list, num_remove)
            for r in remove_set:
                S_pert &= ~(1 << r)
        S_new = local_search(S_pert, max_iter=150)
        if S_new.bit_count() > best_size:
            best_size = S_new.bit_count()
            best_S = S_new

    # Final: try to add any remaining vertices
    S_final = best_S
    for idx in range(n_max):
        if (S_final >> idx) & 1:
            continue
        if is_valid_addition(S_final, idx):
            S_final |= (1 << idx)

    # Branch and bound with time limit
    # We'll try to improve further using B&B on a reduced problem
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5.0:
        # B&B: try to find a larger independent set
        # Start from current best, try to add vertices via B&B
        # This is a simplified approach: try all subsets of missing vertices
        missing = [i for i in range(n_max) if not ((S_final >> i) & 1)]
        
        if missing and remaining_time > 10.0:
            # Try adding subsets of missing vertices
            # Use a greedy B&B approach
            def bb_search(available, current_S, current_size, best_ref):
                if time.time() - start_time > deadline:
                    return
                if not available:
                    if current_size > best_ref[0]:
                        best_ref[0] = current_size
                        best_ref[1] = current_S
                    return
                
                upper = current_size + available.bit_count()
                if upper <= best_ref[0]:
                    return
                
                # Pick vertex with highest conflict count in available
                best_v = -1
                best_conf = -1
                temp = available
                count = 0
                while temp and count < 30:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    cm = conflict_mask[idx] & available
                    cnt = cm.bit_count()
                    if cnt > best_conf:
                        best_conf = cnt
                        best_v = idx
                    temp ^= lsb
                    count += 1
                
                if best_v == -1:
                    return
                
                v_bit = 1 << best_v
                
                # Branch 1: Include best_v
                if is_valid_addition(current_S, best_v):
                    new_avail = available & ~conflict_mask[best_v] & ~v_bit
                    bb_search(new_avail, current_S | v_bit, current_size + 1, best_ref)
                
                # Branch 2: Exclude best_v
                bb_search(available & ~v_bit, current_S, current_size, best_ref)
            
            best_ref = [S_final.bit_count(), S_final]
            avail_mask = 0
            for idx in missing:
                avail_mask |= (1 << idx)
            bb_search(avail_mask, S_final, S_final.bit_count(), best_ref)
            
            if best_ref[0] > best_size:
                best_size = best_ref[0]
                S_final = best_ref[1]

    result = sorted(i + 1 for i in range(n_max) if (S_final >> i) & 1)
    if not result:
        result = [1]
    return result
