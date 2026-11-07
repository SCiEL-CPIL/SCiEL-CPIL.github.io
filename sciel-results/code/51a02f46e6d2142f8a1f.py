import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)

    N = 256
    candidates = list(range(2, N + 1))
    M = len(candidates)  # 255 elements

    # Precompute: for each pair (i,j), whether there exists k != i,j such that
    # candidates[i] | candidates[j]*candidates[k]
    # If yes, then i and j cannot both be in S along with any such k.
    # Key insight: if we precompute conflict[i][j] = True if exists k≠i,j with i|j*k,
    # then a valid set S must satisfy: for all distinct i,j in S, conflict[i][j] is False
    # OR if conflict[i][j] is True, then NO k that causes the conflict is in S.
    
    # Actually, let me think more carefully.
    # The constraint is: for all distinct a,b,c in S, a ∤ b*c.
    # This is equivalent to: for all distinct i,j in S, there is no k in S (k≠i,j) 
    # such that candidates[i] | candidates[j]*candidates[k].
    
    # So I'll precompute for each pair (i,j): the set of k values that would make 
    # the triple invalid. Then in B&B, when I have a partial set, I can check validity.

    # Precompute: bad_k[i][j] = bitmask of k (k≠i, k≠j) such that candidates[i] | candidates[j]*candidates[k]
    # Note: this is directional. We also need bad_k[j][i] for the reverse.
    # Actually for triple (i,j,k) to be invalid, we need ANY of:
    #   candidates[i] | candidates[j]*candidates[k]
    #   candidates[j] | candidates[i]*candidates[k]  
    #   candidates[k] | candidates[i]*candidates[j]
    
    # Let me precompute: for each ordered pair (i,j), i≠j:
    #   mask[i][j] = bitmask of k such that candidates[i] | candidates[j]*candidates[k]
    # Then triple (i,j,k) is invalid iff:
    #   bit k in mask[i][j] OR bit k in mask[j][i] OR bit j in mask[k][i] (i.e., k|i*j)
    # Wait, the third condition is candidates[k] | candidates[i]*candidates[j], which is
    # bit (i*j mod something)... no. Let me use: candidates[k] | candidates[i]*candidates[j]
    # means (candidates[i]*candidates[j]) % candidates[k] == 0.
    # This is equivalent to: for pair (i,j), the set of k that divide i*j.
    # Let me precompute: divisors_of_product[i][j] = bitmask of k such that candidates[k] | candidates[i]*candidates[j]

    # Precompute masks
    # mask_div[i][j] = bitmask of k (k != i, k != j) such that candidates[i] | candidates[j]*candidates[k]
    # This means: candidates[j]*candidates[k] is divisible by candidates[i]
    # Equivalent: candidates[k] is divisible by candidates[i] / gcd(candidates[i], candidates[j])
    # Let g = gcd(candidates[i], candidates[j]), then we need candidates[i]/g | candidates[k]
    
    mask_div = [[0] * M for _ in range(M)]
    for i in range(M):
        ci = candidates[i]
        for j in range(M):
            if i == j:
                continue
            cj = candidates[j]
            g = gcd(ci, cj)
            needed = ci // g  # we need candidates[k] divisible by this
            if needed == 1:
                # ci | cj always, so ci | cj*ck for any k
                # mask includes all k != i, j
                m = 0
                for k in range(M):
                    if k != i and k != j:
                        m |= (1 << k)
                mask_div[i][j] = m
            else:
                m = 0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    if candidates[k] % needed == 0:
                        m |= (1 << k)
                mask_div[i][j] = m

    # For triple (i,j,k) to be invalid:
    #   (bit k in mask_div[i][j]) OR (bit k in mask_div[j][i]) OR (bit i in mask_div[k][j]) 
    # Wait no. Let me re-derive.
    # Triple (i,j,k) invalid if:
    #   candidates[i] | candidates[j]*candidates[k]  -> bit k in mask_div[i][j]
    #   candidates[j] | candidates[i]*candidates[k]  -> bit k in mask_div[j][i]
    #   candidates[k] | candidates[i]*candidates[j]  -> bit i in mask_div[k][j] (since k|i*j means i is a "k" for pair (j,k)?)
    # Actually: candidates[k] | candidates[i]*candidates[j] means for pair (i,j), k divides their product.
    # Let me define: prod_div[i][j] = bitmask of k such that candidates[k] | candidates[i]*candidates[j]
    
    prod_div = [[0] * M for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            pi = candidates[i] * candidates[j]
            m = 0
            for k in range(M):
                if k != i and k != j and pi % candidates[k] == 0:
                    m |= (1 << k)
            prod_div[i][j] = m
            prod_div[j][i] = m

    # Now triple (i,j,k) is invalid iff:
    #   (mask_div[i][j] >> k) & 1  OR  (mask_div[j][i] >> k) & 1  OR  (prod_div[i][j] >> k) & 1
    # Wait, the third: candidates[k] | candidates[i]*candidates[j] is captured by prod_div[i][j] having bit k.
    # Yes! So triple invalid iff:
    #   bit k in mask_div[i][j] OR bit k in mask_div[j][i] OR bit k in prod_div[i][j]
    
    # Precompute: for each pair (i,j), the "conflict mask" = mask_div[i][j] | mask_div[j][i] | prod_div[i][j]
    # This is the set of k that would make triple (i,j,k) invalid.
    conflict_mask = [[0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j:
                conflict_mask[i][j] = mask_div[i][j] | mask_div[j][i] | prod_div[i][j]

    # For a set S to be valid: for all distinct i,j in S, (conflict_mask[i][j] & S_mask) == 0
    # where S_mask is the bitmask of S (excluding i and j themselves, but conflict_mask already excludes i,j)
    
    # Now let's do branch and bound.
    # State: current_mask (bitmask of selected), available_mask (bitmask of candidates not yet decided)
    # When we add vertex v to current:
    #   For each u in current: check (conflict_mask[v][u] & current_mask) == 0
    #   Actually we need: for each u in current, (conflict_mask[v][u] & current_mask) should have no bits
    #   Wait, conflict_mask[v][u] contains k values. We need that none of those k are in current.
    #   So: (conflict_mask[v][u] & current_mask) == 0 for all u in current.
    
    def can_add(v, current_mask):
        """Check if v can be added to the set represented by current_mask"""
        cm = current_mask
        while cm:
            lsb = cm & (-cm)
            u = lsb.bit_length() - 1
            cm ^= lsb
            if (conflict_mask[v][u] & current_mask) != 0:
                return False
        return True

    # Upper bound: |current| + |available| (trivial)
    # Better bound: count how many available vertices can actually be added
    
    # Branch and bound
    best_mask = [0]
    best_size = [0]
    nodes_visited = [0]

    def bb(current_mask, available_mask, current_size):
        nodes_visited[0] += 1
        if time.time() > deadline:
            return
        
        if current_size + available_mask.bit_count() <= best_size[0]:
            return
        
        if available_mask == 0:
            if current_size > best_size[0]:
                best_mask[0] = current_mask
                best_size[0] = current_size
            return
        
        # Pick branching vertex: choose one with highest "conflict degree" in available
        # to prune more
        v = -1
        best_deg = -1
        cm = available_mask
        while cm:
            lsb = cm & (-cm)
            u = lsb.bit_length() - 1
            cm ^= lsb
            # Count how many available vertices conflict with u
            deg = 0
            am = available_mask
            while am:
                lsb2 = am & (-am)
                w = lsb2.bit_length() - 1
                am ^= lsb2
                if w != u and (conflict_mask[u][w] & available_mask) != 0:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                v = u
        
        if v == -1:
            return

        # Branch 1: include v
        if can_add(v, current_mask):
            new_current = current_mask | (1 << v)
            new_available = available_mask & ~(1 << v)
            # Remove from available any vertex that conflicts with v given current
            # A vertex u in new_available can only be added if it doesn't conflict with v
            # i.e., (conflict_mask[v][u] & new_current) == 0
            # Since new_current = current_mask | (1<<v), and conflict_mask[v][u] excludes v,
            # we need (conflict_mask[v][u] & current_mask) == 0
            # But we already checked can_add(v, current_mask) which checks all u in current.
            # For future additions, we need to filter available.
            # Actually, let's just filter: remove u from available if adding v makes u impossible
            # u is impossible if (conflict_mask[v][u] & (current_mask | (1<<v))) != 0
            # Since conflict_mask[v][u] doesn't contain v (it excludes i,j), this is:
            # (conflict_mask[v][u] & current_mask) != 0
            # But wait, can_add already verified this for all u in current.
            # The issue is: u might conflict with v in the sense that some k in current is in conflict_mask[v][u]
            # which means v and u can't both be in S with that k. But k is in current, so u can't be added.
            # So we should remove such u from available.
            
            filtered_available = 0
            am = new_available
            while am:
                lsb = am & (-am)
                u = lsb.bit_length() - 1
                am ^= lsb
                # Check if u can still be added (it needs to not conflict with v given current)
                if (conflict_mask[v][u] & current_mask) == 0:
                    # Also need to check u doesn't conflict with other elements in current
                    # But that's handled recursively
                    filtered_available |= (1 << u)
            bb(new_current, filtered_available, current_size + 1)
        
        # Branch 2: exclude v
        bb(current_mask, available_mask & ~(1 << v), current_size)

    # Initial greedy to get a good lower bound
    def greedy_initial():
        s = []
        s_mask = 0
        # Try multiple orderings
        best_g = []
        best_g_mask = 0
        orders = [
            sorted(range(M), reverse=True),
            sorted(range(M)),
            sorted(range(M), key=lambda x: candidates[x]),
        ]
        # Add prime-first
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i+2) == 0:
                    return False
                i += 6
            return True
        primes = [i for i in range(M) if is_prime(candidates[i])]
        composites = [i for i in range(M) if not is_prime(candidates[i])]
        orders.append(primes + composites)
        orders.append(sorted(primes, reverse=True) + sorted(composites, reverse=True))
        
        for order in orders:
            s = []
            s_mask = 0
            for v in order:
                if can_add(v, s_mask):
                    s.append(v)
                    s_mask |= (1 << v)
            if len(s) > len(best_g):
                best_g = s
                best_g_mask = s_mask
        return best_g_mask, len(best_g)

    init_mask, init_size = greedy_initial()
    if init_size > best_size[0]:
        best_mask[0] = init_mask
        best_size[0] = init_size

    # Run B&B
    all_mask = (1 << M) - 1
    bb(0, all_mask, 0)

    # Local search: try 1-1 swaps and 1-2 exchanges
    def local_search(start_mask, max_iters=10000):
        mask = start_mask
        size = mask.bit_count()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            
            # 1-1 swap: remove one, add one
            cm = mask
            while cm:
                lsb = cm & (-cm)
                rem = lsb.bit_length() - 1
                cm ^= lsb
                new_mask = mask & ~(1 << rem)
                # Try to add any available vertex
                am = all_mask & ~new_mask
                while am:
                    lsb2 = am & (-am)
                    add = lsb2.bit_length() - 1
                    am ^= lsb2
                    if (1 << add) & new_mask:
                        continue
                    if can_add(add, new_mask):
                        candidate_mask = new_mask | (1 << add)
                        if candidate_mask.bit_count() > size:
                            mask = candidate_mask
                            size = mask.bit_count()
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # 1-2 exchange: remove one, add two
            cm = mask
            while cm:
                lsb = cm & (-cm)
                rem = lsb.bit_length() - 1
                cm ^= lsb
                new_mask = mask & ~(1 << rem)
                am = all_mask & ~new_mask
                # Try pairs
                am1 = am
                while am1:
                    lsb2 = am1 & (-am1)
                    add1 = lsb2.bit_length() - 1
                    am1 ^= lsb2
                    if (1 << add1) & new_mask:
                        continue
                    if can_add(add1, new_mask):
                        mid_mask = new_mask | (1 << add1)
                        am2 = am & ~mid_mask
                        while am2:
                            lsb3 = am2 & (-am2)
                            add2 = lsb3.bit_length() - 1
                            am2 ^= lsb3
                            if (1 << add2) & mid_mask:
                                continue
                            if can_add(add2, mid_mask):
                                candidate_mask = mid_mask | (1 << add2)
                                if candidate_mask.bit_count() > size:
                                    mask = candidate_mask
                                    size = mask.bit_count()
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 2-1 exchange: remove two, add one (usually not helpful for maximizing, skip)
            
        return mask, size

    # Run local search on best found
    if time.time() < deadline - 5:
        improved_mask, improved_size = local_search(best_mask[0])
        if improved_size > best_size[0]:
            best_mask[0] = improved_mask
            best_size[0] = improved_size

    # Randomized restarts with greedy + local search
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        order = list(range(M))
        random.shuffle(order)
        s = []
        s_mask = 0
        for v in order:
            if can_add(v, s_mask):
                s.append(v)
                s_mask |= (1 << v)
        if len(s) > best_size[0]:
            best_mask[0] = s_mask
            best_size[0] = len(s)
        if time.time() < deadline - 10:
            im, isz = local_search(s_mask, max_iters=3000)
            if isz > best_size[0]:
                best_mask[0] = im
                best_size[0] = isz

    # Final polish
    if time.time() < deadline - 3:
        im, isz = local_search(best_mask[0], max_iters=20000)
        if isz > best_size[0]:
            best_mask[0] = im
            best_size[0] = isz

    # Convert bitmask to list
    result = []
    cm = best_mask[0]
    while cm:
        lsb = cm & (-cm)
        v = lsb.bit_length() - 1
        result.append(candidates[v])
        cm ^= lsb

    # Final validation
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = result[i], result[j], result[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    # Fallback: return primes
                    result = [x for x in range(2, 257) if all(x % p != 0 for p in range(2, int(x**0.5)+1))]
                    break
            else:
                continue
            break

    return result
