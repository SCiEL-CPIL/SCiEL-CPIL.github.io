import time
import random
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 55.0  # Safety margin

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute bad triples as bitmasks
    bad_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[b][c] == gab and gcd_table[c][a] == gab:
                    mask |= (1 << c)
            bad_mask[a][b] = mask
            bad_mask[b][a] = mask

    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    # Structural analysis: for each d, find elements x where x is multiple of d
    # and check pairwise coprimality of x/d values
    # Group elements by their "reduced" form under each gcd
    def get_structural_priority(x):
        """Priority based on structural properties: elements with fewer structural conflicts first"""
        # Count how many bad triples x participates in
        count = 0
        for a in range(1, N + 1):
            if a == x:
                continue
            mask = bad_mask[a][x]
            count += bin(mask).count('1')
        # Elements with fewer conflicts are "safer" to include
        # Also consider: elements that are multiples of many small numbers
        # have more structural constraints
        divisors = 0
        for d in range(1, x + 1):
            if x % d == 0:
                divisors += 1
        # Higher priority (added earlier) for elements with fewer conflicts
        # and fewer divisors (less structurally constrained)
        return (count, divisors, x)

    def greedy_build(order):
        S = set()
        S_list = []
        conflict = [0] * (N + 1)
        
        for x in order:
            if conflict[x] == 0:
                S.add(x)
                S_list.append(x)
                for a in S_list[:-1]:
                    mask = bad_mask[a][x]
                    for c in iter_bits(mask):
                        conflict[c] += 1
        
        return S, conflict

    def local_search(S, conflict, time_limit=5.0):
        S_list = list(S)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # 1. Try to add any element with zero conflict
            for x in range(1, N + 1):
                if x not in S and conflict[x] == 0:
                    S.add(x)
                    S_list.append(x)
                    for a in S_list[:-1]:
                        mask = bad_mask[a][x]
                        for c in iter_bits(mask):
                            conflict[c] += 1
                    improved = True
                    break
            
            # 2. Try 1-for-1 swaps
            if not improved:
                found_swap = False
                for u in S_list:
                    for v in range(1, N + 1):
                        if v in S:
                            continue
                        # Check if v can replace u
                        conf_v = conflict[v]
                        for a in S_list:
                            if a == u:
                                continue
                            if (bad_mask[a][u] >> v) & 1:
                                conf_v -= 1
                        if conf_v == 0:
                            # Perform swap
                            S.remove(u)
                            S.add(v)
                            for a in S_list:
                                if a == u:
                                    continue
                                mask = bad_mask[a][u]
                                for c in iter_bits(mask):
                                    conflict[c] -= 1
                            for a in S_list:
                                if a == v:
                                    continue
                                mask = bad_mask[a][v]
                                for c in iter_bits(mask):
                                    conflict[c] += 1
                            S_list.remove(u)
                            S_list.append(v)
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
            
            # 3. Try 2-for-1 swaps (remove two, add one that enables more)
            if not improved:
                found_2swap = False
                S_list_copy = list(S_list)
                for i in range(len(S_list_copy)):
                    for j in range(i + 1, len(S_list_copy)):
                        u, w = S_list_copy[i], S_list_copy[j]
                        # Try adding v to S \ {u, w}
                        for v in range(1, N + 1):
                            if v in S:
                                continue
                            conf_v = conflict[v]
                            for a in S_list:
                                if a == u or a == w:
                                    continue
                                if (bad_mask[a][u] >> v) & 1:
                                    conf_v -= 1
                                if (bad_mask[a][w] >> v) & 1:
                                    conf_v -= 1
                            # Also check conflicts between u,w and v
                            if (bad_mask[u][w] >> v) & 1:
                                conf_v -= 1
                            if conf_v == 0:
                                # Check if this enables net gain (at least break even, prefer gain)
                                # For now, accept if size doesn't decrease
                                S.remove(u)
                                S.remove(w)
                                S.add(v)
                                # Rebuild conflict
                                new_S_list = [x for x in S_list if x != u and x != w]
                                new_conflict = [0] * (N + 1)
                                for ii in range(len(new_S_list)):
                                    a = new_S_list[ii]
                                    for jj in range(ii + 1, len(new_S_list)):
                                        b = new_S_list[jj]
                                        mask = bad_mask[a][b]
                                        for c in iter_bits(mask):
                                            new_conflict[c] += 1
                                # Add v
                                for a in new_S_list:
                                    mask = bad_mask[a][v]
                                    for c in iter_bits(mask):
                                        new_conflict[c] += 1
                                new_S_list.append(v)
                                # Try to add more
                                for x in range(1, N + 1):
                                    if x not in S and new_conflict[x] == 0:
                                        S.add(x)
                                        new_S_list.append(x)
                                        for a in new_S_list[:-1]:
                                            mask = bad_mask[a][x]
                                            for c in iter_bits(mask):
                                                new_conflict[c] += 1
                                if len(S) > len(S_list_copy):
                                    S_list = new_S_list
                                    conflict = new_conflict
                                    found_2swap = True
                                    improved = True
                                    break
                                else:
                                    # Revert
                                    S.add(u)
                                    S.add(w)
                                    S.remove(v)
                        if found_2swap:
                            break
                    if found_2swap:
                        break
            
            if not improved:
                break
        
        return S, conflict

    best_size = 0
    best_set = set()
    
    base_order = list(range(1, N + 1))
    
    # Strategy 1: Natural order
    S, conf = greedy_build(base_order)
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 2: Reverse order
    S, conf = greedy_build(base_order[::-1])
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 3: Structural priority (fewer conflicts first)
    order_struct = sorted(range(1, N + 1), key=get_structural_priority)
    S, conf = greedy_build(order_struct)
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 4: Structural priority reversed
    order_struct_rev = order_struct[::-1]
    S, conf = greedy_build(order_struct_rev)
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 5: Primes first (structurally simple)
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True
    
    primes = [x for x in range(1, N + 1) if is_prime(x)]
    composites = [x for x in range(1, N + 1) if not is_prime(x)]
    order_prime = primes + composites
    S, conf = greedy_build(order_prime)
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Random restarts
    rng = random.Random(42)
    no_improve = 0
    while time.time() < deadline:
        order = base_order[:]
        rng.shuffle(order)
        S, conf = greedy_build(order)
        S, conf = local_search(S, conf, time_limit=1.5)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > 80:
            break

    # Final polish
    conflict = [0] * (N + 1)
    S_list = list(best_set)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            mask = bad_mask[a][b]
            for c in iter_bits(mask):
                conflict[c] += 1
                
    for x in range(1, N + 1):
        if x not in best_set and conflict[x] == 0:
            best_set.add(x)
            for a in S_list:
                mask = bad_mask[a][x]
                for c in iter_bits(mask):
                    conflict[c] += 1
            S_list.append(x)

    return list(best_set) if best_set else [1]
