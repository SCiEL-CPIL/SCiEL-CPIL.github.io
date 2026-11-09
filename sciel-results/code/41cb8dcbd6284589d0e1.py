import time
import math
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCM table for O(1) lookup
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = a * b // gcd(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad triples and conflict masks
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    edge_weight = [[0.0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lab == lcm_table[a][c] == lcm_table[b][c]:
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    edge_weight[a][b] += 1.0
                    edge_weight[b][a] += 1.0
                    edge_weight[a][c] += 1.0
                    edge_weight[c][a] += 1.0
                    edge_weight[b][c] += 1.0
                    edge_weight[c][b] += 1.0
    
    # Build adjacency for conflict graph
    adj = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if edge_weight[a][b] > 0:
                adj[a].append((b, edge_weight[a][b]))
                adj[b].append((a, edge_weight[b][a]))
    
    # Compute degree-weighted conflict score
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b, w in adj[a]:
            conflict_score[a] += w
    
    # Fast greedy with bitmask
    def weighted_greedy(order):
        S = []
        S_mask = 0
        for x in order:
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S
    
    # Try several greedy orderings
    best_set = []
    best_size = 0
    
    # Natural order
    S = weighted_greedy(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Reverse order
    S = weighted_greedy(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(range(1, N + 1), key=lambda x: conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Sort by conflict score (descending)
    order = sorted(range(1, N + 1), key=lambda x: -conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Sort by number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order = sorted(range(1, N + 1), key=num_prime_factors)
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Spectral ordering: use power iteration on (D - W) to get ranking
    random.seed(42)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = random.random()
    
    norm = math.sqrt(sum(v[i]**2 for i in range(1, N + 1)))
    if norm > 0:
        for i in range(1, N + 1):
            v[i] /= norm
    
    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        new_v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            new_v[i] = conflict_score[i] * v[i]
            for j, w in adj[i]:
                new_v[i] -= w * v[j]
        
        norm = math.sqrt(sum(new_v[i]**2 for i in range(1, N + 1)))
        if norm > 1e-10:
            for i in range(1, N + 1):
                new_v[i] /= norm
        v = new_v
    
    spectral_order = sorted(range(1, N + 1), key=lambda x: -v[x])
    S = weighted_greedy(spectral_order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    spectral_order_asc = sorted(range(1, N + 1), key=lambda x: v[x])
    S = weighted_greedy(spectral_order_asc)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Primes and prime powers first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True
    
    primes = [x for x in range(1, N + 1) if is_prime(x)]
    prime_powers = [x for x in range(1, N + 1) if is_prime_power(x) and x not in primes]
    others = [x for x in range(1, N + 1) if x not in primes and x not in prime_powers]
    
    order_pp = primes + prime_powers + others
    S = weighted_greedy(order_pp)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Local search polish: try to improve best_set by swapping or adding elements
    def polish(S):
        S = S[:]
        S_set = set(S)
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)
        
        improved = True
        while improved and time.time() - start_time < deadline - 1:
            improved = False
            # Try adding each excluded element
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                compatible = True
                for s in S:
                    if conflict[s][x] & S_mask:
                        compatible = False
                        break
                if compatible:
                    S.append(x)
                    S_set.add(x)
                    S_mask |= (1 << x)
                    improved = True
            
            # Try swapping: remove one element, add one that was blocked
            for i in range(len(S)):
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    # Remove S[i]
                    removed = S[i]
                    S_set.remove(removed)
                    S_mask ^= (1 << removed)
                    S[i] = x
                    S_set.add(x)
                    S_mask |= (1 << x)
                    
                    # Check if valid
                    valid = True
                    for j in range(len(S)):
                        if j == i:
                            continue
                        for k in range(j + 1, len(S)):
                            if k == i:
                                continue
                            # Check if S[j], S[k], and x form a bad triple
                            # Actually we need to check all pairs with x
                            pass
                    
                    # Simpler: just check x against all other elements in S
                    for j in range(len(S)):
                        if j == i:
                            continue
                        if conflict[S[j]][x] & S_mask:
                            valid = False
                            break
                    
                    if valid:
                        improved = True
                    else:
                        # Revert
                        S_set.remove(x)
                        S_mask ^= (1 << x)
                        S[i] = removed
                        S_set.add(removed)
                        S_mask |= (1 << removed)
        
        return S
    
    polished = polish(best_set)
    if len(polished) > best_size:
        best_size = len(polished)
        best_set = polished
    
    # Branch and bound with improved upper bound
    def max_clique_bound(available):
        """Upper bound on independent set size using greedy coloring of conflict graph."""
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low
        
        if not avail_list:
            return 0
        
        # Greedy coloring for upper bound on maximum clique in conflict graph
        # (which gives upper bound on independent set in complement)
        # Actually for independent set, we want upper bound.
        # Use: |IS| <= |V| - min_vertex_cover, but that's hard.
        # Simpler: greedy coloring of the conflict graph gives upper bound on clique,
        # not directly on independent set. Let's use a simpler bound.
        
        # Upper bound: number of available elements
        return len(avail_list)
    
    def solve(selected, available):
        nonlocal best_size, best_set
        
        if time.time() - start_time > deadline:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        if not available:
            return
        
        avail_count = available.bit_count()
        if len(selected) + avail_count <= best_size:
            return
        
        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low
        
        if not avail_list:
            return
        
        # Pick element with highest conflict degree in available
        best_x = avail_list[0]
        best_conflict = -1
        for x in avail_list:
            c = 0.0
            for y, w in adj[x]:
                if (available >> y) & 1:
                    c += w
            if c > best_conflict:
                best_conflict = c
                best_x = x
        x = best_x
        
        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        for s in selected:
            new_avail &= ~conflict[s][x]
        
        solve(new_selected, new_avail)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x))
    
    # Start search
    solve([], full_mask)
    
    # Final polish
    polished = polish(best_set)
    if len(polished) > best_size:
        best_size = len(polished)
        best_set = polished
    
    return best_set
