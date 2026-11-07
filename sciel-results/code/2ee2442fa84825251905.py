import sys
sys.setrecursionlimit(3000)

def construct():
    MAX_N = 256
    
    # 1. Precomputation
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    factors = [set() for _ in range(MAX_N + 1)]
    for i in range(1, MAX_N + 1):
        factors[i] = get_prime_factors(i)

    # conflict_mask[i] is a bitmask of all j such that gcd(i,j) > 1
    conflict_mask = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        mask = 0
        fi = factors[i]
        if not fi: continue
        for j in range(1, MAX_N + 1):
            if i == j: continue
            if fi.intersection(factors[j]):
                mask |= (1 << j)
        conflict_mask[i] = mask

    weights = list(range(MAX_N + 1))
    
    best_sum = [0]
    best_set = [set()]

    # Greedy Coloring Upper Bound
    # We process vertices in descending order of weight.
    # This tends to produce a tighter bound because high-weight vertices get distinct colors early.
    def get_bound(candidates_mask):
        # Extract vertices
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        if not verts:
            return 0
            
        verts.sort(key=lambda x: -weights[x])
        
        color_of = {}
        # max_weight_per_color[c] = max weight of vertex with color c
        max_weight_per_color = []
        
        for v in verts:
            # Find colors of neighbors in candidates
            neigh_mask = conflict_mask[v] & candidates_mask
            used_colors = set()
            nm = neigh_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                nm ^= lsb
            
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[v] = c
            if c >= len(max_weight_per_color):
                max_weight_per_color.append(weights[v])
            else:
                if weights[v] > max_weight_per_color[c]:
                    max_weight_per_color[c] = weights[v]
                    
        return sum(max_weight_per_color)

    def solve(candidates_mask, current_sum, current_set):
        if not candidates_mask:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(current_set)
            return

        ub = get_bound(candidates_mask)
        if current_sum + ub <= best_sum[0]:
            return

        # Select vertex with max weight to branch on
        m = candidates_mask
        v = 0
        max_w = -1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                v = idx
            m ^= lsb
        
        # Branch 1: Include v
        # Remove v and its neighbors
        new_mask = candidates_mask & ~(conflict_mask[v] | (1 << v))
        current_set.add(v)
        solve(new_mask, current_sum + weights[v], current_set)
        current_set.remove(v)
        
        # Branch 2: Exclude v
        new_mask = candidates_mask & ~(1 << v)
        solve(new_mask, current_sum, current_set)

    initial_mask = 0
    for i in range(1, MAX_N + 1):
        initial_mask |= (1 << i)
        
    solve(initial_mask, 0, set())
    
    result = list(best_set[0])
    result.sort()
    return result
