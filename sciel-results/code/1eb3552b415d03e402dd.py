# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 82.0
    n = 128
    N = n  # elements are 1..n, use 0-based index: element i is at index i-1
    
    # Precompute LCM table
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # lcm_table[a][b] for 1<=a<=b<=n
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a, n+1):
            lcm_table[a][b] = lcm(a, b)
    
    # For each pair (a,b), find all c such that lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Store as bitmask: conflict_mask[a][b] has bit (c-1) set if c forms bad triple with a,b
    # a,b are 1-based, c is 1-based, bit index is c-1
    conflict_mask = [[0]*(n+1) for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            L = lcm_table[a][b]
            mask = 0
            for c in range(1, n+1):
                if c == a or c == b:
                    continue
                # Need lcm(a,c) == L and lcm(b,c) == L
                if lcm_table[a][c] == L and lcm_table[b][c] == L:
                    mask |= (1 << (c-1))
            conflict_mask[a][b] = mask
            conflict_mask[b][a] = mask
    
    # For a set S (bitmask), can we add element x (1-based)?
    # x can be added iff for all pairs (a,b) in S, x is not in conflict_mask[a][b]
    # Equivalently: for all a in S, the bits in S & conflict_mask[x][a] must be empty (except x itself, but x not in S)
    # Actually: x conflicts with pair (a,b) in S iff conflict_mask[a][b] has bit (x-1) set
    # So x can be added iff for all a in S, (S & conflict_mask[x][a]) == 0 (checking if any b in S has conflict_mask[a][b] containing x)
    # Wait, that's not quite right. Let me think again.
    # x forms a bad triple with a,b (both in S) iff conflict_mask[a][b] has bit (x-1) set.
    # So x can be added iff for all a,b in S with a<b, conflict_mask[a][b] does NOT have bit (x-1).
    # Equivalently: for all a in S, the set {b in S : b != a and conflict_mask[min(a,b)][max(a,b)] has bit (x-1)} must be empty.
    # For efficiency: precompute for each x, a list of pairs (a,b) that conflict with x.
    # But bitmask approach: for each x, define conflict_with_x[a] = bitmask of b such that (a,b) is a conflicting pair for x.
    # Then x can be added to S iff for all a in S, (S & conflict_with_x[a]) == 0.
    
    # Precompute: for each x (1-based), conflict_with_x[x][a] = bitmask of b (1-based, bit b-1) such that x conflicts with pair (a,b)
    # i.e., conflict_mask[a][b] has bit (x-1) set
    conflict_with = [[0]*(n+1) for _ in range(n+1)]  # conflict_with[x][a] = bitmask of b
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            mask = conflict_mask[a][b]
            # For each c in mask, conflict_with[c][a] |= (1<<(b-1)), conflict_with[c][b] |= (1<<(a-1))
            m = mask
            while m:
                low = m & (-m)
                c = low.bit_length()  # c is 1-based since bit (c-1) is set, bit_length gives c
                m ^= low
                conflict_with[c][a] |= (1 << (b-1))
                conflict_with[c][b] |= (1 << (a-1))
    
    def can_add(x, S):
        """Check if element x (1-based) can be added to set S (bitmask)."""
        cx = conflict_with[x]
        s = S
        while s:
            low = s & (-s)
            a = low.bit_length()
            s ^= low
            # Check if any b in S conflicts with pair (a,x) for element x
            if (S & cx[a]):
                return False
        return True
    
    def greedy(order):
        S = 0
        for x in order:
            if not (S & (1 << (x-1))):
                if can_add(x, S):
                    S |= (1 << (x-1))
        return S
    
    def popcount(S):
        return bin(S).count('1')
    
    def elements_of(S):
        elems = []
        s = S
        while s:
            low = s & (-s)
            elems.append(low.bit_length())
            s ^= low
        return elems
    
    best_S = 0
    best_size = 0
    
    # 1. Multiple greedy orderings
    rng = random.Random(12345)
    
    # Natural order
    S1 = greedy(list(range(1, n+1)))
    if popcount(S1) > best_size:
        best_S = S1
        best_size = popcount(S1)
    
    # Reverse
    S2 = greedy(list(range(n, 0, -1)))
    if popcount(S2) > best_size:
        best_S = S2
        best_size = popcount(S2)
    
    # Degree-based (fewer conflicts first)
    degs = [0]*(n+1)
    for x in range(1, n+1):
        degs[x] = bin(conflict_with[x]).count('1')  # not quite right, let me compute differently
    # Actually, degree of x = number of pairs (a,b) that conflict with x
    # = sum over a of popcount(conflict_with[x][a]) / 2 ... no
    # Let me just count: for each x, how many pairs (a,b) have x in conflict_mask[a][b]
    # That's sum over a<b of [conflict_mask[a][b] has bit (x-1)]
    # = for each a, count b>a such that conflict_mask[a][b] has bit (x-1)
    # = for each a, popcount(conflict_with[x][a] & ~((1<<(a-1))-1)) ... complicated
    # Simpler: just count total conflicts for x
    for x in range(1, n+1):
        cnt = 0
        for a in range(1, n+1):
            cnt += bin(conflict_with[x][a]).count('1')
        degs[x] = cnt
    
    deg_order = sorted(range(1, n+1), key=lambda x: degs[x])
    S3 = greedy(deg_order)
    if popcount(S3) > best_size:
        best_S = S3
        best_size = popcount(S3)
    
    # Random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S = greedy(order)
        sz = popcount(S)
        if sz > best_size:
            best_S = S
            best_size = sz
    
    # 2. Local search: 1-for-1 swaps
    def local_search(S, max_iters=500):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            elems = elements_of(S)
            for x in elems:
                S_without_x = S & ~(1 << (x-1))
                for y in range(1, n+1):
                    if y == x or (S_without_x & (1 << (y-1))):
                        continue
                    if can_add(y, S_without_x):
                        S_new = S_without_x | (1 << (y-1))
                        # Same size, but might enable further improvements
                        # Only accept if it leads to better or equal - for now, accept to explore
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
        return S
    
    # 3. 2-opt: remove 2, add 1 (net -1, not useful) or remove 1, add 2 (net +1)
    def two_opt(S):
        """Try to remove 1 element and add 2 elements to increase size by 1."""
        elems = elements_of(S)
        for x in elems:
            S_wo_x = S & ~(1 << (x-1))
            # Try all pairs (y,z) not in S_wo_x
            for y in range(1, n+1):
                if (S_wo_x & (1 << (y-1))):
                    continue
                S_with_y = S_wo_x | (1 << (y-1))
                if not can_add(y, S_wo_x):
                    continue
                for z in range(y+1, n+1):
                    if (S_with_y & (1 << (z-1))):
                        continue
                    if can_add(z, S_with_y):
                        return S_with_y | (1 << (z-1))
        return None
    
    # 4. Simulated annealing
    def simulated_annealing(S, steps=5000):
        cur_S = S
        cur_sz = popcount(S)
        best_S_sa = S
        best_sz_sa = cur_sz
        T = 1.0
        for step in range(steps):
            if time.time() > deadline:
                break
            # Random move: remove random element, add random element
            elems = elements_of(cur_S)
            x = rng.choice(elems)
            S_wo_x = cur_S & ~(1 << (x-1))
            # Try to add a random element
            candidates = [y for y in range(1, n+1) if not (S_wo_x & (1 << (y-1))) and can_add(y, S_wo_x)]
            if not candidates:
                T *= 0.999
                continue
            y = rng.choice(candidates)
            new_S = S_wo_x | (1 << (y-1))
            new_sz = popcount(new_S)
            delta = new_sz - cur_sz
            if delta > 0 or rng.random() < (0.3 ** (-delta / max(T, 0.01))):
                cur_S = new_S
                cur_sz = new_sz
                if cur_sz > best_sz_sa:
                    best_S_sa = cur_S
                    best_sz_sa = cur_sz
            T *= 0.9995
        return best_S_sa, best_sz_sa
    
    # Run local search on best
    if time.time() < deadline:
        S_ls = local_search(best_S)
        if popcount(S_ls) > best_size:
            best_S = S_ls
            best_size = popcount(best_S)
    
    # Try 2-opt repeatedly
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            S_2opt = two_opt(best_S)
            if S_2opt is not None:
                best_S = S_2opt
                best_size = popcount(best_S)
            else:
                break
    
    # Simulated annealing
    if time.time() < deadline:
        S_sa, sz_sa = simulated_annealing(best_S, steps=3000)
        if sz_sa > best_size:
            best_S = S_sa
            best_size = sz_sa
    
    # More random restarts with local search
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S = greedy(order)
        S = local_search(S, max_iters=100)
        sz = popcount(S)
        if sz > best_size:
            best_S = S
            best_size = sz
    
    # Final 2-opt polish
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            S_2opt = two_opt(best_S)
            if S_2opt is not None:
                best_S = S_2opt
                best_size = popcount(best_S)
            else:
                break
    
    result = elements_of(best_S)
    if not result:
        return [1]
    return sorted(result)
# EVOLVE-BLOCK-END
