import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115
    rng = random.Random(42)

    def has_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False

    def find_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None

    def find_ind_k(n, A, k):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            nonlocal found
            if found[0] is not None:
                return
            if len(R) == k:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < k:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    def max_independent_set_size(n, A):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        best = [0]
        
        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count("1") <= best[0]:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return best[0]

    def valid(n, A):
        if has_k4(n, A):
            return False
        if max_independent_set_size(n, A) >= 6:
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def check_add_vertex_fast(cur_n, cur_A, neighbors_mask):
        """
        Check if adding a new vertex with given neighbors_mask maintains:
        1. K4-free: N(new) must be triangle-free
        2. alpha < 6: G[V \\ N(new)] must have alpha < 5
        """
        # Check 1: N(new) must be triangle-free
        neigh_list = []
        m = neighbors_mask
        while m:
            low = m & -m
            v = low.bit_length() - 1
            neigh_list.append(v)
            m ^= low
            
        L = len(neigh_list)
        for i in range(L):
            u = neigh_list[i]
            for j in range(i + 1, L):
                v = neigh_list[j]
                if not ((cur_A[v] >> u) & 1):
                    continue
                common = cur_A[u] & cur_A[v] & neighbors_mask
                if common:
                    return False

        # Check 2: G[V \\ N(new)] must have alpha < 5
        non_neigh_mask = ((1 << cur_n) - 1) & ~neighbors_mask
        if bin(non_neigh_mask).count("1") < 5:
            return True

        sub_verts = []
        m = non_neigh_mask
        while m:
            low = m & -m
            v = low.bit_length() - 1
            sub_verts.append(v)
            m ^= low
            
        k_sub = len(sub_verts)
        if k_sub < 5:
            return True
            
        idx_map = {v: i for i, v in enumerate(sub_verts)}
        sub_A = [0] * k_sub
        for i in range(k_sub):
            u = sub_verts[i]
            for j in range(i + 1, k_sub):
                v = sub_verts[j]
                if (cur_A[v] >> u) & 1:
                    sub_A[i] |= (1 << j)
                    sub_A[j] |= (1 << i)
                    
        if find_ind_k(k_sub, sub_A, 5) is not None:
            return False
        return True

    def try_grow(cur_n, cur_A, deadline):
        """Try to grow by adding one vertex with a valid neighborhood."""
        if cur_n >= 40:
            return None
        
        # Try multiple strategies
        for strategy in range(30):
            if time.time() > deadline - 15:
                break
            
            if strategy == 0:
                # Greedy: build triangle-free set, then check
                verts = list(range(cur_n))
                rng.shuffle(verts)
                current_mask = 0
                for v in verts:
                    neighbors_of_v_in_current = cur_A[v] & current_mask
                    has_edge = False
                    m = neighbors_of_v_in_current
                    while m:
                        low = m & -m
                        u = low.bit_length() - 1
                        m ^= low
                        if cur_A[u] & (neighbors_of_v_in_current & ~(1 << u)):
                            has_edge = True
                            break
                    if not has_edge:
                        current_mask |= (1 << v)
                
                if current_mask and check_add_vertex_fast(cur_n, cur_A, current_mask):
                    return current_mask
            elif strategy == 1:
                # Random subset of size ~cur_n/2
                target_size = cur_n // 2 + rng.randint(-2, 2)
                verts = list(range(cur_n))
                rng.shuffle(verts)
                current_mask = 0
                for v in verts[:target_size]:
                    current_mask |= (1 << v)
                if check_add_vertex_fast(cur_n, cur_A, current_mask):
                    return current_mask
            else:
                # Start with half, then add/remove to fix
                verts = list(range(cur_n))
                rng.shuffle(verts)
                current_mask = 0
                for v in verts[:cur_n // 2]:
                    current_mask |= (1 << v)
                
                # Try to fix by adding vertices
                for v in verts[cur_n // 2:]:
                    if time.time() > deadline - 15:
                        break
                    new_mask = current_mask | (1 << v)
                    if check_add_vertex_fast(cur_n, cur_A, new_mask):
                        return new_mask
                    current_mask = new_mask
                
                if check_add_vertex_fast(cur_n, cur_A, current_mask):
                    return current_mask
        
        return None

    def local_repair(cur_n, cur_A, rng, deadline):
        """Fix K4s and independent sets by flipping edges."""
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            k4 = find_k4(cur_n, cur_A)
            if k4:
                a, b, c, d = k4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                cur_A[u] &= ~(1 << v)
                cur_A[v] &= ~(1 << u)
                continue
            
            ind6 = find_ind_k(cur_n, cur_A, 6)
            if ind6:
                verts = list(ind6)
                u, v = rng.sample(verts, 2)
                cur_A[u] |= (1 << v)
                cur_A[v] |= (1 << u)
                continue
            
            break
        
        return cur_A

    def random_flips(cur_n, cur_A, rng, deadline, n_flips=3):
        for _ in range(n_flips):
            if time.time() > deadline - 5:
                break
            i = rng.randrange(cur_n)
            j = rng.randrange(cur_n)
            if i == j:
                continue
            if (cur_A[j] >> i) & 1:
                cur_A[i] &= ~(1 << j)
                cur_A[j] &= ~(1 << i)
            else:
                cur_A[i] |= (1 << j)
                cur_A[j] |= (1 << i)
        return cur_A

    # Start with a good seed: Paley(17)
    q = 17
    squares = {x * x % q for x in range(1, q)}
    n0 = q
    A = [0] * n0
    for i in range(n0):
        for j in range(n0):
            if i != j and (j - i) % q in squares:
                A[i] |= (1 << j)
    
    assert valid(n0, A), "Paley(17) should be valid"
    best_n = n0
    best_A = list(A)

    # Main search loop
    while time.time() < deadline - 10:
        cur_n = best_n
        cur_A = list(best_A)
        
        # Try to grow
        if cur_n < 40:
            mask = try_grow(cur_n, cur_A, deadline)
            if mask is not None:
                new_n = cur_n + 1
                new_A = cur_A + [0]
                for i in range(cur_n):
                    if (mask >> i) & 1:
                        new_A[i] |= (1 << new_n - 1)
                        new_A[new_n - 1] |= (1 << i)
                
                if valid(new_n, new_A):
                    best_n = new_n
                    best_A = new_A
                    continue
        
        # Perturb and repair
        if time.time() < deadline - 20:
            perturbed = random_flips(cur_n, cur_A, rng, deadline, n_flips=rng.randint(1, 5))
            repaired = local_repair(cur_n, perturbed, rng, deadline)
            if not has_k4(cur_n, repaired) and max_independent_set_size(cur_n, repaired) < 6:
                mask = try_grow(cur_n, repaired, deadline)
                if mask is not None:
                    new_n = cur_n + 1
                    new_A = repaired + [0]
                    for i in range(cur_n):
                        if (mask >> i) & 1:
                            new_A[i] |= (1 << new_n - 1)
                            new_A[new_n - 1] |= (1 << i)
                    if valid(new_n, new_A):
                        best_n = new_n
                        best_A = new_A

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
# EVOLVE-BLOCK-END
