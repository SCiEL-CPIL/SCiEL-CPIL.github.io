import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def has_independent_set_10(adj, n):
        mask_all = (1 << n) - 1
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if best_v == -1:
                break
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
        if size >= 10:
            return True
            
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            if best_v == -1:
                return False
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def make_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    def mycielski(adj, n):
        new_n = 2 * n + 1
        new_adj = [0] * new_n
        for i in range(n):
            new_adj[i] = adj[i]
        for i in range(n):
            for j in range(n):
                if adj[i] & (1 << j):
                    new_adj[n + i] |= (1 << j)
                    new_adj[j] |= (1 << (n + i))
        for i in range(n):
            new_adj[2 * n] |= (1 << (n + i))
            new_adj[n + i] |= (1 << (2 * n))
        return new_adj, new_n

    def blow_up(adj, n, sizes):
        new_n = sum(sizes)
        new_adj = [0] * new_n
        offsets = [0] * (n + 1)
        for i in range(n):
            offsets[i + 1] = offsets[i] + sizes[i]
        for i in range(n):
            for j in range(n):
                if i != j and (adj[i] & (1 << j)):
                    for a in range(offsets[i], offsets[i + 1]):
                        for b in range(offsets[j], offsets[j + 1]):
                            new_adj[a] |= (1 << b)
                            new_adj[b] |= (1 << a)
        return new_adj, new_n

    def make_clebsch():
        # Clebsch graph: 16 vertices, strongly regular (16,5,0,2)
        # It is the folded 5-cube.
        # Vertices are binary strings of length 4, identified with their complements.
        # Or simply: 16 vertices, each connected to 5 others.
        # Construction: Start with 5-cycle? No.
        # Clebsch is the complement of the line graph of K_8? No.
        # Let's use a known adjacency list or generation.
        # Vertices 0..15.
        # i connected to j if popcount(i xor j) is 1 or 4?
        # In the 4-cube, degree 4.
        # Clebsch is the quotient of the 4-cube by the antipodal map.
        # So vertices are pairs {x, ~x}.
        # There are 16/2 = 8 pairs? No, 2^4=16 vertices in cube.
        # Quotient has 8 vertices?
        # Wait, Clebsch has 16 vertices.
        # It is the graph on {0,1}^4 where x~y if Hamming distance is 1 or 4?
        # If dist 1, degree 4. If dist 4, degree 1 (complement). Total 5.
        # Yes.
        adj = [0] * 16
        for i in range(16):
            for j in range(16):
                if i != j:
                    d = bin(i ^ j).count('1')
                    if d == 1 or d == 4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        return adj, 16

    best_result = None

    # Andrásfai And(9)
    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}

    # Grötzsch graph (Mycielski of C5)
    adj_c5 = [0] * 5
    for i in range(5):
        adj_c5[i] |= (1 << ((i + 1) % 5))
        adj_c5[i] |= (1 << ((i + 4) % 5))
    
    adj_grotzsch, n_grotzsch = mycielski(adj_c5, 5)
    if n_grotzsch <= 41 and not has_independent_set_10(adj_grotzsch, n_grotzsch):
        if best_result is None or n_grotzsch > best_result['n']:
            best_result = {'n': n_grotzsch, 'edges': build_edge_list(adj_grotzsch, n_grotzsch)}

    # Blow up Grötzsch by 3
    sizes_g = [3] * n_grotzsch
    adj_g_blow, n_g_blow = blow_up(adj_grotzsch, n_grotzsch, sizes_g)
    if n_g_blow <= 41 and not has_independent_set_10(adj_g_blow, n_g_blow):
        if best_result is None or n_g_blow > best_result['n']:
            best_result = {'n': n_g_blow, 'edges': build_edge_list(adj_g_blow, n_g_blow)}

    # Clebsch graph
    adj_clebsch, n_clebsch = make_clebsch()
    if n_clebsch <= 41 and not has_independent_set_10(adj_clebsch, n_clebsch):
        if best_result is None or n_clebsch > best_result['n']:
            best_result = {'n': n_clebsch, 'edges': build_edge_list(adj_clebsch, n_clebsch)}

    # Mycielski of Clebsch
    adj_mc, n_mc = mycielski(adj_clebsch, n_clebsch)
    if n_mc <= 41 and not has_independent_set_10(adj_mc, n_mc):
        if best_result is None or n_mc > best_result['n']:
            best_result = {'n': n_mc, 'edges': build_edge_list(adj_mc, n_mc)}

    # Random search
    min_target = (best_result['n'] + 1) if best_result else 28
    for target_n in range(40, min_target - 1, -1):
        if time.time() > deadline:
            break
        found = False
        attempts = 0
        max_attempts = 5000
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            strategy = random.randint(0, 2)
            if strategy == 0:
                adj = make_random_triangle_free(target_n)
            elif strategy == 1:
                adj = random_bipartite(target_n, 0.5)
                half = target_n // 2
                extra_edges = [(i, j) for i in range(half) for j in range(i + 1, half)]
                extra_edges += [(i, j) for i in range(half, target_n) for j in range(i + 1, target_n)]
                random.shuffle(extra_edges)
                for u, v in extra_edges:
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            else:
                base_n = 5
                base_adj = adj_c5[:]
                sizes = [1] * base_n
                remaining = target_n - base_n
                for _ in range(remaining):
                    sizes[random.randint(0, base_n - 1)] += 1
                adj, _ = blow_up(base_adj, base_n, sizes)
            
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        if found:
            break

    if best_result is None:
        adj9, n9 = make_andrasfai(9)
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
    
    return best_result
