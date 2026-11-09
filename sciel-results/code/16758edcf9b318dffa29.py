import random
import time
from itertools import combinations

def _is_k4_free(adj, n=18):
    """Fast K4-free check using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                if common_abc:
                    return False
                common_ab ^= lsb
    return True

def _has_triangle_mask(mask, adj, n=18):
    """Check if induced subgraph on mask has a triangle."""
    m = mask
    while m:
        lsb_u = m & -m
        u = lsb_u.bit_length() - 1
        m ^= lsb_u
        nbrs_u = adj[u] & mask & ~((1 << (u + 1)) - 1)
        while nbrs_u:
            lsb_v = nbrs_u & -nbrs_u
            v = lsb_v.bit_length() - 1
            nbrs_u ^= lsb_v
            if adj[u] & adj[v] & mask:
                return True
    return False

def _exists_tf_subset(k, adj, n=18, deadline=None):
    """
    Check if there exists a triangle-free induced subgraph of size k.
    Uses backtracking with pruning.
    """
    vertices = list(range(n))
    
    def backtrack(start_idx, chosen_count, chosen_mask):
        if deadline and time.time() > deadline:
            return False
        if chosen_count == k:
            return not _has_triangle_mask(chosen_mask, adj, n)
        if start_idx >= n:
            return False
        if n - start_idx < k - chosen_count:
            return False
        
        v = vertices[start_idx]
        
        # Try including v
        new_mask = chosen_mask | (1 << v)
        # Check if adding v creates a triangle with already chosen vertices
        nbrs_v_in_chosen = adj[v] & chosen_mask
        creates_tri = False
        m = nbrs_v_in_chosen
        while m:
            lsb_u = m & -m
            u = lsb_u.bit_length() - 1
            m ^= lsb_u
            common = adj[u] & adj[v] & chosen_mask
            if common & ~(1 << u) & ~(1 << v):
                creates_tri = True
                break
        if not creates_tri:
            if backtrack(start_idx + 1, chosen_count + 1, new_mask):
                return True
        
        # Try excluding v
        if backtrack(start_idx + 1, chosen_count, chosen_mask):
            return True
        
        return False
    
    return backtrack(0, 0, 0)

def _compute_largest_tf(adj, n=18, deadline=None):
    """
    Find the largest triangle-free induced subgraph.
    Uses backtracking to check if TF subset of size k exists.
    """
    for k in range(n, 0, -1):
        if _exists_tf_subset(k, adj, n, deadline):
            return k
    return 0

def _adj_to_edges(adj, n=18):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _gen_random_k4_free(n=18, seed=None):
    """Generate a random K4-free graph."""
    if seed is not None:
        random.seed(seed)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Repair
    for _ in range(200):
        if _is_k4_free(adj, n):
            return adj
        found = False
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)): continue
                common = adj[a] & adj[b]
                common &= ~((1 << (b + 1)) - 1)
                c = common
                while c:
                    lsb = c & -c
                    x = lsb.bit_length() - 1
                    c -= lsb
                    common_abc = adj[a] & adj[b] & adj[x]
                    common_abc &= ~((1 << (x + 1)) - 1)
                    if common_abc:
                        # Find d
                        d = (common_abc & -common_abc).bit_length() - 1
                        # Remove edge (a, d)
                        adj[a] &= ~(1 << d)
                        adj[d] &= ~(1 << a)
                        found = True
                        break
                if found: break
            if found: break
        if not found:
            return adj
    return adj

def _gen_turan_k(n=18, parts_count=3, seed=None):
    """Generate Turan graph T(n, parts_count)."""
    if seed is not None:
        random.seed(seed)
    adj = [0] * n
    # Split vertices into parts as evenly as possible
    sizes = [n // parts_count] * parts_count
    remainder = n % parts_count
    for i in range(remainder):
        sizes[i] += 1
    
    parts = []
    start = 0
    for size in sizes:
        parts.append(list(range(start, start + size)))
        start += size
    
    for i in range(parts_count):
        for j in range(i + 1, parts_count):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    return adj

def _gen_4partite(n=18, seed=None):
    """Generate a 4-partite graph with random edge removals."""
    if seed is not None:
        random.seed(seed)
    adj = [0] * n
    # 4 parts: 4, 4, 5, 5
    parts = [list(range(0, 4)), list(range(4, 8)), list(range(8, 13)), list(range(13, 18))]
    
    for i in range(4):
        for j in range(i + 1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Remove random edges to ensure K4-free
    for _ in range(1000):
        if _is_k4_free(adj, n):
            break
        # Find a K4 and remove an edge
        found = False
        for a in parts[0]:
            for b in parts[1]:
                if not (adj[a] & (1 << b)):
                    continue
                for c in parts[2]:
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                        continue
                    for d in parts[3]:
                        if not (adj[a] & (1 << d)) or not (adj[b] & (1 << d)) or not (adj[c] & (1 << d)):
                            continue
                        edge_choices = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                        e = random.choice(edge_choices)
                        adj[e[0]] &= ~(1 << e[1])
                        adj[e[1]] &= ~(1 << e[0])
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    
    best_l = n + 1
    best_adj = None
    
    def update_best(adj, l):
        nonlocal best_l, best_adj
        if l < best_l:
            best_l = l
            best_adj = adj[:]
    
    random.seed(42)
    
    candidates = []
    
    # Generate diverse initial candidates
    # 1. Turan T(18,3)
    adj_turan3 = _gen_turan_k(n, 3, seed=42)
    candidates.append(adj_turan3)
    
    # 2. Turan T(18,4)
    adj_turan4 = _gen_turan_k(n, 4, seed=43)
    candidates.append(adj_turan4)
    
    # 3. Turan T(18,5)
    adj_turan5 = _gen_turan_k(n, 5, seed=44)
    candidates.append(adj_turan5)
    
    # 4. Random K4-free graphs
    for seed in range(45, 65):
        adj_rand = _gen_random_k4_free(n, seed=seed)
        if adj_rand:
            candidates.append(adj_rand)
    
    # 5. 4-partite graphs
    for seed in range(65, 85):
        adj_4p = _gen_4partite(n, seed=seed)
        if _is_k4_free(adj_4p, n):
            candidates.append(adj_4p)
    
    # Evaluate all candidates
    for adj in candidates:
        if time.time() > deadline:
            break
        l = _compute_largest_tf(adj, n, deadline)
        update_best(adj, l)
    
    if best_adj is not None and time.time() < deadline - 10:
        current_adj = best_adj[:]
        current_l = best_l
        T = 1.0
        
        for step in range(10000):
            if time.time() > deadline:
                break
            
            i, j = random.sample(range(n), 2)
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            if not _is_k4_free(current_adj, n):
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue
            
            new_l = _compute_largest_tf(current_adj, n, deadline)
            
            if new_l < current_l:
                current_l = new_l
                update_best(current_adj, new_l)
            elif new_l == current_l:
                pass
            else:
                delta = new_l - current_l
                if random.random() < pow(2.71828, -delta / T):
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            
            T *= 0.998
    
    if best_adj is None:
        return _adj_to_edges(candidates[-1])
    
    return _adj_to_edges(best_adj)
