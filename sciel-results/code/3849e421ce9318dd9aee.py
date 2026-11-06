"""Optimized: SA with fast B&B for K4-free graph triangle-free core minimization."""
import random
import time

N = 18
DEADLINE = 145.0

def construct():
    start_time = time.time()
    
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common &= ~(1 << i) & ~(1 << j)
                    c = common
                    while c:
                        lsb = c & (-c)
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        if common & adj[k]:
                            return False
        return True
    
    def max_triangle_free_size(adj):
        best = [0]
        all_mask = (1 << N) - 1
        
        # Precompute: for each pair (i,j) that is an edge, the common neighbors
        # tri_common[i][j] = bitmask of common neighbors of i and j (only if i-j is edge)
        tri_common = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    tri_common[i][j] = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                    tri_common[j][i] = tri_common[i][j]
        
        def bnb(avail, cur_mask, cur_size):
            if time.time() - start_time > DEADLINE - 3:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if avail == 0:
                return
            # Upper bound
            if cur_size + bin(avail).count('1') <= best[0]:
                return
            
            # Greedy upper bound: from avail, greedily add vertices that don't conflict
            # This gives a tighter upper bound
            greedy_avail = avail
            greedy_size = 0
            while greedy_avail:
                lsb = greedy_avail & (-greedy_avail)
                v = lsb.bit_length() - 1
                greedy_size += 1
                # Remove v and its "conflicts" with current set
                forbidden = 0
                cm = cur_mask
                while cm:
                    lsb2 = cm & (-cm)
                    u = lsb2.bit_length() - 1
                    cm ^= lsb2
                    if tri_common[u][v]:
                        forbidden |= tri_common[u][v]
                greedy_avail &= ~(1 << v) & ~forbidden
            if cur_size + greedy_size <= best[0]:
                return
            
            # Pick vertex with max conflicts
            v = -1
            best_conf = -1
            a = avail
            while a:
                lsb = a & (-a)
                i = lsb.bit_length() - 1
                a ^= lsb
                conf = 0
                cm = cur_mask
                while cm:
                    lsb2 = cm & (-cm)
                    u = lsb2.bit_length() - 1
                    cm ^= lsb2
                    if tri_common[u][i]:
                        conf += bin(tri_common[u][i] & avail).count('1')
                if conf > best_conf:
                    best_conf = conf
                    v = i
            
            if v < 0:
                return
            
            # Include v
            forbidden = 0
            cm = cur_mask
            while cm:
                lsb2 = cm & (-cm)
                u = lsb2.bit_length() - 1
                cm ^= lsb2
                if tri_common[u][v]:
                    forbidden |= tri_common[u][v]
            new_avail = avail & ~(1 << v) & ~forbidden
            bnb(new_avail, cur_mask | (1 << v), cur_size + 1)
            
            # Exclude v
            bnb(avail & ~(1 << v), cur_mask, cur_size)
        
        bnb(all_mask, 0, 0)
        return best[0]
    
    # Initialize with K6,6,6
    adj = [0] * N
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    best_score = max_triangle_free_size(adj)
    best_graph = adj[:]
    
    # SA
    T = 1.0
    T_min = 0.01
    cooling = 0.9997
    rng = random.Random(42)
    
    while time.time() - start_time < DEADLINE - 5:
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        new_adj = best_graph[:]
        if new_adj[u] & (1 << v):
            new_adj[u] ^= (1 << v)
            new_adj[v] ^= (1 << u)
        else:
            common = new_adj[u] & new_adj[v]
            common &= ~(1 << u) & ~(1 << v)
            k4_ok = True
            c = common
            while c:
                lsb = c & (-c)
                a = lsb.bit_length() - 1
                c ^= lsb
                if common & new_adj[a]:
                    k4_ok = False
                    break
            if not k4_ok:
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        new_score = max_triangle_free_size(new_adj)
        
        if new_score < best_score:
            best_score = new_score
            best_graph = new_adj[:]
            T = max(T * cooling, T_min)
        else:
            delta = new_score - best_score
            if delta > 0 and rng.random() < 2.718281828 ** (-delta / max(T, 1e-10)):
                best_graph = new_adj[:]
            T = max(T * cooling, T_min)
    
    final_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_graph[i] & (1 << j):
                final_edges.append([i, j])
    
    return final_edges
