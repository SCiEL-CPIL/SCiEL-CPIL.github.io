import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound - try multiple orderings
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        # Try degree ordering
        order = sorted(range(n), key=lambda x: -degrees[x])
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        # Try random orderings for better lower bound
        for _ in range(5):
            random_order = list(range(n))
            random.shuffle(random_order)
            greedy_set2 = 0
            for v in random_order:
                if is_independent(adj[v] & greedy_set2, adj):
                    greedy_set2 |= (1 << v)
            if bin(greedy_set2).count('1') > best[0]:
                best[0] = bin(greedy_set2).count('1')
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C for better pruning
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # Branch 1: Include v first (more likely to find good solution)
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
            
            # Branch 2: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    random.seed(42)
    best_adj = [0] * n
    best_score = n

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    while time.time() < deadline - 2.0:
        adj = [0] * n
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        score = largest_tf(adj, time_limit=deadline - 2.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"New best: {best_score}")
            
        # Local Search: Try adding and removing edges
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if time.time() > deadline - 5.0:
                        break
                    has_edge = (adj[u] >> v) & 1
                    if not has_edge:
                        if can_add_edge(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            new_score = largest_tf(adj, time_limit=deadline - 5.0)
                            if new_score < best_score:
                                best_score = new_score
                                best_adj = adj[:]
                                improved = True
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                    else:
                        # Try removing edge
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        new_score = largest_tf(adj, time_limit=deadline - 5.0)
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = adj[:]
                            improved = True
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score}, edges={len(edges)}")
    return edges
