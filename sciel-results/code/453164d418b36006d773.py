import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    # Helper: Check if bitmask represents an independent set
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        # Common neighbors must form an independent set
        return is_independent(common, adj)

    # Exact B&B for largest triangle-free induced subgraph
    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound
        # Heuristic: Order by degree descending, add if valid
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        # B&B
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            # Upper bound
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
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
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v
            # Check if S U {v} is triangle-free
            # This requires N(v) & S to be independent
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
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
    best_score = n # Max possible is n

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    while time.time() < deadline - 2.0:
        # Random restart
        adj = [0] * n
        random.shuffle(all_edges)
        
        # Greedy fill to make it dense
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Evaluate
        t_start = time.time()
        score = largest_tf(adj, time_limit=deadline - 2.0)
        t_end = time.time()
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"New best: {best_score}, time taken: {t_end-t_start:.2f}s")
            
        # Local Search: Try to add more edges
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if not ((adj[u] >> v) & 1):
                        if can_add_edge(adj, u, v):
                            # Tentatively add
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
                            # Check if score improves
                            if random.random() < 0.1: # Check 10% of the time
                                new_score = largest_tf(adj, time_limit=deadline - 2.0)
                                if new_score > best_score: # Score got worse (larger)
                                    # Revert
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                else:
                                    if new_score < best_score:
                                        best_score = new_score
                                        best_adj = adj[:]
                                        improved = True
                                    # Else keep
                            else:
                                # Just keep, assume denser is better
                                pass
                            
                            if time.time() > deadline - 5.0:
                                break
        # End Local Search

    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score}, edges={len(edges)}")
    return edges
