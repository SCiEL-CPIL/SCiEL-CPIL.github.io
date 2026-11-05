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
        
        # Greedy Lower Bound: build a valid independent set
        greedy_set = 0
        remaining = full_mask
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            best_v = -1
            best_deg = n + 1
            m = remaining
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            greedy_set |= (1 << best_v)
            remaining &= ~adj[best_v]
            remaining &= ~(1 << best_v)
        best[0] = bin(greedy_set).count('1')
        
        # Branch and bound for exact maximum independent set
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
            
            # Branch: exclude v first (often prunes better)
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch: include v (only if valid)
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

    def fast_exact_tf(adj):
        """Fast exact check: is there a triangle-free induced subgraph of size > k?"""
        full_mask = (1 << n) - 1
        # Use the same BnB but with early termination
        best = [0]
        
        # Quick greedy lower bound
        greedy_set = 0
        remaining = full_mask
        while remaining:
            best_v = -1
            best_deg = n + 1
            m = remaining
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            greedy_set |= (1 << best_v)
            remaining &= ~adj[best_v]
            remaining &= ~(1 << best_v)
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            # Pick vertex with max degree in C
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
            bnb(S_mask, C_mask & ~v_bit)
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
        
        bnb(0, full_mask)
        return best[0]

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    random.seed(42)
    best_adj = None
    best_score = n

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        adj = generate_random_k4_free()
        
        # Initial Evaluation
        score = largest_tf(adj, time_limit=deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Restart {restarts}: New best {best_score}")
            
        # Local Search: batch edge additions
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            random.shuffle(all_edges)
            # Collect all valid edge additions
            candidates = []
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        candidates.append((u, v))
            
            # Try adding each candidate edge and evaluate
            for u, v in candidates:
                if time.time() > deadline - 5.0:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                new_score = largest_tf(adj, time_limit=deadline - 5.0)
                
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_adj = adj[:]
                        print(f"  Local Search Improvement: {best_score}")
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                            
        if time.time() > deadline - 10.0:
            break

    if best_adj is None:
        # Fallback
        best_adj = generate_random_k4_free()

    # Final verification and potential polish
    if time.time() < deadline - 2.0:
        final_score = fast_exact_tf(best_adj)
        if final_score < best_score:
            best_score = final_score

    edges = get_edge_list(best_adj)
    return edges
