import time
import random
import math

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
        
        # Greedy Lower Bound - fixed: check if v has NO neighbors in greedy_set
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if not (adj[v] & greedy_set):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        # Precompute popcount for speed
        popcount = [0] * (1 << n)
        for i in range(1, 1 << n):
            popcount[i] = popcount[i >> 1] + (i & 1)
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = popcount[S_mask]
            if sz + popcount[C_mask] <= best[0]:
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
                deg_in_c = popcount[adj[vid] & C_mask]
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            bnb(S_mask, C_mask & ~v_bit)
            
            if not (adj[v] & S_mask):
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

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)
    best_adj = None
    best_score = n

    # Try multiple random restarts with simulated annealing
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
        
        # Simulated Annealing Local Search
        T = 2.0  # Initial temperature
        T_min = 0.01
        cooling = 0.98
        improved = True
        sa_iter = 0
        while improved and time.time() < deadline - 5.0:
            improved = False
            sa_iter += 1
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline - 5.0:
                    break
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        new_score = largest_tf(adj, time_limit=deadline - 5.0)
                        
                        if new_score < score:
                            score = new_score
                            improved = True
                            if score < best_score:
                                best_score = score
                                best_adj = adj[:]
                                print(f"  SA Improvement: {best_score}")
                        else:
                            # Accept with probability exp(-(new_score - score) / T)
                            delta = new_score - score
                            if delta > 0 and random.random() < math.exp(-delta / T):
                                score = new_score
                                improved = True
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                        
                        T = max(T_min, T * cooling)
                        
        if time.time() > deadline - 10.0:
            break

    if best_adj is None:
        # Fallback
        best_adj = generate_random_k4_free()

    edges = get_edge_list(best_adj)
    return edges
