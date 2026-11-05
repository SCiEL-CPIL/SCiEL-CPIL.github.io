import time
import random
import math
import numpy as np

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

    def can_remove_edge(adj, u, v):
        # Always safe to remove an edge
        return True

    def largest_tf_fast(adj, time_limit=None):
        """Exact branch-and-bound for largest triangle-free induced subgraph."""
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy lower bound
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        greedy_set = 0
        for v in order:
            if not (adj[v] & greedy_set):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        # Popcount lookup
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

            # Pick vertex with max degree in C for better pruning
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

    def build_2c9():
        """Build two disjoint C9 cycles on 18 vertices. Score = 8."""
        adj = [0] * n
        # First C9: vertices 0-8
        for i in range(9):
            j = (i + 1) % 9
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        # Second C9: vertices 9-17
        for i in range(9):
            j = (i + 1) % 9
            a = 9 + i
            b = 9 + j
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        return adj

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def sa_search(adj, score, time_limit):
        """Simulated annealing with edge add/remove moves."""
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        T = 1.5
        T_min = 0.05
        cooling = 0.995
        
        while time.time() < time_limit:
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > time_limit:
                    break
                has_edge = (adj[u] >> v) & 1
                
                if has_edge:
                    # Try removing edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    new_score = largest_tf_fast(adj, time_limit=time_limit)
                    delta = new_score - score
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                        score = new_score
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Try adding edge (must be K4-free)
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        new_score = largest_tf_fast(adj, time_limit=time_limit)
                        delta = new_score - score
                        if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                            score = new_score
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
            
            T = max(T_min, T * cooling)
        
        return score

    random.seed(42)
    best_adj = None
    best_score = n

    # Start with 2*C9 baseline (score 8)
    adj = build_2c9()
    score = largest_tf_fast(adj, time_limit=deadline - 10.0)
    best_score = score
    best_adj = adj[:]
    print(f"Baseline 2xC9: score = {score}")

    # SA from 2xC9
    time_per_phase = min(30.0, (deadline - start_time) * 0.4)
    phase_end = start_time + time_per_phase
    score = sa_search(adj, score, phase_end)
    if score < best_score:
        best_score = score
        best_adj = adj[:]
        print(f"After SA from 2xC9: score = {best_score}")

    # Random restarts
    remaining = deadline - 5.0 - time.time()
    if remaining > 5.0:
        restarts = 0
        while time.time() < deadline - 5.0:
            restarts += 1
            adj = generate_random_k4_free()
            score = largest_tf_fast(adj, time_limit=deadline - 5.0)
            
            if score < best_score:
                best_score = score
                best_adj = adj[:]
                print(f"Restart {restarts}: New best {best_score}")
            
            # SA on this graph
            sa_time = min(15.0, deadline - 5.0 - time.time())
            if sa_time > 3.0:
                score = sa_search(adj, score, deadline - 5.0)
                if score < best_score:
                    best_score = score
                    best_adj = adj[:]
                    print(f"  SA improvement: {best_score}")
            
            if time.time() > deadline - 10.0:
                break

    if best_adj is None:
        best_adj = build_2c9()

    edges = get_edge_list(best_adj)
    return edges
