import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s kill
    n = 40

    # Precompute all vertex pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    n_pairs = len(all_pairs)
    pair_to_idx = {(u, v): i for i, (u, v) in enumerate(all_pairs)}
    idx_to_pair = all_pairs  # just use index

    # Adjacency as bitmasks for fast C6 detection
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 through edge (u,v) is: u-a-b-c-d-v-u
        So we need: a in N(u)\\{v}, d in N(v)\\{u}, a!=d,
        b in N(a)\\{u,v,d}, c in N(d)\\{u,v,a,b}, and b-c edge exists.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        a_mask = nu
        while a_mask:
            lb = a_mask & (-a_mask)
            ai = lb.bit_length() - 1
            a_mask ^= lb
            
            d_mask = nv & ~lb  # d != a
            while d_mask:
                ld = d_mask & (-d_mask)
                di = ld.bit_length() - 1
                d_mask ^= ld
                
                # b in N(a) \ {u,v,d,a}
                b_mask = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~lb
                # c in N(d) \ {u,v,a,d}
                c_mask = adj[di] & ~(1 << u) & ~(1 << v) & ~lb & ~ld
                
                # Check if any b in b_mask has any c in c_mask as neighbor
                bm = b_mask
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    if adj[bi] & c_mask:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # ===== Strategy: Multi-restart simulated annealing with smart moves =====
    
    def random_initial_graph():
        """Create a random C6-free graph using greedy random insertion."""
        adj = [0] * n
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def greedy_fill(adj):
        """Greedy add all possible edges in random order."""
        adj = adj[:]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def one_step_sa(adj, temperature):
        """Perform one SA step: try to add or remove one edge."""
        adj = adj[:]
        
        # Randomly choose to add or remove
        if random.random() < 0.6:
            # Try to add a random non-edge
            non_edges = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
            if not non_edges:
                return adj
            u, v = random.choice(non_edges)
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                return adj
            else:
                # Can't add, maybe remove a random edge to create space
                return adj
        else:
            # Remove a random edge
            edges = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
            if not edges:
                return adj
            u, v = random.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return adj

    def swap_move(adj):
        """Remove one edge, then greedily try to add edges (including the removed one)."""
        adj = adj[:]
        edges = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        if not edges:
            return adj
        
        rem_u, rem_v = random.choice(edges)
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)
        
        # Greedily add back, with removed edge prioritized
        candidates = [(rem_u, rem_v)]
        for u, v in all_pairs:
            if (u, v) != (rem_u, rem_v) and not ((adj[u] >> v) & 1):
                candidates.append((u, v))
        random.shuffle(candidates[1:])
        
        for u, v in candidates:
            if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def pair_swap_move(adj):
        """Remove two edges, greedily add back."""
        adj = adj[:]
        edges = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        if len(edges) < 2:
            return adj
        
        idx1, idx2 = random.sample(range(len(edges)), 2)
        e1, e2 = edges[idx1], edges[idx2]
        adj[e1[0]] &= ~(1 << e1[1])
        adj[e1[1]] &= ~(1 << e1[0])
        adj[e2[0]] &= ~(1 << e2[1])
        adj[e2[1]] &= ~(1 << e2[0])
        
        candidates = [e1, e2]
        for u, v in all_pairs:
            if (u, v) not in (e1, e2) and not ((adj[u] >> v) & 1):
                candidates.append((u, v))
        random.shuffle(candidates[2:])
        
        for u, v in candidates:
            if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def neighborhood_polish(adj):
        """Local search: for each edge, try removing it and re-adding optimally."""
        adj = adj[:]
        edges = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        if not edges:
            return adj
        
        best_adj = adj[:]
        best_count = count_edges(adj)
        
        for rem_u, rem_v in edges:
            if time.time() > deadline - 3:
                break
            
            test_adj = adj[:]
            test_adj[rem_u] &= ~(1 << rem_v)
            test_adj[rem_v] &= ~(1 << rem_u)
            
            # Greedily add back
            candidates = [(rem_u, rem_v)]
            for u, v in all_pairs:
                if (u, v) != (rem_u, rem_v) and not ((test_adj[u] >> v) & 1):
                    candidates.append((u, v))
            random.shuffle(candidates[1:])
            
            for u, v in candidates:
                if not ((test_adj[u] >> v) & 1) and can_add_edge(test_adj, u, v):
                    test_adj[u] |= (1 << v)
                    test_adj[v] |= (1 << u)
            
            c = count_edges(test_adj)
            if c > best_count:
                best_count = c
                best_adj = test_adj[:]
            
            # Restore
            adj = best_adj[:]  # Keep trying from best
        
        return best_adj

    def run_sa_round(seed_val, time_budget):
        """Run a SA round with given seed and time budget."""
        random.seed(seed_val)
        end_time = time.time() + time_budget
        
        # Start with a random greedy graph
        adj = random_initial_graph()
        adj = greedy_fill(adj)
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        T = 30.0
        cooling_rate = 0.995
        
        while time.time() < end_time:
            # Choose move type
            r = random.random()
            if r < 0.4:
                new_adj = one_step_sa(adj, T)
            elif r < 0.7:
                new_adj = swap_move(adj)
            else:
                new_adj = pair_swap_move(adj)
            
            new_count = count_edges(new_adj)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = best_count - new_count  # positive means worse
            if new_count >= current_count:
                adj = new_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = new_adj[:]
            else:
                if T > 0.1 and random.random() < np.exp(-delta / T):
                    adj = new_adj
                    current_count = new_count
            
            T *= cooling_rate
            if T < 1.0:
                T = 1.0
        
        # Final polish
        if time.time() < end_time - 1:
            polished = neighborhood_polish(best_adj)
            pc = count_edges(polished)
            if pc > best_count:
                best_count = pc
                best_adj = polished
        
        return best_adj, best_count

    # ===== Main: Multi-round SA with different seeds and strategies =====
    
    global_best_adj = None
    global_best_count = 0
    
    # Round 0: Try the 8xK5 base structure and polish
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    
    polished_base = greedy_fill(base_adj)
    bc = count_edges(polished_base)
    if bc > global_best_count:
        global_best_count = bc
        global_best_adj = polished_base[:]
    
    # Try neighborhood polish on base
    if time.time() < deadline - 20:
        polished2 = neighborhood_polish(polished_base)
        pc2 = count_edges(polished2)
        if pc2 > global_best_count:
            global_best_count = pc2
            global_best_adj = polished2[:]
    
    if global_best_count >= 105:
        # Extract edges and return
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (global_best_adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # SA rounds with different seeds
    seeds = [42, 123, 456, 789, 101112, 202021, 303031, 505051, 606061, 707071]
    
    for si, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - 5 - time.time()
        if remaining < 3:
            break
        
        # Allocate time: more for earlier rounds
        if si < 3:
            tb = min(remaining, 8.0)
        elif si < 6:
            tb = min(remaining, 5.0)
        else:
            tb = min(remaining, 3.0)
        
        adj, cnt = run_sa_round(seed, tb)
        if cnt > global_best_count:
            global_best_count = cnt
            global_best_adj = adj[:]
        
        if global_best_count >= 105:
            break
    
    # Final polish if time permits
    if global_best_adj is not None and time.time() < deadline - 3:
        polished = neighborhood_polish(global_best_adj)
        pc = count_edges(polished)
        if pc > global_best_count:
            global_best_count = pc
            global_best_adj = polished
    
    if global_best_adj is None:
        # Fallback: return empty graph
        return []
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (global_best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
