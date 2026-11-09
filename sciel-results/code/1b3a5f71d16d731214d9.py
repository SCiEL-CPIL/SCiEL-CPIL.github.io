"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Use bitset representation for fast validity checks (Python ints for 20-bit adjacency)
- Precompute and incrementally update common-neighbor matrix
- Multi-restart local search with diverse swap moves (1-for-1, 1-for-2, 2-for-2)
- Systematic final polish with exhaustive single-edge and 2-edge swap searches
- Return the best graph found within the time budget.
"""

import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2
    
    def valid_add(adj, cn, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free using precomputed cn matrix."""
        if adj[u] & (1 << v):
            return False
        # 1. u and v must have at most 2 common neighbors
        if cn[u][v] > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][v] > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][u] > 1:
                return False
        return True
    
    def add_edge(adj, cn, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update common neighbor matrix
        for w in range(n):
            if w != u and w != v:
                cn[u][w] = popcount(adj[u] & adj[w])
                cn[w][u] = cn[u][w]
                cn[v][w] = popcount(adj[v] & adj[w])
                cn[w][v] = cn[v][w]
        cn[u][v] = popcount(adj[u] & adj[v])
        cn[v][u] = cn[u][v]
    
    def remove_edge(adj, cn, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Update common neighbor matrix
        for w in range(n):
            if w != u and w != v:
                cn[u][w] = popcount(adj[u] & adj[w])
                cn[w][u] = cn[u][w]
                cn[v][w] = popcount(adj[v] & adj[w])
                cn[w][v] = cn[v][w]
        cn[u][v] = popcount(adj[u] & adj[v])
        cn[v][u] = cn[u][v]
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges
    
    def get_non_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append((u, v))
        return edges
    
    def build_cn(adj):
        """Build common neighbor matrix from adjacency."""
        cn = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn[a][b] = c
                cn[b][a] = c
        return cn
    
    def greedy_fill(adj, cn):
        """Greedily add all possible edges in random order."""
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add(adj, cn, u, v):
                add_edge(adj, cn, u, v)
    
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    adj[base + a] |= (1 << (base + b))
                    adj[base + b] |= (1 << (base + a))
        cn = build_cn(adj)
        return adj, cn
    
    def systematic_polish(adj, cn, time_limit):
        """Exhaustive single-edge additions then 2-edge swaps."""
        edges_count = count_edges(adj)
        
        # Phase 1: Try all possible single edge additions
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(adj)
            for u, v in non_edges:
                if time.time() - start_time > time_limit:
                    return edges_count
                if valid_add(adj, cn, u, v):
                    add_edge(adj, cn, u, v)
                    edges_count += 1
                    changed = True
            if time.time() - start_time > time_limit:
                return edges_count
        
        # Phase 2: Try 2-edge swaps (remove one, add one)
        for _ in range(10):
            if time.time() - start_time > time_limit:
                return edges_count
            improved = False
            edges = get_edges(adj)
            non_edges = get_non_edges(adj)
            for u1, v1 in edges:
                if time.time() - start_time > time_limit:
                    return edges_count
                for u2, v2 in non_edges:
                    if time.time() - start_time > time_limit:
                        return edges_count
                    remove_edge(adj, cn, u1, v1)
                    if valid_add(adj, cn, u2, v2):
                        add_edge(adj, cn, u2, v2)
                        # Try to add more edges after the swap
                        save_adj = adj[:]
                        save_cn = [row[:] for row in cn]
                        save_count = edges_count
                        greedy_fill(save_adj, save_cn)
                        new_count = count_edges(save_adj)
                        if new_count > edges_count:
                            adj = save_adj
                            cn = save_cn
                            edges_count = new_count
                            improved = True
                            break
                        else:
                            # Revert
                            for i in range(n):
                                cn[i] = save_cn[i][:]
                            remove_edge(adj, cn, u2, v2)
                            add_edge(adj, cn, u1, v1)
                    else:
                        add_edge(adj, cn, u1, v1)
                if improved:
                    break
            if not improved:
                break
        
        return edges_count
    
    best_edges = 0
    best_adj = None
    best_cn = None
    
    # First, establish baseline
    adj0, cn0 = make_k4_union()
    greedy_fill(adj0, cn0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]
    best_cn = [row[:] for row in cn0]
    
    # Multi-restart simulated annealing
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Different starting points
        if r == 0:
            adj, cn = make_k4_union()
        elif r < 5:
            # Start from best found, then perturb more aggressively
            adj = best_adj[:]
            cn = [row[:] for row in best_cn]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:5]:
                remove_edge(adj, cn, u, v)
        else:
            # Start from best found, then perturb mildly
            adj = best_adj[:]
            cn = [row[:] for row in best_cn]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:2]:
                remove_edge(adj, cn, u, v)
        
        greedy_fill(adj, cn)
        current_edges = count_edges(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            best_cn = [row[:] for row in cn]
        
        # Simulated annealing
        T = 0.8
        T_min = 0.02
        cooling = 0.993
        
        while T > T_min and (time.time() - start_time < time_limit):
            move_type = random.random()
            
            if move_type < 0.30:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(adj, cn, u, v):
                    add_edge(adj, cn, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_cn = [row[:] for row in cn]
            
            elif move_type < 0.45:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.5 / T):
                    remove_edge(adj, cn, u, v)
                    current_edges -= 1
            
            elif move_type < 0.65:
                # 1-for-1 swap: remove one edge, add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                remove_edge(adj, cn, u1, v1)
                if valid_add(adj, cn, u2, v2):
                    add_edge(adj, cn, u2, v2)
                    # Accept with some probability
                    if random.random() < 0.7:
                        current_edges = current_edges  # same count
                    else:
                        remove_edge(adj, cn, u2, v2)
                        add_edge(adj, cn, u1, v1)
                else:
                    add_edge(adj, cn, u1, v1)
            
            elif move_type < 0.80:
                # 1-for-2 swap: remove one edge, try to add two edges
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                remove_edge(adj, cn, u1, v1)
                added = 0
                added_edges = []
                for u2, v2 in random.sample(non_edges, min(3, len(non_edges))):
                    if valid_add(adj, cn, u2, v2):
                        add_edge(adj, cn, u2, v2)
                        added_edges.append((u2, v2))
                        added += 1
                        if added == 2:
                            break
                if added >= 1:
                    current_edges += added - 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_cn = [row[:] for row in cn]
                else:
                    add_edge(adj, cn, u1, v1)
            
            else:
                # 2-for-2 swap: remove two edges, add two edges
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    T *= cooling
                    continue
                (u1, v1), (u2, v2) = random.sample(edges, 2)
                (u3, v3), (u4, v4) = random.sample(non_edges, 2)
                remove_edge(adj, cn, u1, v1)
                remove_edge(adj, cn, u2, v2)
                added = 0
                added_edges = []
                for u, v in [(u3, v3), (u4, v4)]:
                    if valid_add(adj, cn, u, v):
                        add_edge(adj, cn, u, v)
                        added_edges.append((u, v))
                        added += 1
                if added >= 1:
                    current_edges += added - 2
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_cn = [row[:] for row in cn]
                else:
                    add_edge(adj, cn, u1, v1)
                    add_edge(adj, cn, u2, v2)
            
            T *= cooling
    
    # Final polish on the best solution
    if best_adj is not None:
        adj = best_adj[:]
        cn = [row[:] for row in best_cn]
        final_edges = systematic_polish(adj, cn, time_limit)
        if final_edges > best_edges:
            best_edges = final_edges
            best_adj = adj
            best_cn = cn
    
    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END
