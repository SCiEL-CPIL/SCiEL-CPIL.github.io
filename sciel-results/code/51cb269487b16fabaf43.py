import time
import random
import math
import numpy as np

def count_c4_from_bitsets(adj):
    """
    Counts the number of C4 subgraphs in a graph represented by bitsets.
    adj[i] is an integer bitmask of neighbors of i.
    """
    n = len(adj)
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                # Count common neighbors of i and j
                common = adj[i] & adj[j]
                c = common.bit_count()
                c4_count += c * (c - 1) // 2
    return c4_count

def c4s_created_by_adding(adj, u, v):
    """Count C4s created by adding edge (u, v) to graph with adjacency bitsets."""
    nu = adj[u]
    nv = adj[v]
    count = 0
    w_mask = nu
    while w_mask:
        lowbit = w_mask & (-w_mask)
        w = lowbit.bit_length() - 1
        w_mask ^= lowbit
        common = adj[w] & nv
        count += common.bit_count()
    return count

def c4s_lost_by_removing(adj, u, v):
    """Count C4s containing edge (u, v) in graph with adjacency bitsets."""
    nu = adj[u] & ~(1 << v)
    nv = adj[v] & ~(1 << u)
    count = 0
    w_mask = nu
    while w_mask:
        lowbit = w_mask & (-w_mask)
        w = lowbit.bit_length() - 1
        w_mask ^= lowbit
        common = adj[w] & nv
        count += common.bit_count()
    return count

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # 1. Construct Polarity Graph of PG(2, 5)
    # Points in PG(2, 5) can be represented as (x, y, z) not all 0, modulo scaling.
    # We use a canonical form: z=1 (25 points), z=0, y=1 (5 points), (0,0,1) is covered by z=1? No.
    # Standard enumeration:
    # Type 1: (1, y, z) for y,z in 0..4 -> 25 points
    # Type 2: (0, 1, z) for z in 0..4 -> 5 points
    # Type 3: (0, 0, 1) -> 1 point
    # Total 31 points.
    
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = [0] * n
    edge_set = set()
    
    # Polarity condition: dot product is 0 mod 5
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    
    # 2. Add edges to reach m=91
    # The polarity graph of PG(2,5) has 90 edges and is C4-free.
    # We need to add 1 edge. We choose the one that creates the fewest C4s.
    
    if current_edges < m:
        edges_to_add = m - current_edges
        
        best_edge = None
        min_cost = float('inf')
        
        # Find the best single edge to add
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    cost = c4s_created_by_adding(adj, i, j)
                    if cost < min_cost:
                        min_cost = cost
                        best_edge = (i, j)
        
        if best_edge:
            u, v = best_edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add(best_edge)
            current_edges += 1
            
        # If we still need more edges (unlikely for this specific construction, but good for robustness)
        while current_edges < m and time.time() < deadline:
            best_edge_iter = None
            min_cost_iter = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in edge_set:
                        cost = c4s_created_by_adding(adj, i, j)
                        if cost < min_cost_iter:
                            min_cost_iter = cost
                            best_edge_iter = (i, j)
            if best_edge_iter:
                u, v = best_edge_iter
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add(best_edge_iter)
                current_edges += 1
            else:
                break

    # Calculate initial C4 count
    current_c4 = count_c4_from_bitsets(adj)
    best_c4 = current_c4
    best_adj = adj[:]
    best_edge_set = set(edge_set)

    # 3. Local Search: 1-edge swaps
    # We want to minimize C4 count.
    # Swap (u1, v1) -> (u2, v2).
    # Delta = C4s_created(u2, v2) - C4s_lost(u1, v1)
    
    def local_search_1swap(adj, edge_set, current_c4, max_time, deadline):
        """Perform 1-edge swap local search. Returns (adj, edge_set, current_c4, best_c4, best_adj, best_edge_set)."""
        best_c4 = current_c4
        best_adj = adj[:]
        best_edge_set = set(edge_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            edges_list = list(edge_set)
            non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
            
            for e_rm in edges_list:
                if time.time() > deadline:
                    break
                u1, v1 = e_rm
                lost = c4s_lost_by_removing(adj, u1, v1)
                
                for e_add in non_edges_list:
                    u2, v2 = e_add
                    gained = c4s_created_by_adding(adj, u2, v2)
                    delta = gained - lost
                    
                    if delta < 0:
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        
                        current_c4 += delta
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_adj = adj[:]
                            best_edge_set = set(edge_set)
                        
                        improved = True
                        break
                
                if improved:
                    break
        
        return adj, edge_set, current_c4, best_c4, best_adj, best_edge_set

    adj, edge_set, current_c4, best_c4, best_adj, best_edge_set = local_search_1swap(adj, edge_set, current_c4, deadline, deadline)

    # 4. Simulated Annealing (if time permits)
    if time.time() < deadline - 15:
        adj = best_adj[:]
        edge_set = set(best_edge_set)
        current_c4 = best_c4
        
        T = 8.0
        T_min = 0.005
        cooling_rate = 0.9995
        steps_per_temp = 50
        
        while T > T_min and time.time() < deadline - 5:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge to remove
                if not edge_set:
                    break
                e_rm = random.choice(list(edge_set))
                u1, v1 = e_rm
                
                # Pick random non-edge to add
                while True:
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (u2, v2) not in edge_set:
                        break
                
                lost = c4s_lost_by_removing(adj, u1, v1)
                gained = c4s_created_by_adding(adj, u2, v2)
                delta = gained - lost
                
                # Apply temporarily
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                new_c4 = current_c4 + delta
                
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_c4 = new_c4
                    edge_set.remove(e_rm)
                    edge_set.add((u2, v2))
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj = adj[:]
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
            
            T *= cooling_rate
        
        # Local search after SA
        if time.time() < deadline - 2:
            adj, edge_set, current_c4, best_c4, best_adj, best_edge_set = local_search_1swap(adj, edge_set, current_c4, deadline, deadline)

    # 5. Final Local Search Pass
    adj = best_adj[:]
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    
    if time.time() < deadline - 1:
        adj, edge_set, current_c4, best_c4, best_adj, best_edge_set = local_search_1swap(adj, edge_set, current_c4, deadline, deadline)

    # Return the best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        # Fallback
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
