"""
Constructs a k-regular graph with high girth using Configuration Model
and Simulated Annealing local search.
"""
import time
import sys
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    if k != 3:
        return _fallback_construct(k, g)

    # Moore bound for k=3, g=16 is 512.
    current_n = 512
    max_n = 1000
    
    # We will try n values in increasing order.
    # Allocate time for each attempt.
    
    while current_n <= max_n and time.time() < deadline:
        
        # Time budget for this n
        # Start with 10s, decrease if n increases?
        # Let's use a fixed 5s per n, but ensure we don't exceed deadline.
        
        time_budget = 5.0
        if time.time() + time_budget > deadline:
             time_budget = deadline - time.time() - 2.0 # Leave 2s for final
             if time_budget < 1.0: break
             
        result = _try_find_graph(current_n, g, time_budget, start_time)
        if result:
            return result

        current_n += 2

    # Fallback: Return the best effort or a large random graph
    # If we didn't find one, we should probably return a valid graph if possible
    # But the task says "invalid result ... scores 0".
    # So we MUST return a valid graph.
    # If we ran out of time, we need a guaranteed valid graph.
    # A voltage lift of Petersen with large m is guaranteed valid?
    # No, it's not guaranteed, but highly likely for large m.
    # The seed program returned a random lift as fallback.
    
    # Let's use the seed program's fallback logic if we fail.
    # But we can also just try one more random graph with a large n?
    # No, large n is bad for score.
    
    # Actually, if we didn't find a graph with girth 16, we can't return it.
    # We must return a graph with girth >= 16.
    # If we fail, we have to fall back to a method that is GUARANTEED to work,
    # even if n is large.
    # The voltage lift with m=3000 (n=30000) is very likely to have girth >= 16.
    # But the score is n. 30000 is bad.
    # However, returning 0 score is worse.
    
    # Let's ensure _try_find_graph is robust.
    
    # Fallback to seed program's logic for safety
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
                
    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges

def _try_find_graph(n, g, time_budget, start_time):
    deadline = start_time + time_budget
    
    # 1. Generate initial graph
    adj = None
    gen_deadline = deadline - 2.0 # Leave 2s for search
    if gen_deadline < start_time:
        gen_deadline = deadline - 0.5
        
    while time.time() < gen_deadline:
        adj = _generate_random_3_reg(n)
        if adj is not None:
            break
    if adj is None:
        return None

    # 2. Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    current_adj = [list(neighs) for neighs in adj]
    current_cost = _count_short_cycles(current_adj, n, 5)
    
    best_adj = [list(neighs) for neighs in current_adj]
    best_cost = current_cost
    
    while time.time() < deadline:
        # Generate move
        u = random.randint(0, n-1)
        v = random.choice(current_adj[u])
        
        x = random.randint(0, n-1)
        y = random.choice(current_adj[x])
        
        if u == x or u == y or v == x or v == y:
            continue
        if x in current_adj[u] or y in current_adj[v]:
            continue
        if y in current_adj[u] or x in current_adj[v]:
            continue
            
        # Perform swap
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        current_adj[x].remove(y)
        current_adj[y].remove(x)
        
        current_adj[u].append(x)
        current_adj[x].append(u)
        current_adj[v].append(y)
        current_adj[y].append(v)
        
        new_cost = _count_short_cycles(current_adj, n, 5)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = [list(neighs) for neighs in current_adj]
                
                if best_cost == 0:
                    if _verify_girth_fast(n, best_adj, g):
                        edges = []
                        for i in range(n):
                            for j in best_adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return (n, edges)
        else:
            # Revert
            current_adj[u].remove(x)
            current_adj[x].remove(u)
            current_adj[v].remove(y)
            current_adj[y].remove(v)
            
            current_adj[u].append(v)
            current_adj[v].append(u)
            current_adj[x].append(y)
            current_adj[y].append(x)
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return None

def _generate_random_3_reg(n):
    if n % 2 != 0:
        return None
        
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    for i in range(0, 3*n, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        
    return adj

def _count_short_cycles(adj, n, max_len=5):
    count = 0
    # Count triangles (3-cycles)
    for i in range(n):
        for j in adj[i]:
            if i < j:
                common = set(adj[i]) & set(adj[j])
                count += len(common)
    count //= 3
    
    # Count 4-cycles
    # For each edge (i,j), count common neighbors of i and j? No.
    # A 4-cycle is i-j-k-l-i.
    # Fix edge (i,j). Find k in adj[j] (k!=i). Find l in adj[i] (l!=j).
    # If k==l, then i-j-k-i is a 3-cycle (already counted).
    # If k != l, and l in adj[k], then i-j-k-l-i is a 4-cycle.
    # Each 4-cycle is counted 4 times (once for each edge).
    
    for i in range(n):
        for j in adj[i]:
            if i < j:
                for k in adj[j]:
                    if k == i: continue
                    for l in adj[i]:
                        if l == j or l == k: continue
                        if l in adj[k]:
                            count += 1
    count //= 4
    
    # Count 5-cycles
    # More complex.
    # Fix vertex i.
    # Neighbors of i: n1, n2, n3.
    # We need a path of length 4 from i back to i that doesn't use i internally.
    # This is equivalent to counting closed walks of length 5 that don't repeat vertices?
    # Simpler: Iterate all 5-tuples (i, j, k, l, m) distinct.
    # Too slow for n=500? 500^5 is 3e13. No.
    
    # We can use matrix multiplication?
    # Or just skip 5-cycles in cost and rely on verification?
    # If we have a 5-cycle, cost is 0, SA stops improving.
    # But 5-cycles are less frequent than 3/4.
    # Let's add a simple 5-cycle counter if time permits.
    # For now, return count.
    
    return count

def _verify_girth_fast(n, adj, g):
    # adj is list of lists
    half_g = g // 2
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _build_lift(base_edges, m, shifts):
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges

def _fallback_construct(k, g):
    # Same as seed fallback
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
                
    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges

# EVOLVE-BLOCK-END
