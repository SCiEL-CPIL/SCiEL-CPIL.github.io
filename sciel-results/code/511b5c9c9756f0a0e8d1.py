import time
import random
from collections import deque
import sys

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _compute_girth_fast(adjacency, n, early_stop=None):
    min_girth = float('inf')
    if early_stop is None:
        early_stop = 100
    
    for start in range(n):
        if min_girth <= early_stop:
            break
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= early_stop:
                            return min_girth
        if min_girth <= early_stop:
            break
    return min_girth if min_girth != float('inf') else -1

def _voltage_graph_construct(k, g):
    # Base graph is K_{k,k}
    # Here k=5
    n_base = 2 * k
    # Edges in K_{k,k}
    # Left: 0..k-1, Right: k..2k-1
    base_edges = []
    edge_index = {}
    idx = 0
    for u in range(k):
        for v in range(k, n_base):
            base_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            idx += 1
    
    # Find all simple cycles of length 4, 6, 8 in K_{k,k}
    # Represent cycle as tuple of edge indices
    cycles = {4: [], 6: [], 8: []}
    
    # Helper to get neighbor in base
    adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)

    # DFS to find cycles
    # Since K_{5,5} is small, we can brute force or use specific logic
    # For length 4: u1-v1-u2-v2-u1
    for u1 in range(k):
        for v1 in adj[u1]:
            for u2 in adj[v1]:
                if u2 == u1: continue
                for v2 in adj[u2]:
                    if v2 == v1: continue
                    if u1 in adj[v2]:
                        # Cycle u1-v1-u2-v2-u1
                        e1 = edge_index[(u1, v1)]
                        e2 = edge_index[(u2, v1)] # v1-u2
                        e3 = edge_index[(u2, v2)]
                        e4 = edge_index[(u1, v2)] # v2-u1
                        # Check uniqueness
                        # u1 < u2, v1 < v2? 
                        # To avoid duplicates, enforce u1 < u2 and v1 < v2?
                        # No, just collect all and deduplicate later?
                        # Or enforce canonical order.
                        # Let's just collect and use a set of frozensets of edge indices?
                        # No, order matters for sum? No, sum is commutative.
                        # But we need to check if sum is 0.
                        # We just need the set of edges.
                        pass

    # Better approach for K_{5,5} cycles:
    # 4-cycles: choose 2 distinct u's, 2 distinct v's.
    # There is exactly one 4-cycle for each pair of pairs?
    # No, the cycle is u1-v1-u2-v2-u1.
    # Edges: (u1,v1), (u2,v1), (u2,v2), (u1,v2).
    for u1 in range(k):
        for u2 in range(u1+1, k):
            for v1 in range(k, n_base):
                for v2 in range(v1+1, n_base):
                    e1 = edge_index[(u1, v1)]
                    e2 = edge_index[(u2, v1)]
                    e3 = edge_index[(u2, v2)]
                    e4 = edge_index[(u1, v2)]
                    cycles[4].append((e1, e2, e3, e4))
                    
    # 6-cycles: choose 3 distinct u's, 3 distinct v's.
    # In K_{3,3}, there are 6 distinct 6-cycles.
    # They correspond to the 6 ways to pair the 3 u's with the 3 v's in a cycle?
    # Actually, a 6-cycle is a Hamiltonian cycle in the induced K_{3,3}.
    # Number of such cycles is (3-1)! * 2 / 2? No, it's 6.
    # Let's list them.
    # Vertices: U={0,1,2}, V={3,4,5} (relative)
    # Cycles:
    # 0-3-1-4-2-5-0
    # 0-3-1-5-2-4-0
    # 0-3-2-4-1-5-0
    # 0-3-2-5-1-4-0
    # And reverses? No, we count undirected cycles.
    # The 6 cycles are generated by permuting the V's?
    # Let's just iterate.
    for u1 in range(k):
        for u2 in range(u1+1, k):
            for u3 in range(u2+1, k):
                us = [u1, u2, u3]
                for v1 in range(k, n_base):
                    for v2 in range(v1+1, n_base):
                        for v3 in range(v2+1, n_base):
                            vs = [v1, v2, v3]
                            # Generate all 6 Hamiltonian cycles in K_{3,3}
                            # Fix u1.
                            # Next is one of vs.
                            # Then one of remaining us.
                            # Then one of remaining vs.
                            # Then last u.
                            # Then last v.
                            # Then back to u1.
                            # There are 2 * 2 * 1 = 4?
                            # Actually, there are 6 distinct cycles.
                            # Let's just check all permutations of vs for the 2nd, 4th, 6th steps?
                            # Step 1: u1 -> v_a
                            # Step 2: v_a -> u_b (u_b != u1)
                            # Step 3: u_b -> v_c (v_c != v_a)
                            # Step 4: v_c -> u_d (u_d != u1, u_b)
                            # Step 5: u_d -> v_e (v_e != v_a, v_c)
                            # Step 6: v_e -> u1
                            
                            # To avoid duplicates, we can just pick one canonical form?
                            # Or just generate all and deduplicate by frozenset of edges?
                            # Since we only care if sum is 0, and sum is commutative,
                            # we just need the set of edges.
                            # But different cycles might use the same set of edges?
                            # No, a set of 6 edges in K_{3,3} forms a unique 6-cycle?
                            # Yes, K_{3,3} is 3-regular. A 6-cycle uses all 6 edges?
                            # No, K_{3,3} has 9 edges. A 6-cycle uses 6 edges.
                            # So different cycles use different sets of edges.
                            
                            # Let's just generate the 6 cycles.
                            # Permutations of vs:
                            for p in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                                # p is permutation of indices 0,1,2 for vs
                                # Sequence: u1 - vs[p[0]] - u? - vs[p[1]] - u? - vs[p[2]] - u1
                                # The u's must be u2, u3 in some order.
                                # If we fix the order of u's as u1, u2, u3?
                                # No, the cycle must visit all u's.
                                # Let's iterate permutations of us[1:] and vs.
                                pass
                            
                            # Simpler: Just iterate all permutations of vs and us[1:]
                            # There are 3! * 2! = 12?
                            # But each cycle is counted twice (direction).
                            # So 6 unique.
                            for pv in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                                for pu in [(0,1), (1,0)]:
                                    # u_seq = [u1, us[pu[0]], us[pu[1]]]
                                    # v_seq = [vs[pv[0]], vs[pv[1]], vs[pv[2]]]
                                    # Cycle: u1-v0-u1'-v1-u2'-v2-u1
                                    u_seq = [u1, us[pu[0]], us[pu[1]]]
                                    v_seq = [vs[pv[0]], vs[pv[1]], vs[pv[2]]]
                                    
                                    # Check if this forms a valid cycle (edges exist)
                                    # Edges:
                                    # (u1, v0), (u1', v0), (u1', v1), (u2', v1), (u2', v2), (u1, v2)
                                    # Wait, the cycle is u1-v0-u1'-v1-u2'-v2-u1
                                    # Edges:
                                    e1 = edge_index.get((u_seq[0], v_seq[0]))
                                    e2 = edge_index.get((u_seq[1], v_seq[0]))
                                    e3 = edge_index.get((u_seq[1], v_seq[1]))
                                    e4 = edge_index.get((u_seq[2], v_seq[1]))
                                    e5 = edge_index.get((u_seq[2], v_seq[2]))
                                    e6 = edge_index.get((u_seq[0], v_seq[2]))
                                    
                                    if e1 is not None and e2 is not None and e3 is not None and \
                                       e4 is not None and e5 is not None and e6 is not None:
                                        # Add to cycles
                                        # To avoid duplicates, we can just add it.
                                        # We will deduplicate later if needed, or just check all.
                                        # Since we want to check if ANY sum is 0, checking duplicates is fine.
                                        cycles[6].append((e1, e2, e3, e4, e5, e6))

    # 8-cycles are more complex.
    # For now, let's skip 8-cycles in the precomputation if it's too hard,
    # and rely on the BFS check on the actual graph?
    # No, BFS on 1000 vertices is fast.
    # But checking 8-cycles in base is faster than building graph.
    # Let's assume 4 and 6 cycles are the main bottlenecks.
    # If we pass 4 and 6, we build the graph and check girth exactly.

    start_time = time.time()
    best_n = float('inf')
    best_graph = None
    
    # Try m from 50 to 150
    for m in range(50, 151):
        if time.time() - start_time > 120:
            break
            
        for attempt in range(2000):
            if time.time() - start_time > 125:
                break
                
            # Random voltages
            voltages = [random.randint(1, m-1) for _ in range(len(base_edges))]
            
            # Check 4-cycles
            valid = True
            for cyc in cycles[4]:
                s = (voltages[cyc[0]] + voltages[cyc[1]] + voltages[cyc[2]] + voltages[cyc[3]]) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check 6-cycles
            for cyc in cycles[6]:
                s = (voltages[cyc[0]] + voltages[cyc[1]] + voltages[cyc[2]] + 
                     voltages[cyc[3]] + voltages[cyc[4]] + voltages[cyc[5]]) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If passed 4 and 6, build graph and check girth exactly
            # Build lift
            n_lift = 10 * m
            # Map (v, g) to index
            # v in 0..9, g in 0..m-1
            # index = v * m + g
            
            adjacency = [set() for _ in range(n_lift)]
            
            for u, v in base_edges:
                w_uv = voltages[edge_index[(u, v)]]
                w_vu = (-w_uv) % m
                
                for g in range(m):
                    u_idx = u * m + g
                    v_idx = v * m + (g + w_uv) % m
                    adjacency[u_idx].add(v_idx)
                    
                    # Since we iterate all edges, and add both directions?
                    # No, base_edges contains (u,v) where u < k <= v.
                    # We add u->v.
                    # We also need v->u.
                    # But (v,u) is not in base_edges list directly?
                    # We have edge_index[(v,u)] = same index.
                    # So we should only iterate u in 0..k-1, v in k..2k-1.
                    # And add both directions in the lift.
                    pass
            
            # Re-do graph construction carefully
            adjacency = [set() for _ in range(n_lift)]
            for u in range(k):
                for v in range(k, n_base):
                    w = voltages[edge_index[(u, v)]]
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx = v * m + (g + w) % m
                        adjacency[u_idx].add(v_idx)
                        adjacency[v_idx].add(u_idx)
            
            # Check girth
            # Use BFS from a few vertices? Or all?
            # n_lift is ~1000. BFS from all is 1000 * 1000 = 1e6 ops. Fast.
            girth = _compute_girth_fast([list(s) for s in adjacency], n_lift, early_stop=g)
            
            if girth >= g:
                # Find connected components
                visited = [False] * n_lift
                components = []
                for i in range(n_lift):
                    if not visited[i]:
                        comp = []
                        queue = deque([i])
                        visited[i] = True
                        while queue:
                            u = queue.popleft()
                            comp.append(u)
                            for v in adjacency[u]:
                                if not visited[v]:
                                    visited[v] = True
                                    queue.append(v)
                        components.append(comp)
                        
                # Return smallest component
                min_comp = min(components, key=len)
                n_comp = len(min_comp)
                
                if n_comp < best_n:
                    # Relabel
                    mapping = {old: new for new, old in enumerate(min_comp)}
                    edges = []
                    for u in min_comp:
                        for v in adjacency[u]:
                            if u < v:
                                edges.append((mapping[u], mapping[v]))
                    best_n = n_comp
                    best_graph = (n_comp, edges)
                    
                # If we found a very small one, we can stop early?
                # But we want the minimum.
                # If m is increasing, we might find smaller n later?
                # No, n is proportional to m.
                # So we should search m from low to high.
                # If we find one for m=50, n=500.
                # If we find one for m=51, n=510.
                # So we should return the first valid one found for the smallest m?
                # Or keep searching for smaller m?
                # We are iterating m from 50 upwards.
                # So the first valid m gives the smallest n.
                # We can return immediately.
                return (n_comp, edges)
                
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # Try voltage graph
        res = _voltage_graph_construct(k, g)
        if res is not None:
            return res
            
        # Fallback to LUW
        return _luw_construct(k, g)
        
    return _luw_construct(k, g)
