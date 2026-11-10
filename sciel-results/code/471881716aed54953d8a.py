import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    
    # Helper to check validity and get cycle lengths
    def analyze(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) < 3:
                return None, None
        
        # Check triangle-free (bipartite check is stronger, but let's just check triangles)
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return None, None
        
        # Find all simple cycle lengths
        # Since n=12, we can do DFS from each node, but it's expensive.
        # Optimization: Only need to know *distinct* lengths.
        # We can use a bitmask or just a set.
        # To speed up, we can limit cycle length to n.
        
        cycle_lengths = set()
        
        def find_cycles(start):
            # DFS to find cycles starting and ending at start
            # Only visit nodes > start to avoid duplicates? No, simple cycles can be found by fixing start.
            # Standard DFS for simple cycles.
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 3: # Triangle is len 3, but we checked triangles. So >= 4.
                        cycle_lengths.add(len(path))
                    elif neighbor not in visited and neighbor > start:
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}))
                        
        for v in range(n):
            find_cycles(v)
            
        if not cycle_lengths:
            return None, None
            
        return edges, cycle_lengths

    # Initial State: Two disjoint K3,3
    # K3,3 on 0-2 and 3-5
    # K3,3 on 6-8 and 9-11
    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([offset + i, offset + 3 + j])
        return edges
        
    initial_edges = make_k33(0) + make_k33(6)
    best_edges, best_lengths = analyze(initial_edges)
    best_score = len(best_lengths) if best_lengths else 999
    
    # Bipartition: A = {0,1,2,6,7,8}, B = {3,4,5,9,10,11}
    # Actually, in K3,3(0-5), A={0,1,2}, B={3,4,5}.
    # In K3,3(6-11), A={6,7,8}, B={9,10,11}.
    # Union A = {0,1,2,6,7,8}, B = {3,4,5,9,10,11}.
    
    # To perform a 2-switch (edge swap) that preserves 3-regularity and bipartiteness:
    # Pick two edges (u1, v1) and (u2, v2) where u1,u2 in A and v1,v2 in B.
    # Remove (u1, v1) and (u2, v2). Add (u1, v2) and (u2, v1).
    # This preserves degrees and bipartiteness.
    
    # We need to ensure we don't create duplicate edges or loops (loops impossible in bipartite switch if u!=v).
    # We must ensure (u1, v2) and (u2, v1) are not already in the graph.
    
    edge_set = set((min(u,v), max(u,v)) for u,v in initial_edges)
    
    # Precompute neighbors for fast lookup? 
    # adj is updated dynamically.
    
    current_edges = [list(e) for e in initial_edges]
    current_adj = defaultdict(set)
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
        
    random.seed(42)
    
    # Local Search
    # We want to minimize len(distinct_cycle_lengths).
    # Ideally 1.
    
    # To speed up cycle detection, we can cache or just rely on the fact that n=12 is small.
    # The DFS above is $O(2^n)$ worst case, but for sparse graphs it's okay.
    # However, doing it for every swap (thousands of times) might be slow.
    # Optimization: Only check if swap breaks a 6-cycle?
    # Simpler: Just run analyze() but it might be slow.
    
    # Let's optimize analyze for speed.
    # Instead of full DFS, we can count cycles of specific lengths?
    # Or just use the fact that if we start with {4,6}, we only care if we lose 6 or gain others.
    # If we gain 5, 7, 8..., score increases or stays same (if 4 is still there).
    # If we lose 6 and keep 4, score becomes 1.
    # If we lose 4 and keep 6, score becomes 1.
    # If we lose both, score 0 (impossible, must have cycle).
    
    # So, we just need to check if the set of lengths is {4} or {6} or {4,6} or others.
    
    # Fast check:
    # Count 4-cycles and 6-cycles.
    # If count_4 > 0 and count_6 == 0 and no other cycles?
    # Detecting "no other cycles" is hard without full enumeration.
    # But for 12 vertices, full enumeration is likely fast enough if optimized in C (NumPy?) 
    # Or just PyPy speed? No, standard Python.
    
    # Let's try to make the cycle detection faster.
    # Use bitmasks for visited nodes?
    
    def fast_analyze(adj):
        n = 12
        cycle_lengths = set()
        
        # Precompute adjacency masks
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in adj[u]:
                m |= (1 << v)
            masks[u] = m
            
        def dfs(start, node, path_len, visited_mask):
            # Pruning: if path_len > 10, we can't form a cycle <= 12 that is simple?
            # Max cycle length is 12.
            if path_len > 12:
                return
            
            # Check if we can close cycle
            if node == start:
                if path_len >= 4: # Triangle-free already checked in main, but let's be safe
                    cycle_lengths.add(path_len)
                return
                
            # If we have visited all nodes, we can't go further
            if visited_mask == (1 << n) - 1:
                return

            neighbors = masks[node]
            # Iterate neighbors
            # We only allow moving to unvisited nodes, OR to start if path_len >= 4
            # To avoid duplicates, we can restrict direction? 
            # No, standard DFS with visited mask works.
            
            # Optimization: Only explore neighbors that are not visited, except start.
            # Also, to reduce symmetry, we can enforce that the smallest node in the cycle is 'start'.
            # But 'start' is fixed in outer loop.
            
            cand = neighbors & ~visited_mask
            # Also check start
            if (neighbors & (1 << start)) and path_len >= 4:
                cycle_lengths.add(path_len)
                
            while cand:
                lsb = cand & -cand
                v = lsb.bit_length() - 1
                cand ^= lsb
                dfs(start, v, path_len + 1, visited_mask | (1 << v))
                
        for s in range(n):
            # Only start DFS from s if s is the smallest index in any cycle found starting from s?
            # This is tricky. Standard way: 
            # For each s, find cycles where s is the minimum vertex.
            # This ensures each cycle is found exactly once.
            dfs(s, s, 1, 1 << s)
            
        return cycle_lengths

    # Update analyze to use fast_analyze
    def get_score(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degrees
        for i in range(n):
            if len(adj[i]) != 3:
                return 999, None
                
        # Check triangles
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return 999, None
        
        cl = fast_analyze(adj)
        if not cl:
            return 999, None
        return len(cl), cl

    # Re-evaluate initial
    score, lengths = get_score(initial_edges)
    best_score = score
    best_edges = [list(e) for e in initial_edges]
    
    # Search
    # We represent the graph by the set of edges.
    # For swaps, we need to pick two edges (u1, v1), (u2, v2) from A-B.
    # Let's maintain list of A-nodes and B-nodes.
    A_nodes = [0, 1, 2, 6, 7, 8]
    B_nodes = [3, 4, 5, 9, 10, 11]
    
    # Current adjacency
    curr_adj = defaultdict(set)
    for u, v in best_edges:
        curr_adj[u].add(v)
        curr_adj[v].add(u)
        
    # For efficiency, keep a set of edges for O(1) lookup
    curr_edge_set = set((min(u,v), max(u,v)) for u,v in best_edges)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Pick random u1, u2 from A, v1, v2 from B
        u1 = random.choice(A_nodes)
        u2 = random.choice(A_nodes)
        if u1 == u2:
            continue
        v1 = random.choice(B_nodes)
        v2 = random.choice(B_nodes)
        if v1 == v2:
            continue
            
        # Check if (u1, v1) and (u2, v2) are edges
        e1 = (min(u1, v1), max(u1, v1))
        e2 = (min(u2, v2), max(u2, v2))
        if e1 not in curr_edge_set or e2 not in curr_edge_set:
            continue
            
        # Check if new edges (u1, v2) and (u2, v1) are NOT edges
        ne1 = (min(u1, v2), max(u1, v2))
        ne2 = (min(u2, v1), max(u2, v1))
        if ne1 in curr_edge_set or ne2 in curr_edge_set:
            continue
            
        # Perform swap
        # Remove e1, e2
        curr_edge_set.remove(e1)
        curr_edge_set.remove(e2)
        curr_adj[u1].remove(v1)
        curr_adj[v1].remove(u1)
        curr_adj[u2].remove(v2)
        curr_adj[v2].remove(u2)
        
        # Add ne1, ne2
        curr_edge_set.add(ne1)
        curr_edge_set.add(ne2)
        curr_adj[u1].add(v2)
        curr_adj[v2].add(u1)
        curr_adj[u2].add(v1)
        curr_adj[v1].add(u2)
        
        # Evaluate
        new_edges = [list(e) for e in curr_edge_set]
        new_score, new_lengths = get_score(new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
            if best_score == 1:
                # Found optimal?
                # Verify and return
                return best_edges
                
        # Undo swap (or keep if it's better? No, we want to explore. 
        # Actually, if it's better, we should keep it as the new state? 
        # Yes, hill climbing.)
        # Wait, the code above updates curr_adj/curr_edge_set permanently.
        # If new_score >= best_score, we should undo.
        if new_score >= best_score:
             # Undo
             curr_edge_set.remove(ne1)
             curr_edge_set.remove(ne2)
             curr_adj[u1].remove(v2)
             curr_adj[v2].remove(u1)
             curr_adj[u2].remove(v1)
             curr_adj[v1].remove(u2)
             
             curr_edge_set.add(e1)
             curr_edge_set.add(e2)
             curr_adj[u1].add(v1)
             curr_adj[v1].add(u1)
             curr_adj[u2].add(v2)
             curr_adj[v2].add(u2)
             
    return best_edges
