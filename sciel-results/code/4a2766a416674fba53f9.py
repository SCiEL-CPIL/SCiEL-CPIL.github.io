import time
from itertools import combinations, product
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 12
    min_deg = 3
    
    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_cycle_of_length(adj, n, L):
        """Check if there is a simple cycle of length L."""
        # For small L, we can use combinatorial checks or DFS with depth limit
        # This is a basic check. For L=4, 6, 8 it might be slow if done naively for every candidate.
        # We will use this selectively.
        pass

    def count_distinct_cycle_lengths_fast(adj, n, target_lengths=[4, 6, 8, 10, 12]):
        """
        Try to find cycle lengths.
        We prioritize finding 4 and 6.
        If we find a 6, we know score >= 2 (unless we also find 4 and no others? No, distinct count).
        We want to minimize the set of lengths.
        """
        lengths_found = set()
        
        # Check for 4-cycles
        # A 4-cycle is a pair of vertices in A with 2 common neighbors in B
        # Or generally: u-v-w-x-u
        # Bipartite check:
        # If graph is bipartite, all cycles are even.
        # We assume we are searching bipartite graphs.
        
        # Let's just do a BFS/DFS to find cycles up to length 12
        # But this is what was slow before.
        
        # Alternative: Use the property that if we find a 6-cycle, we can stop if we only care about "is it 1?".
        # If we find a 4-cycle and a 6-cycle, score is at least 2.
        # If we find ONLY 4-cycles, score is 1.
        
        # Efficient check for 6-cycles in bipartite graph:
        # A 6-cycle exists if there are 3 vertices in A and 3 in B such that they form a "hexagon".
        # This is equivalent to checking if the 3x3 submatrix has a permanent > 0? No.
        
        # Let's stick to a robust but optimized DFS for cycle detection, 
        # but prune aggressively.
        
        # Actually, let's just try to find a graph with NO 6-cycles first.
        # If a bipartite graph has no 6-cycles, does it have 8-cycles?
        # For 12 vertices, it's possible to have only 4-cycles.
        
        # Heuristic: 
        # 1. Check if 4-cycle exists.
        # 2. Check if 6-cycle exists.
        # 3. If no 6-cycle, check 8-cycle.
        
        # Check 4-cycle:
        # For each pair of vertices u, v, count common neighbors.
        # If any pair has >= 2 common neighbors, 4-cycle exists.
        has_4 = False
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    has_4 = True
                    break
            if has_4:
                break
        if has_4:
            lengths_found.add(4)
            
        # Check 6-cycle:
        # A 6-cycle is u-v-w-x-y-z-u.
        # In bipartite graph, u,w,y in A and v,x,z in B.
        # We can check by looking at paths of length 3.
        # If there is a path of length 3 from u to v, and u-v is an edge, that's a 4-cycle (already counted).
        # A 6-cycle involves 3 vertices in A and 3 in B.
        
        # Simple DFS for 6-cycles:
        # Pick a start vertex, DFS depth 5, check if last connects to start.
        # To avoid duplicates, only start from smallest vertex in the cycle?
        # This is complex.
        
        # Let's try a different tactic: 
        # Random search for 3-regular bipartite graphs.
        # For each, count distinct cycle lengths using a fast enumerator.
        
        return lengths_found

    def find_cycles(adj, n):
        """
        Find all simple cycles. 
        Optimized for small n=12.
        """
        cycles = set()
        
        def dfs(start, current, path, visited):
            if len(path) > 12: # Max cycle length
                return
            if current == start and len(path) > 2:
                # Normalize: tuple of sorted vertices? 
                # No, cycle is a sequence. 
                # To deduplicate, we can store the frozenset of edges? 
                # Or the lexicographically smallest rotation.
                # For counting lengths, we just need the length.
                # But we need to know if it's the SAME cycle to avoid double counting?
                # No, we just need the SET of lengths.
                # So if we find a cycle of length 4, we add 4.
                # We don't need to deduplicate cycles of the same length.
                # We just need to know if length L exists.
                
                # So we can just return the length.
                return len(path)
            
            for neighbor in adj[current]:
                if neighbor == start and len(path) > 2:
                    return len(path)
                if neighbor not in visited:
                    # Pruning: only visit neighbors > start to reduce symmetry?
                    # This works if we iterate start from 0 to n-1.
                    # But we must ensure we find ALL cycles.
                    # If we restrict to neighbor > start, we might miss cycles where start is not the smallest?
                    # No, if we iterate all starts, and only go to > start, 
                    # we will find each cycle exactly once (starting from its min vertex).
                    if neighbor > start:
                        res = dfs(start, neighbor, path + [neighbor], visited | {neighbor})
                        if res:
                            return res
            return None
        
        # We just want the set of lengths.
        lengths = set()
        
        # To be safe and fast, let's just do a full enumeration but stop early if we have enough lengths?
        # No, we want MINIMAL lengths.
        
        # Let's just enumerate all cycles and collect lengths.
        # With n=12, this is feasible if optimized.
        
        for start in range(n):
            # DFS
            stack = [(start, start, [start], {start})]
            while stack:
                s, curr, path, visited = stack.pop()
                if curr == s and len(path) > 2:
                    lengths.add(len(path))
                    continue
                
                for nb in adj[curr]:
                    if nb == s and len(path) > 2:
                        lengths.add(len(path))
                    elif nb > s and nb not in visited:
                        stack.append((s, nb, path + [nb], visited | {nb}))
                        
        return lengths

    best_edges = None
    best_score = float('inf')
    
    # Baseline: Two K_{3,3}
    edges = []
    for i in range(3):
        for j in range(3):
            edges.append([i, j+3])
            edges.append([i+6, j+9])
    adj = build_adj(edges, n)
    lengths = find_cycles(adj, n)
    best_score = len(lengths)
    best_edges = edges.copy()
    
    import random
    
    A = list(range(6))
    B = list(range(6, 12))
    
    # Random search
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Generate random 3-regular bipartite graph
        # Use configuration model
        adj_a = {v: [] for v in A}
        adj_b = {v: [] for v in B}
        
        valid = True
        for a in A:
            neighbors = random.sample(B, 3)
            adj_a[a] = neighbors
            for b in neighbors:
                adj_b[b].append(a)
                
        for b in B:
            if len(adj_b[b]) != 3:
                valid = False
                break
                
        if not valid:
            continue
            
        edges = []
        for a in A:
            for b in adj_a[a]:
                edges.append([a, b])
                
        adj = build_adj(edges, n)
        
        # Quick check: is it triangle free? (Bipartite implies yes)
        # Check degree
        if any(len(adj[i]) < 3 for i in range(n)):
            continue
            
        # Count lengths
        lengths = find_cycles(adj, n)
        score = len(lengths)
        
        if score < best_score:
            best_score = score
            best_edges = edges.copy()
            if best_score == 1:
                break
                
    if best_edges is None:
        best_edges = []
        for i in range(3):
            for j in range(3):
                best_edges.append([i, j+3])
                best_edges.append([i+6, j+9])
                
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return sorted(result)
