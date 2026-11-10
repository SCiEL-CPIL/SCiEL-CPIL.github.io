import time
import random

def construct():
    """
    Returns edges for a triangle-free graph on 12 vertices with min degree >= 3
    that minimizes the number of distinct simple cycle lengths.

    Strategy: Try to find a graph with raw value 1 (all cycles same length),
    falling back to two disjoint K_{3,3} components (raw value 2) if not found.
    
    Primary approach: Two disjoint K_{3,3} components gives cycle lengths {4, 6},
    raw value 2. This is deterministic and guaranteed valid.
    
    Secondary approach: Search for a graph where all cycles have the same length
    (e.g., only 4-cycles or only 6-cycles) to achieve raw value 1.
    """
    t_start = time.time()
    
    # Fallback: Two disjoint K_{3,3}
    def make_two_k33():
        comp1 = [
            [0, 3], [0, 4], [0, 5],
            [1, 3], [1, 4], [1, 5],
            [2, 3], [2, 4], [2, 5]
        ]
        comp2 = [
            [6, 9], [6, 10], [6, 11],
            [7, 9], [7, 10], [7, 11],
            [8, 9], [8, 10], [8, 11]
        ]
        edges = comp1 + comp2
        return [[int(e[0]), int(e[1])] for e in edges]
    
    # Try to find a graph with only 4-cycles (raw value 1)
    # A 3-regular bipartite graph on 12 vertices with girth 4 and no 6-cycles
    # This means it must be a union of 4-cycles... but 3-regular can't be just 4-cycles
    # Actually, let's try a C4-free 3-regular bipartite graph with only 6-cycles?
    # Or a graph with all cycles of length 4?
    
    # For 3-regular bipartite graphs on 12 vertices:
    # - K3,3 has cycles of length 4 and 6
    # - The utility graph is K3,3
    # - Another option: a 3-regular bipartite graph that is a "cube" Q3?
    #   Q3 has 8 vertices, not 12
    # - 12 vertices, 3-regular: total edges = 18
    #   In a bipartite graph, this means 6 edges from each partition of 6
    
    # Let's try a systematic search for a 3-regular bipartite graph on 12 vertices
    # (partitions of 6) that has only one cycle length
    
    # Actually, for a 3-regular bipartite graph, the possible cycle lengths are
    # even. The girth is at least 4. If girth = 4, there are 4-cycles.
    # Can we eliminate all 6-cycles? Or eliminate all 4-cycles (girth=6)?
    
    # A 3-regular bipartite graph with girth 6 on 12 vertices:
    # This would be a (3,6)-cage-like graph. The Hoffman-Singleton theorem and
    # related results suggest this is hard. Let's check: for girth 6, the minimum
    # number of vertices for 3-regular is 10 (the Petersen graph is not bipartite,
    # but there's a bipartite (3,6)-cage with 12 vertices? Let me think...)
    
    # The 3-regular bipartite graph with girth 6 on 12 vertices:
    # This is possible. It's the incidence graph of the Fano plane? No, that's (3,7) with 14 vertices.
    # Actually, a 3-regular bipartite graph with girth 6 on 12 vertices is the
    # incidence graph of a (6,3)-bipartite structure. Let me construct one.
    
    # Let's try: vertices 0-5 in left, 6-11 in right
    # Each left vertex connects to 3 right vertices, and vice versa
    # Girth 6 means no 4-cycles: for any two left vertices, they share at most 1 common neighbor
    
    # This is equivalent to a 3-regular bipartite graph whose adjacency matrix
    # has no 4-cycles, i.e., any two columns have at most one 1 in the same row
    
    # Let me try to construct such a graph:
    # Left: 0,1,2,3,4,5
    # Right: 6,7,8,9,10,11
    
    # Try:
    # 0: 6, 7, 8
    # 1: 6, 9, 10
    # 2: 7, 9, 11
    # 3: 8, 10, 11
    # 4: ?
    # 5: ?
    
    # Check: 0 and 1 share 6. 0 and 2 share 7. 0 and 3 share 8.
    # 1 and 2 share 9. 1 and 3 share 10. 2 and 3 share 11.
    # Good so far. Now 4 and 5 need to connect to 3 vertices each from right,
    # and each right vertex needs degree 3.
    
    # Right degrees so far: 6:2(0,1), 7:2(0,2), 8:2(0,3), 9:2(1,2), 10:2(1,3), 11:2(2,3)
    # Need one more for each. So 4 and 5 each connect to 3 right vertices, and
    # the right vertices get their 3rd connection.
    
    # 4: 6, 7, 9
    # 5: 8, 10, 11
    
    # Check 4-cycles: 4 and 0 share 6,7. Two common neighbors! 4-cycle 0-6-4-7-0? No,
    # that's not a valid cycle representation. Actually, a 4-cycle exists if two left
    # vertices share two common right neighbors. 0 and 4 share 6 and 7. So there's a
    # 4-cycle: 0-6-4-7-0. This violates girth 6.
    
    # Let me try a different approach. Use known 3-regular bipartite graphs with girth 6.
    # The (3,6)-bipartite cage is the incidence graph of the projective plane of order 2
    # but that has 14 vertices. For 12 vertices, we need a specific construction.
    
    # Actually, I recall that the 3-regular bipartite graph on 12 vertices with girth 6
    # is the complement of the perfect matching in K6,6? No, that's not 3-regular.
    
    # Let me just do a randomized search for a 3-regular bipartite graph on 12 vertices
    # (6+6) with girth at least 6 (no 4-cycles). Then check what cycle lengths it has.
    
    def has_4_cycle(adj, left_size):
        """Check if a bipartite graph has a 4-cycle."""
        n_right = 12 - left_size
        for i in range(left_size):
            for j in range(i+1, left_size):
                common = set(adj[i]) & set(adj[j])
                if len(common) >= 2:
                    return True
        return False
    
    def count_cycle_lengths(edges, max_cycles=500):
        """Enumerate simple cycles and return set of cycle lengths."""
        from collections import defaultdict
        
        n = 12
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cycle_lengths = set()
        
        def dfs(start, current, visited, path):
            nonlocal cycle_lengths
            for nb in adj[current]:
                if nb == start and len(path) >= 3:
                    cycle_lengths.add(len(path))
                elif not visited[nb] and nb > start:
                    # Only count cycles where start is the smallest vertex
                    if all(nb > s for s in path[:-1]):
                        visited[nb] = True
                        path.append(nb)
                        dfs(start, nb, visited, path)
                        path.pop()
                        visited[nb] = False
        
        for start in range(n):
            visited = set()
            visited.add(start)
            dfs(start, start, visited, [start])
        
        return cycle_lengths
    
    # Try randomized search for a graph with raw value 1
    random.seed(42)
    deadline = t_start + 45.0  # Leave 15s margin
    
    best_edges = None
    best_raw = 2  # Start with K3,3+K3,3 raw value
    
    # Fallback construction
    fallback_edges = make_two_k33()
    
    # Try to find a graph with only one cycle length
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate random 3-regular bipartite graph on 6+6 vertices
        left = list(range(6))
        right = [6+i for i in range(6)]
        
        edges = []
        valid = True
        
        # Randomly pair up connections
        left_remaining = {i: 3 for i in left}
        right_remaining = {i: 3 for i in right}
        
        adj = {i: set() for i in range(12)}
        
        for i in left:
            candidates = [j for j in right if left_remaining[i] > 0 and right_remaining[j] > 0 and j not in adj[i]]
            if len(candidates) < left_remaining[i]:
                valid = False
                break
            chosen = random.sample(candidates, left_remaining[i])
            for j in chosen:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                left_remaining[i] -= 1
                right_remaining[j] -= 1
        
        if not valid:
            continue
        
        # Check 4-cycles
        if has_4_cycle(adj, 6):
            continue
        
        # Check minimum degree (should be 3 by construction)
        if all(len(adj[i]) >= 3 for i in range(12)):
            # Count cycle lengths
            cycle_lens = count_cycle_lengths(edges, max_cycles=500)
            if len(cycle_lens) == 1:
                best_edges = edges
                best_raw = 1
                break
            elif len(cycle_lens) < best_raw:
                best_edges = edges
                best_raw = len(cycle_lens)
    
    if best_edges is not None and best_raw == 1:
        final_edges = [[int(e[0]), int(e[1])] for e in best_edges]
        return final_edges
    
    # Use fallback
    return fallback_edges
