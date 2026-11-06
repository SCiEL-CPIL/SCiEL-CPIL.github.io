"""Sparse pancyclic graph construction on 40 vertices."""
import time
from collections import deque

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Base: a single Hamiltonian cycle on vertices 0..39
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set(frozenset(e) for e in base_edges)
    
    # Build adjacency for base cycle
    adj = [set() for _ in range(N)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all cycle lengths in base graph (just 40)
    def find_cycle_lengths(edge_list):
        """Find all cycle lengths in the graph defined by edge_list."""
        n = N
        adj_local = [set() for _ in range(n)]
        for u, v in edge_list:
            adj_local[u].add(v)
            adj_local[v].add(u)
        
        lengths = set()
        # For each pair of adjacent vertices in the cycle, try to find cycles of various lengths
        # Use BFS from each vertex to find shortest paths, then cycles
        for start in range(n):
            # BFS to find distances
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj_local[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            # Check for cycles through start
            for v in adj_local[start]:
                if dist[v] > 0:
                    # Cycle length = dist[v] + 1
                    cl = dist[v] + 1
                    if 3 <= cl <= n:
                        lengths.add(cl)
        return lengths
    
    # Actually, let me use a more direct approach: enumerate simple cycles via DFS
    # But that's expensive. Instead, use the fact that with a base cycle + chords,
    # cycles are formed by taking an arc of the base cycle and replacing it with a path through chords.
    
    # Better approach: for each chord (u,v), the cycles it creates have lengths:
    # - The two arcs of the base cycle between u and v, plus the chord
    # - But also combinations with multiple chords
    
    # Let me use a simpler greedy: for each candidate chord, compute which cycle lengths it enables
    # by checking if a cycle of that length exists in the graph with that chord added.
    
    # Precompute: for the base cycle, arc lengths between any two vertices
    # Vertex i is at position i on the cycle 0-1-2-...-39-0
    # Arc from i to j (going forward): (j - i) % N edges
    # The two arcs between i and j have lengths a = (j-i)%N and N-a
    
    def arc_lengths(u, v):
        a = (v - u) % N
        if a == 0:
            return (N, 0)
        return (a, N - a)
    
    # For a single chord (u,v), it creates cycles of length:
    # - a + 1 and N - a + 1 (where a is one arc length)
    # But also, with the base cycle, we get exactly those two cycles.
    # With multiple chords, we can get more.
    
    # Let me think differently. The key realization: if I have chords that create a "ladder" or 
    # "grid-like" structure, I can get all cycle lengths.
    
    # Alternative approach: use a known construction. A graph with a Hamiltonian cycle 
    # plus chords connecting i to i+k for various k can be pancyclic.
    
    # Let me try: base cycle + chords (0, k) for k = 2, 3, ..., m
    # This creates many cycle lengths.
    
    # Actually, the most efficient known construction for pancyclic graphs:
    # A cycle C_n plus a "star" of chords from one vertex gives O(n) edges.
    # But we can do better with a "double cycle" or "grid" structure.
    
    # Let me try a concrete approach:
    # Base cycle: 0-1-2-...-39-0
    # Add chords to create a structure where all cycle lengths 3-40 are achievable.
    
    # Greedy approach: start with base cycle, iteratively add the chord that covers the most uncovered lengths.
    
    def get_cycle_lengths_with_edges(edge_set):
        """Compute all cycle lengths present in the graph."""
        adj_local = [set() for _ in range(N)]
        for e in edge_set:
            u, v = e
            adj_local[u].add(v)
            adj_local[v].add(u)
        
        lengths = set()
        # For each vertex, do BFS to find all simple cycle lengths
        # This is expensive, so use a smarter approach for our structured graphs.
        
        # Use the fact that cycles in a cycle+chords graph can be enumerated
        # by considering paths between chord endpoints.
        
        # For now, use a simpler check: for each length L from 3 to N,
        # check if a cycle of length L exists.
        for L in range(3, N + 1):
            if _has_cycle_of_length(adj_local, L):
                lengths.add(L)
        return lengths
    
    def _has_cycle_of_length(adj, L):
        """Check if there's a simple cycle of exactly length L."""
        # Try from each vertex as start
        for start in range(N):
            if _dfs_cycle(adj, start, L, 1, {start}, adj[start]):
                return True
        return False
    
    def _dfs_cycle(adj, current, target_len, current_len, visited, neighbors):
        if current_len == target_len:
            # Check if we can return to start (but we don't track start here)
            return False  # Simplified; this won't work correctly
        if current_len > target_len:
            return False
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                if _dfs_cycle(adj, nb, target_len, current_len + 1, visited, adj[nb]):
                    return True
                visited.remove(nb)
        return False
    
    # This DFS approach is too slow. Let me use a different strategy.
    
    # Known result: A 2-connected graph with a Hamiltonian cycle and minimum degree >= 3
    # is pancyclic if it has certain properties. But let's be more direct.
    
    # Construction: Use a "cycle with chords" where chords form a second cycle.
    # Vertices 0..19 on outer cycle, 20..39 on inner cycle, connected by rungs.
    # This is a "prism-like" graph.
    
    # Actually, let me try the simplest effective construction:
    # Base cycle + chords (i, i+20) for i=0..19 (a "double cycle" / prism)
    # This gives 40 + 20 = 60 edges. Not great.
    
    # Better: base cycle + a small number of well-chosen chords.
    # 
    # Key insight from the knowledge: 58 edges (score 2.056) was achieved by
    # covering 36 lengths with 18 chords. So 18 chords + 40 base = 58 edges.
    # Score = 58 - 40 = 18. Wait, score is edge_count - 40, so 58-40=18? 
    # But the seed has 59 edges, score 1.0 means 59-40=19. So 58 edges -> score 18/19? 
    # No, the score IS edge_count - 40. Seed: 59-40=19, but score is 1.0...
    # 
    # Re-reading: "Minimize edge count minus 40" and "The evaluator measures the raw value (lower is better)"
    # "The search score is seed/value, so 1.0 equals the seed program"
    # So seed value = 19 (59-40), seed score = 19/19 = 1.0.
    # If we get 58 edges, value = 18, score = 19/18 = 1.056.
    # The knowledge says 1.947 -> 2.056, which corresponds to values 19/1.947≈9.76 and 19/2.056≈9.24.
    # That doesn't match. Let me re-read.
    #
    # "Reducing edge count from 59 to 58... Measured: 1.947 -> 2.056"
    # So 59 edges -> score 1.947, 58 edges -> score 2.056.
    # Score = seed_value / value. Seed value = 19.
    # 19/1.947 ≈ 9.76, 19/2.056 ≈ 9.24. So values are ~9.76 and ~9.24?
    # That means edge counts are ~49.76 and ~49.24, i.e., 50 and 49 edges!
    # 
    # So the improved solution has 49-50 edges, not 58! The "18 chords" must mean something else.
    # 40 (base cycle) + 9 chords = 49 edges. That makes more sense.
    
    # So the goal is to get down to ~49 edges. Let me aim for that.
    
    # With a base cycle of 40 edges, I need only 9 additional edges to cover all cycle lengths 3-40.
    # The base cycle covers length 40. Each chord can cover multiple lengths.
    
    # Let me try: chords from vertex 0 to vertices 2,3,4,5,6,7,8,9,10
    # This creates many cycle lengths.
    
    # Actually, let me just try a few constructions and validate.
    
    def validate_and_score(edges, cycles):
        """Validate the solution and return the score value (edge_count - 40)."""
        edge_set = set(frozenset(e) for e in edges)
        # Check edges are simple
        if len(edge_set) != len(edges):
            return None
        # Check cycles
        for i, cycle in enumerate(cycles):
            L = i + 3  # lengths 3..40
            if len(cycle) != L:
                return None
            if len(set(cycle)) != L:
                return None
            for j in range(L):
                u, v = cycle[j], cycle[(j+1) % L]
                if frozenset((u,v)) not in edge_set:
                    return None
        return len(edges) - N
    
    # Try construction: base cycle + chords from 0 to 2..10
    for num_chords in range(5, 15):
        if time.time() > deadline:
            break
        
        chords = [(0, i) for i in range(2, 2 + num_chords)]
        all_edges = base_edges + chords
        edge_set = set(frozenset(e) for e in all_edges)
        
        # Find all cycle lengths
        adj_local = [set() for _ in range(N)]
        for u, v in all_edges:
            adj_local[u].add(v)
            adj_local[v].add(u)
        
        # Find cycle lengths using a more efficient method
        # For a graph that's a cycle plus chords from one vertex, 
        # cycles are: the base cycle, and cycles that go from 0 through a chord to k,
        # then along the base cycle to some point, then back.
        
        # Actually, let me just check which lengths are covered by trying to find cycles.
        # For this structured graph, I can enumerate cycles more efficiently.
        
        # A cycle in this graph either:
        # 1. Is the base cycle (length 40)
        # 2. Uses at least one chord. Since all chords are from 0, any cycle using chords
        #    must use 0. So it's: 0 -> chord to k -> path along base cycle -> chord back to 0
        #    or 0 -> chord to k -> path -> chord to m -> path -> 0 (using two chords)
        
        covered_lengths = {40}  # base cycle
        
        # Cycles using exactly one chord (0,k):
        # 0-k-(k+1)-...-(k-1 mod 40)-0, length = 2 + (40-1) = no...
        # 0 to k via chord (1 edge), k to 0 via base cycle: two arcs, lengths k and 40-k
        # So cycle length = 1 + k or 1 + (40-k)
        for k in range(2, 2 + num_chords):
            covered_lengths.add(1 + k)
            covered_lengths.add(1 + (40 - k))
        
        # Cycles using two chords (0,k) and (0,m), k < m:
        # 0-k-(k+1)-...-m-0, length = 1 + (m-k) + 1 = m-k+2
        # 0-k-...-39-0-m-...-k-0? No, that's not simple.
        # 0-k-(k+1)-...-m-0: length = 1 + (m-k) + 1 = m-k+2
        # 0-m-(m+1)-...-k-0: length = 1 + (k-m+40) + 1 = k-m+42
        for i in range(num_chords):
            for j in range(i+1, num_chords):
                k = 2 + i
                m = 2 + j
                covered_lengths.add(m - k + 2)
                covered_lengths.add(k - m + 42)
        
        # Check if all lengths 3..40 are covered
        needed = set(range(3, 41))
        if needed.issubset(covered_lengths):
            # Great! Build the actual cycles
            cycles = []
            for L in range(3, 41):
                cycle = _find_cycle_of_length(L, all_edges, adj_local)
                if cycle is None:
                    break
                cycles.append(cycle)
            else:
                # All cycles found
                return {'edges': all_edges, 'cycles': cycles}
    
    # Fallback: return a valid but not optimal solution
    # Use the seed-like construction
    edges = [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)]
    cycles = []
    for k in range(3, N+1):
        cycles.append(list(range(k)))
    return {'edges': edges, 'cycles': cycles}

def _find_cycle_of_length(L, edges, adj):
    """Find a simple cycle of length L in the graph."""
    # Try from each vertex
    for start in range(N):
        cycle = _dfs_find_cycle(adj, start, L, [start], {start})
        if cycle is not None:
            return cycle
    return None

def _dfs_find_cycle(adj, current, target_len, path, visited):
    if len(path) == target_len:
        if path[0] in adj[current]:
            return path
        return None
    if len(path) >= target_len:
        return None
    for nb in sorted(adj[current]):
        if nb not in visited:
            visited.add(nb)
            path.append(nb)
            result = _dfs_find_cycle(adj, nb, target_len, path, visited)
            if result is not None:
                return result
            path.pop()
            visited.remove(nb)
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_local = 40
    base_edges = [(i, (i + 1) % N_local) for i in range(N_local)]
    
    # Try various chord configurations
    # Strategy: chords from vertex 0 to create a "fan" that covers many lengths
    
    def try_configuration(chords):
        if time.time() > deadline:
            return None
        
        all_edges = base_edges + chords
        edge_set = set(frozenset(e) for e in all_edges)
        
        # Build adjacency
        adj = [set() for _ in range(N_local)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all cycle lengths present
        # For efficiency, use the structure: base cycle + chords from 0
        covered = {40}
        
        # Single chord cycles
        for (u, v) in chords:
            if u == 0:
                k = v
            elif v == 0:
                k = u
            else:
                # General chord, harder to analyze
                continue
            a = k  # arc 0->k forward
            b = N_local - k  # arc k->0 forward (i.e., k->k+1->...->39->0)
            covered.add(a + 1)  # 0-k chord + arc k->0
            covered.add(b + 1)  # 0-k chord + arc 0->k... wait
            
            # Cycle: 0 -> k (chord, 1 edge) -> k+1 -> ... -> 0 (arc of length N-k)
            # Total: 1 + (N-k)
            # Cycle: 0 -> 1 -> ... -> k (arc of length k) -> 0 (chord, 1 edge)
            # Total: k + 1
            covered.add(k + 1)
            covered.add(N_local - k + 1)
        
        # Two-chord cycles (both from 0)
        chord_targets = sorted(set(v for (u,v) in chords if u==0) | set(u for (u,v) in chords if v==0))
        for i in range(len(chord_targets)):
            for j in range(i+1, len(chord_targets)):
                a, b = chord_targets[i], chord_targets[j]
                # 0-a-(a+1)-...-b-0: length 1 + (b-a) + 1 = b-a+2
                # 0-b-(b+1)-...-a-0: length 1 + (a-b+N) + 1 = a-b+N+2
                covered.add(b - a + 2)
                covered.add(a - b + N_local + 2)
        
        needed = set(range(3, 41))
        if not needed.issubset(covered):
            return None
        
        # Build explicit cycles
        cycles = []
        for L in range(3, 41):
            cycle = None
            # Try to find a cycle of length L
            # For structured graphs, we can construct them directly
            if L == 40:
                cycle = list(range(40))
            elif L == 3:
                # Triangle: need three mutually connected vertices
                # In our graph, 0,1,2 form a triangle if (0,2) is an edge
                if frozenset((0,2)) in edge_set:
                    cycle = [0, 1, 2]
                elif frozenset((0,1)) in edge_set and frozenset((1,2)) in edge_set and frozenset((2,0)) in edge_set:
                    cycle = [0, 1, 2]
                else:
                    cycle = _find_cycle_of_length(L, all_edges, adj)
            else:
                # Try to construct: for length L, use arc of base cycle of length L-2
                # plus two chords, or arc of length L-1 plus one chord
                # L-1 + 1 chord: arc of length L-1 on base cycle, plus one chord closing it
                # Arc of length L-1: vertices 0,1,...,L-2, then need edge (L-2, 0)
                if frozenset((0, L-2)) in edge_set and L-2 != 1:
                    cycle = list(range(L-1))  # 0,1,...,L-2
                    if frozenset((0, L-2)) in edge_set:
                        pass  # valid
                    else:
                        cycle = None
                if cycle is None:
                    # Try: two chords. 0-a-...-b-0 with b-a+2 = L, so b-a = L-2
                    found = False
                    for a in chord_targets:
                        b = a + L - 2
                        if b < N_local and frozenset((0,a)) in edge_set and frozenset((0,b)) in edge_set:
                            # Cycle: 0,a,a+1,...,b,0
                            cycle = [0] + list(range(a, b+1))
                            # Verify
                            valid = True
                            for j in range(len(cycle)):
                                u, v = cycle[j], cycle[(j+1)%len(cycle)]
                                if frozenset((u,v)) not in edge_set:
                                    valid = False
                                    break
                            if valid and len(cycle) == L:
                                found = True
                                break
                    if not found:
                        cycle = _find_cycle_of_length(L, all_edges, adj)
            
            if cycle is None:
                return None
            cycles.append(cycle)
        
        # Final validation
        for i, cyc in enumerate(cycles):
            L = i + 3
            if len(cyc) != L or len(set(cyc)) != L:
                return None
            for j in range(L):
                u, v = cyc[j], cyc[(j+1)%L]
                if frozenset((u,v)) not in edge_set:
                    return None
        
        return {'edges': all_edges, 'cycles': cycles}
    
    # Try configurations with increasing number of chords
    for num_chords in range(3, 15):
        if time.time() > deadline:
            break
        
        # Chords from 0 to 2,3,...,num_chords+1
        chords = [(0, i) for i in range(2, num_chords + 2)]
        result = try_configuration(chords)
        if result is not None:
            return result
        
        # Also try chords from 0 to 2,3,4,... but skipping some
        # Try chords to even vertices
        if num_chords >= 4:
            chords_even = [(0, i) for i in range(2, 2*num_chords+1, 2)]
            result = try_configuration(chords_even)
            if result is not None:
                return result
    
    # Fallback: seed-like construction
    edges = [(0,i) for i in range(1,N_local)] + [(i,i+1) for i in range(1,N_local-1)]
    cycles = [list(range(k)) for k in range(3, N_local+1)]
    return {'edges': edges, 'cycles': cycles}

# EVOLVE-BLOCK-END
