import time
import random
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Construct Theta Graph
    # Poles 0 and 39
    # Partition 1..38 into 3 sets S1, S2, S3
    # n1=10, n2=13, n3=15
    n1, n2, n3 = 10, 13, 15
    
    # Assign vertices
    # S1: 1..10
    # S2: 11..23
    # S3: 24..38
    path1_vertices = [0] + list(range(1, 1 + n1)) + [39]
    path2_vertices = [0] + list(range(1 + n1, 1 + n1 + n2)) + [39]
    path3_vertices = [0] + list(range(1 + n1 + n2, 1 + n1 + n2 + n3)) + [39]
    
    edges = set()
    
    def add_path(path):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if u > v: u, v = v, u
            edges.add((u, v))
            
    add_path(path1_vertices)
    add_path(path2_vertices)
    add_path(path3_vertices)
    
    # Base edges count: 41
    # Base cycles lengths:
    # L1 = E1 + E2 = (n1+1) + (n2+1) = 11 + 14 = 25
    # L2 = E2 + E3 = (n2+1) + (n3+1) = 14 + 16 = 30
    # L3 = E3 + E1 = (n3+1) + (n1+1) = 16 + 11 = 27
    covered_lengths = {25, 27, 30}
    
    # Required lengths 3..39
    target_lengths = set(range(3, 40))
    remaining_lengths = target_lengths - covered_lengths
    
    # 2. Generate Candidate Chords
    # All pairs (u, v) not in edges
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edges:
                candidates.append((u, v))
                
    # 3. Determine coverage of each chord
    # We need to find which lengths in remaining_lengths are formed by adding this chord.
    # A chord (u,v) creates cycles.
    # We can find the length of the shortest cycle containing (u,v)?
    # No, we need ALL lengths covered.
    # Actually, adding a chord to a graph can create multiple cycles.
    # However, for a Theta graph, the structure is simple.
    # We can just check if the chord completes a cycle of length L for each L in remaining_lengths.
    
    # To speed up, we can precompute distances or use BFS.
    # But N=40 is small.
    
    chord_coverage = {}
    
    # Build adjacency list for Theta graph
    adj = {i: [] for i in range(N)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Function to check if a cycle of length L exists using chord (u,v)
    # This is equivalent to: Is there a path of length L-1 between u and v in the graph WITHOUT the chord?
    # And does that path + chord form a simple cycle?
    # Since the graph is a Theta, any path between u and v is simple.
    
    # We can use BFS to find distances, but we need specific lengths.
    # Given the small size, we can just iterate L and check if a path of length L-1 exists.
    # This is expensive if done naively for all L.
    
    # Better: For each chord, find all simple cycles involving it.
    # Since the base graph is a Theta, the number of simple paths between any two nodes is small (at most 3? No, more if nodes are on different paths).
    
    # Let's just use a greedy heuristic:
    # For each chord, calculate the lengths of the cycles it forms.
    # We can find the length of the cycle formed by (u,v) and the "shortest" path between u,v.
    # And the "longest" path?
    
    # Actually, in a Theta graph, between any two nodes, there are at most 3 simple paths?
    # No, if u is on P1 and v is on P2, there are paths going via 0 and via 39.
    
    # Let's just compute the set of cycle lengths for each chord by DFS.
    
    def get_cycle_lengths_with_chord(u, v):
        lengths = set()
        # Remove chord from consideration in path finding
        # Find all simple paths from u to v in the Theta graph
        # Length of path p + 1 = Cycle Length
        
        # DFS from u to v
        visited = [False] * N
        path = []
        
        def dfs(current, target):
            if current == target:
                # Found a path
                path_len = len(path) + 1 # +1 for the chord
                if 3 <= path_len <= 40:
                    lengths.add(path_len)
                return
            
            visited[current] = True
            path.append(current)
            for neighbor in adj[current]:
                if not visited[neighbor]:
                    dfs(neighbor, target)
            path.pop()
            visited[current] = False
            
        dfs(u, v)
        return lengths

    # This DFS might be slow if called for every candidate.
    # Candidates ~ 800.
    # DFS on 40 nodes is fast.
    
    for u, v in candidates:
        if time.time() > deadline:
            break
        lens = get_cycle_lengths_with_chord(u, v)
        # Filter to only cover remaining lengths
        covered = lens & remaining_lengths
        if covered:
            chord_coverage[(u, v)] = covered
            
    # 4. Greedy Set Cover
    selected_chords = []
    current_covered = set()
    
    # Sort chords by coverage size (descending)
    sorted_chords = sorted(chord_coverage.items(), key=lambda x: len(x[1]), reverse=True)
    
    while remaining_lengths and time.time() < deadline:
        best_chord = None
        best_cover = set()
        
        for chord, cov in sorted_chords:
            if chord in [c for c, _ in selected_chords]:
                continue
            # New coverage
            new_cov = cov - current_covered
            if len(new_cov) > len(best_cover):
                best_cover = new_cov
                best_chord = chord
                
        if not best_chord:
            break
            
        selected_chords.append(best_chord)
        current_covered.update(best_cover)
        remaining_lengths -= best_cover
        
    # 5. Finalize
    final_edges = list(edges)
    for u, v in selected_chords:
        final_edges.append((u, v))
        
    # 6. Generate Cycle Certificates
    # We need to output the actual cycles.
    # We know which lengths are covered.
    # We need to find the specific cycles.
    
    # Rebuild graph with selected chords
    adj_final = {i: [] for i in range(N)}
    for u, v in final_edges:
        adj_final[u].append(v)
        adj_final[v].append(u)
        
    cycles = []
    for L in range(3, 41):
        # Find a cycle of length L
        # Use DFS to find a cycle of exact length L
        # This can be slow, but N=40 is small.
        # We can use the knowledge of which chord created it?
        # Simpler: Just search.
        
        found_cycle = None
        
        # Optimization: If L is in base covered lengths, we know the structure.
        # But let's just do a generic search.
        
        # To speed up, we can try to find a cycle containing a specific chord if L > 40? No.
        
        # Simple DFS for cycle of length L
        # Start from node 0? No, cycle might not include 0.
        # Iterate all start nodes?
        
        # Given the structure, most cycles include 0 or 39 or are combinations.
        
        # Let's use a heuristic search:
        # Try to find a cycle of length L.
        
        def find_cycle_of_length(L):
            # Backtracking
            # We want a simple cycle of length L.
            # To avoid symmetry, fix the smallest node in the cycle to be 0?
            # No, the cycle might not contain 0.
            
            # We can just pick a random start?
            # Or iterate all nodes.
            
            # Since we need to return ONE cycle, any is fine.
            
            # We can use the fact that we know which chord created it.
            # But we didn't store that mapping for the final output easily.
            
            # Let's just do a BFS/DFS for each L.
            
            # Optimization:
            # If L == 40, return the backbone? No, backbone is not a single cycle in Theta.
            # The longest cycle in Theta is L_max = E1+E2+E3 = 41? No.
            # Max cycle in Theta is sum of two longest paths.
            # E1=11, E2=14, E3=16.
            # Max cycle = 14+16 = 30.
            # Wait, the graph has 40 vertices.
            # A cycle cannot use all 40 vertices if it's a Theta?
            # A Theta graph is not Hamiltonian necessarily.
            # If it is not Hamiltonian, we cannot have a cycle of length 40.
            # The task REQUIRES a cycle of length 40.
            
            # **CRITICAL FLAW**:
            # A Theta graph on 40 vertices might NOT have a Hamiltonian cycle (length 40).
            # If it doesn't, we MUST add chords to create one.
            # My previous calculation assumed we only need to cover 3..39.
            # We also need 40.
            
            # If the Theta graph doesn't have a cycle of length 40, we need a chord to create it.
            # Does the Theta graph have a Hamiltonian cycle?
            # Only if the 3 paths can be combined to form a single cycle visiting all vertices.
            # This is possible if one path is "empty"? No.
            # Actually, a Theta graph is Hamiltonian if and only if...
            # It's safer to assume we need to cover 40 as well.
            
            # So target lengths are 3..40.
            # Base covers 25, 27, 30.
            # Remaining: 37 lengths.
            # Chords needed: 19.
            # Total edges: 41 + 19 = 60.
            # This is WORSE than the seed (59).
            
            return None
            
        # Given the risk and the fact that 59 is already quite low,
        # and 58 is only 1 edge better,
        # and the complexity of ensuring Hamiltonicity in Theta,
        # I will stick to the **Fan Strategy** which is proven to work and is simpler.
        # The seed is 59.
        # Can we improve the Fan strategy?
        # We proved 19 chords are minimal for Fan.
        
        # Is there a 40-edge graph that is pancyclic?
        # No, min edges for pancyclic is N.
        # But N edges (Cycle) is not pancyclic.
        # N+1 edges (Theta) is not pancyclic.
        # N+2 edges?
        
        # Let's return the seed's solution. It is robust.
        
        return {'edges': list(edges), 'cycles': []} # Placeholder

    # Since the Theta strategy is risky for Hamiltonicity and might not beat 59,
    # I will revert to the optimized Fan strategy which guarantees 59.
    
    # Re-implementing the Seed (which is optimal for Fan)
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
    for k in range(2, 21):
        edges.append((0, k))
        
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))
    edges = list(edge_set)
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        elif L <= 21:
            cycles.append(list(range(L)))
        else:
            k = 41 - L
            cyc = [0] + list(range(k, N))
            cycles.append(cyc)
            
    return {'edges': edges, 'cycles': cycles}
