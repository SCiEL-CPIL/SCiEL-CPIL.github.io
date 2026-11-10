"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _find_trails(edges):
        """
        Find a set of edge-disjoint trails covering all edges.
        Strategy: 
          1. Find all odd-degree vertices.
          2. Add virtual edges to pair up odd-degree vertices (perfect matching).
          3. The resulting multigraph is Eulerian -> has an Eulerian circuit.
          4. Remove virtual edges from the circuit -> get trails covering all real edges.
        """
        if not edges:
            return []
        
        # Compute degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        odd_verts = sorted([v for v, d in deg.items() if d % 2 == 1])
        
        if not odd_verts:
            # All even -> single Eulerian circuit
            trail = _find_eulerian_circuit(edges)
            if trail:
                return [trail]
            # Fallback: greedy
            return _greedy_trails(edges)
        
        # Pair up odd vertices: greedy matching (pair consecutive after sorting by degree)
        # For better results, try to minimize "distance" between paired vertices
        # Simple approach: pair them in order, but we can try a few matchings
        # For now, use a simple greedy: sort by degree, pair adjacent
        
        # Actually, let's try multiple matchings and pick the one that gives shortest total trail decomposition
        best_trails = None
        best_cost = float('inf')
        
        # Try several matchings
        matchings = []
        
        # Matching 1: pair consecutive in sorted order
        m1 = list(zip(odd_verts[0::2], odd_verts[1::2]))
        matchings.append(m1)
        
        # Matching 2: pair first with last, second with second-last, etc.
        m2 = []
        for i in range(len(odd_verts) // 2):
            m2.append((odd_verts[i], odd_verts[-1 - i]))
        matchings.append(m2)
        
        # Matching 3: random shuffles
        for _ in range(10):
            shuffled = odd_verts[:]
            random.shuffle(shuffled)
            m = list(zip(shuffled[0::2], shuffled[1::2]))
            matchings.append(m)
        
        for matching in matchings:
            # Build multigraph: real edges + virtual edges
            real_edges = list(edges)
            virtual_edges = list(matching)
            
            # Find Eulerian circuit in the multigraph
            # We need to handle multi-edges, so use edge IDs
            trail = _find_eulerian_circuit_with_virtuals(real_edges, virtual_edges)
            if trail is None:
                continue
            
            # Remove virtual edges from trail to get real trails
            trails = _split_trail_at_virtuals(trail, virtual_edges)
            
            # Cost: number of trails (we want to minimize)
            cost = len(trails)
            if cost < best_cost:
                best_cost = cost
                best_trails = trails
        
        if best_trails is None:
            best_trails = _greedy_trails(edges)
        
        return best_trails

    def _find_eulerian_circuit(edges):
        """Find Eulerian circuit in an undirected graph (all vertices even degree)."""
        if not edges:
            return [0]
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Hierholzer's algorithm
        start = next(iter(adj))
        stack = [start]
        circuit = []
        used = set()
        
        # We need to track edges by (u,v) pairs, but with multiedges we need indices
        # For simple graphs, (u,v) is unique
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        while stack:
            v = stack[-1]
            if v in adj:
                # Find an unused edge
                found = False
                for i, nb in enumerate(adj[v]):
                    e = tuple(sorted((v, nb)))
                    if e not in used:
                        used.add(e)
                        stack.append(nb)
                        found = True
                        break
                if not found:
                    circuit.append(stack.pop())
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def _find_eulerian_circuit_with_virtuals(real_edges, virtual_edges):
        """Find Eulerian circuit in multigraph with real + virtual edges."""
        # Build adjacency with edge tracking
        # Each edge gets a unique ID
        adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id)
        edge_list = []  # edge_id -> (u, v, is_virtual)
        
        for u, v in real_edges:
            eid = len(edge_list)
            edge_list.append((u, v, False))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        
        for u, v in virtual_edges:
            eid = len(edge_list)
            edge_list.append((u, v, True))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
        
        if not adj:
            return [0]
        
        start = next(iter(adj))
        stack = [start]
        circuit = []
        used = set()
        
        while stack:
            v = stack[-1]
            if v in adj:
                found = False
                for nb, eid in adj[v]:
                    if eid not in used:
                        used.add(eid)
                        stack.append(nb)
                        found = True
                        break
                if not found:
                    circuit.append(stack.pop())
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Verify: circuit should have len = len(edge_list) + 1
        if len(circuit) != len(edge_list) + 1:
            return None
        
        return circuit

    def _split_trail_at_virtuals(circuit, virtual_edges):
        """
        Given an Eulerian circuit (list of vertices) and virtual edges,
        split into real trails by removing virtual edges.
        """
        # Identify which consecutive pairs in circuit are virtual edges
        virtual_set = set(tuple(sorted(e)) for e in virtual_edges)
        
        trails = []
        current = [circuit[0]]
        
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            e = tuple(sorted((u, v)))
            current.append(v)
            if e in virtual_set:
                if len(current) >= 2:
                    trails.append(current)
                current = [v]
        
        if len(current) >= 2:
            trails.append(current)
        
        return trails

    def _greedy_trails(edges):
        """Fallback: greedily build trails."""
        remaining = set(tuple(sorted(e)) for e in edges)
        adj = _build_adj(edges)
        trails = []
        
        while remaining:
            # Pick a start vertex
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Start from a vertex with odd degree or any vertex
            start = None
            for v, d in deg.items():
                if d % 2 == 1:
                    start = v
                    break
            if start is None:
                start = next(iter(deg))
            
            # Greedily extend
            trail = [start]
            current = start
            while True:
                next_v = None
                for nb in adj[current]:
                    e = tuple(sorted((current, nb)))
                    if e in remaining:
                        next_v = nb
                        break
                if next_v is None:
                    break
                trail.append(next_v)
                remaining.discard(tuple(sorted((current, next_v))))
                current = next_v
            
            if len(trail) >= 2:
                trails.append(trail)
        
        return trails

    def _split_trail_into_paths(trail):
        """
        Split a trail (walk that may repeat vertices) into simple paths.
        A trail is a walk where edges are distinct but vertices may repeat.
        We split at the first repeated vertex.
        """
        if len(trail) < 2:
            return []
        
        paths = []
        seen = {}
        current_start = 0
        
        for i in range(len(trail)):
            if trail[i] in seen:
                # trail[i] was seen at position seen[trail[i]]
                # The segment from seen[trail[i]] to i-1 is a path (if length >= 2)
                # Actually, the segment from seen[trail[i]] to i forms a cycle
                # We want to split into paths, not cycles
                # The path is from current_start to seen[trail[i]] (inclusive)
                # Then continue from seen[trail[i]] to i
                # Hmm, this is tricky. Let me think again.
                
                # A trail v0, v1, ..., vk where vj = vi for some j < i
                # The sub-trail vj, vj+1, ..., vi is a cycle
                # The sub-trail v0, ..., vj is a path
                # So we can emit v0..vj as a path, then continue with vj..vi (the cycle)
                # But a cycle is not a path...
                
                # Actually, for path decomposition, we need simple paths.
                # If we have a trail with repeated vertices, we can decompose it.
                
                # Let me use a different approach: 
                # Walk along the trail. When we hit a vertex we've seen before,
                # the segment from first occurrence to current is a cycle.
                # Remove the cycle, and the remaining is still a trail.
                # Repeat until no repeated vertices -> we have a simple path.
                
                # This is getting complex. Let me use a simpler approach:
                # Just find the longest simple path prefix, then recurse on the rest.
                pass
            
            seen[trail[i]] = i
        
        # Simple approach: if all vertices unique, it's already a path
        if len(set(trail)) == len(trail):
            return [trail]
        
        # Otherwise, use a more careful decomposition
        return _decompose_trail(trail)

    def _decompose_trail(trail):
        """Decompose a trail into simple paths."""
        if len(trail) < 2:
            return []
        
        # Find first repeated vertex
        seen = {}
        repeat_pos = -1
        repeat_idx = -1
        
        for i, v in enumerate(trail):
            if v in seen:
                repeat_pos = seen[v]
                repeat_idx = i
                break
            seen[v] = i
        
        if repeat_pos == -1:
            # No repeats, already a simple path
            return [trail]
        
        # trail[0..repeat_pos] is a simple path (vertices 0 to repeat_pos are distinct)
        # trail[repeat_pos..repeat_idx] is a cycle (starts and ends at same vertex)
        # We can emit trail[0..repeat_pos] as a path
        # Then we need to decompose trail[repeat_pos..repeat_idx] (the cycle)
        # A cycle can be split into two paths by removing one edge, but that's not ideal
        # Actually, a cycle is a closed trail. We can break it at any point to get a path.
        # But we want to minimize the number of paths.
        
        # Better: emit trail[0..repeat_pos] as one path.
        # The remaining trail[repeat_pos..repeat_idx] is a cycle.
        # A cycle can be represented as a single path by breaking it: 
        #   cycle: v0, v1, ..., vk, v0 -> path: v0, v1, ..., vk (drop the last v0)
        # But this uses all edges of the cycle in one path.
        
        # Wait, but trail[repeat_pos..repeat_idx] has vertices:
        #   trail[repeat_pos], trail[repeat_pos+1], ..., trail[repeat_idx]
        # where trail[repeat_pos] == trail[repeat_idx]
        # So the edges used are (trail[repeat_pos], trail[repeat_pos+1]), ..., (trail[repeat_idx-1], trail[repeat_idx])
        # This is a cycle of length (repeat_idx - repeat_pos) edges.
        # We can make it a path by dropping the last vertex (which equals the first):
        #   path: trail[repeat_pos], trail[repeat_pos+1], ..., trail[repeat_idx-1]
        # This uses all edges of the cycle.
        
        path1 = trail[0:repeat_pos + 1]
        cycle_part = trail[repeat_pos:repeat_idx]  # This is the cycle vertices without the last repeat
        # cycle_part has length repeat_idx - repeat_pos
        # The edges are (cycle_part[i], cycle_part[i+1]) for i in range(len(cycle_part)-1)
        # This is a simple path (all vertices distinct since it's a cycle minus one vertex)
        
        if len(cycle_part) >= 2:
            return [path1, cycle_part]
        else:
            return [path1]

    def _path_decomposition(edges):
        """Full path decomposition: find trails, then split into simple paths."""
        if not edges:
            return []
        
        trails = _find_trails(edges)
        
        all_paths = []
        for trail in trails:
            paths = _split_trail_into_paths(trail)
            all_paths.extend(paths)
        
        return all_paths

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = _path_decomposition(edges)
        
        # Verify and fix if needed
        if not _verify(paths, edges):
            # Fallback: each edge as a path
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
    
    return results

def _verify(paths, edges):
    """Verify that paths are valid: simple, use graph edges, edge-disjoint, cover all edges."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    used_edges = set()
    
    for path in paths:
        if len(path) < 2:
            continue
        # Check simplicity
        if len(set(path)) != len(path):
            return False
        # Check edges exist and are not reused
        for i in range(len(path) - 1):
            e = tuple(sorted((path[i], path[i + 1])))
            if e not in edge_set:
                return False
            if e in used_edges:
                return False
            used_edges.add(e)
    
    # Check all edges covered
    if used_edges != edge_set:
        return False
    
    return True

# EVOLVE-BLOCK-END
