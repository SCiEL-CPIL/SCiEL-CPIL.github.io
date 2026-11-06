import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_eulerian(edges, n):
    """
    Decompose an undirected graph into cycles and edges using the Eulerian trail approach.
    
    Strategy:
    1. Build a directed multigraph with two arcs per undirected edge.
    2. Find an Eulerian circuit (or trails in each connected component).
    3. Decompose each Eulerian circuit into simple cycles using a stack-based algorithm.
    4. The undirected edges covered by each cycle become one piece; uncovered edges become single-edge pieces.
    
    This guarantees a valid decomposition. The number of cycles is related to the number
    of edge-disjoint cycles in the original graph.
    """
    if not edges:
        return []

    # Build adjacency for the directed Eulerian multigraph
    # Each undirected edge (u,v) becomes two arcs: u->v and v->u
    # We use a multigraph representation: adj[u] is a list of (v, edge_id)
    adj = defaultdict(list)
    edge_map = {}  # arc_id -> (u, v) where u->v is the arc
    edge_id = 0
    for u, v in edges:
        # Arc u -> v
        adj[u].append((v, edge_id))
        edge_map[edge_id] = (u, v)
        edge_id += 1
        # Arc v -> u
        adj[v].append((u, edge_id))
        edge_map[edge_id] = (v, u)
        edge_id += 1

    # Find all Eulerian circuits/trails
    # Since each vertex has equal in-degree and out-degree (each undirected edge contributes
    # one in and one out), each connected component is Eulerian.
    
    # We'll process each connected component
    visited_arcs = set()
    all_cycles = []  # List of cycles, each cycle is a list of (u, v) arcs
    
    # Find connected components in the directed multigraph
    # Actually, since it's symmetric, we can just find components in the undirected version
    # and process each one
    
    # Build undirected components
    undirected_adj = defaultdict(set)
    for u, v in edges:
        undirected_adj[u].add(v)
        undirected_adj[v].add(u)
    
    # Find connected components
    visited_nodes = set()
    components = []
    for node in range(n):
        if node in visited_nodes or not undirected_adj[node]:
            continue
        # BFS
        comp = set()
        stack = [node]
        visited_nodes.add(node)
        while stack:
            curr = stack.pop()
            comp.add(curr)
            for neighbor in undirected_adj[curr]:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    stack.append(neighbor)
        components.append(comp)
    
    # For each component, find Eulerian circuit and decompose
    for comp in components:
        if not comp:
            continue
        
        # Find a start node with non-zero degree
        start = None
        for node in comp:
            if adj[node]:
                start = node
                break
        if start is None:
            continue
        
        # Find Eulerian circuit using Hierholzer's algorithm
        # We need to track which arcs have been used
        # Reset visited_arcs for this component
        comp_arcs_used = set()
        
        # Hierholzer's algorithm
        # We'll find the Eulerian circuit as a sequence of vertices
        circuit = []
        
        def find_eulerian(start_node):
            stack = [start_node]
            while stack:
                node = stack[-1]
                # Find an unused arc from this node
                while adj[node]:
                    v, arc_id = adj[node].pop()
                    if arc_id not in comp_arcs_used:
                        comp_arcs_used.add(arc_id)
                        stack.append(v)
                        break
                else:
                    # No more arcs from this node, pop
                    circuit.append(stack.pop())
                    if stack:
                        # Add the arc from stack[-1] to node
                        circuit.append((stack[-1], node))
        
        find_eulerian(start)
        
        # The circuit is now in reverse order. Let's reconstruct it.
        # Actually, Hierholzer's gives us the circuit in reverse.
        # Let's re-do this more carefully.
        
        # Let me use a cleaner implementation of Hierholzer's
        comp_arcs_used.clear()
        circuit_vertices = []
        
        def hierholzer(start_node):
            """Find Eulerian circuit starting at start_node."""
            stack = [(start_node, None)]  # (node, arc_id used to get here)
            path = []
            
            while stack:
                node, in_arc = stack[-1]
                found = False
                while adj[node]:
                    v, arc_id = adj[node].pop()
                    if arc_id not in comp_arcs_used:
                        comp_arcs_used.add(arc_id)
                        stack.append((v, arc_id))
                        found = True
                        break
                if not found:
                    path.append(stack.pop())
            
            # path is in reverse order
            path.reverse()
            return path
        
        euler_path = hierholzer(start)
        
        # euler_path is a sequence of vertices: [v0, v1, v2, ..., vk]
        # where consecutive pairs are arcs in the Eulerian circuit
        # The circuit is v0->v1->v2->...->vk (and vk should equal v0 for a circuit)
        
        # Now decompose this Eulerian circuit into simple cycles
        # Using the stack-based algorithm:
        # Walk along the circuit, maintaining a stack of vertices.
        # When we encounter a vertex already on the stack, the subsequence
        # from that vertex to the current position forms a cycle.
        
        if not euler_path:
            continue
        
        # The Eulerian circuit as a list of vertices
        # Note: for a circuit, the last vertex equals the first
        # But in Hierholzer's output, it might not explicitly close.
        # Let's check: the path should form a closed walk.
        
        # Actually, let me reconsider. The standard way to decompose an Eulerian
        # circuit into simple cycles is:
        # 1. Walk along the circuit.
        # 2. Keep a dictionary mapping vertex -> position in current path.
        # 3. When you see a vertex that's already in the current path, 
        #    the subpath from that vertex to the current position is a cycle.
        # 4. Remove that cycle and continue.
        
        # But a simpler approach: since we have the Eulerian circuit as a sequence
        # of vertices, we can find all simple cycles by looking for repeated vertices.
        
        # Let's use a different approach: 
        # The Eulerian circuit visits vertices in a specific order. 
        # Each time a vertex is revisited, we can extract a cycle.
        
        # Actually, the cleanest way: 
        # Maintain a stack. Push vertices as we walk. 
        # When we push a vertex that's already on the stack, pop up to and including
        # that vertex to form a cycle.
        
        if len(euler_path) < 2:
            continue
        
        # Ensure the circuit is closed: if last != first, add first at end
        if euler_path[0] != euler_path[-1]:
            euler_path.append(euler_path[0])
        
        # Now decompose
        cycles_in_component = []
        stack = []
        vertex_positions = {}
        
        for i, v in enumerate(euler_path):
            if v in vertex_positions:
                # Found a cycle: from vertex_positions[v] to current position i
                cycle_start_idx = vertex_positions[v]
                cycle_vertices = euler_path[cycle_start_idx:i+1]
                if len(cycle_vertices) >= 3:
                    cycles_in_component.append(cycle_vertices)
                # Remove from stack up to and including the previous occurrence
                while stack and stack[-1] != v:
                    del vertex_positions[stack.pop()]
                if stack and stack[-1] == v:
                    stack.pop()
                    del vertex_positions[v]
                # Now re-push v
                stack.append(v)
                vertex_positions[v] = i
            else:
                stack.append(v)
                vertex_positions[v] = i
        
        # Add cycles to global list
        all_cycles.extend(cycles_in_component)
    
    # Now, convert cycles (sequences of vertices) to edge sets
    # and determine which undirected edges are covered by cycles vs. remain as single edges
    
    covered_edges = set()
    pieces = []
    
    for cycle in all_cycles:
        # cycle is a list of vertices [v0, v1, ..., vk] where vk == v0
        # The edges are (v0,v1), (v1,v2), ..., (vk-1,vk)
        cycle_edges = set()
        valid = True
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            e = (min(u, v), max(u, v))
            if e in covered_edges:
                # This edge is already covered, which shouldn't happen
                # but let's handle it gracefully
                valid = False
                break
            cycle_edges.add(e)
        
        if valid and len(cycle_edges) > 0:
            # Check that all these edges are actually in the original graph
            all_in_graph = True
            for e in cycle_edges:
                if e not in set(edges):
                    all_in_graph = False
                    break
            
            if all_in_graph:
                covered_edges.update(cycle_edges)
                # Output the cycle as a piece (remove the repeated last vertex)
                if cycle[0] == cycle[-1]:
                    pieces.append(cycle[:-1])
                else:
                    pieces.append(cycle)
    
    # Add uncovered edges as single-edge pieces
    for u, v in edges:
        e = (min(u, v), max(u, v))
        if e not in covered_edges:
            pieces.append([u, v])
    
    return pieces


def _decompose_greedy_cycles(edges, n, time_limit):
    """
    Alternative: greedily find and remove cycles.
    Used as a fallback or comparison.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining:
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a cycle
        found_cycle = None
        for start in range(n):
            if time.time() - start_time > time_limit:
                break
            if not adj[start] or len(adj[start]) < 2:
                continue
            
            # DFS to find a cycle
            path = [start]
            on_path = {start}
            
            def dfs(u):
                if time.time() - start_time > time_limit:
                    return None
                for v in sorted(adj[u]):
                    if time.time() - start_time > time_limit:
                        return None
                    if v < start:
                        continue
                    if v == start:
                        if len(path) >= 3:
                            return path[:]
                        continue
                    if v not in on_path:
                        on_path.add(v)
                        path.append(v)
                        result = dfs(v)
                        if result:
                            return result
                        path.pop()
                        on_path.discard(v)
                return None
            
            cycle = dfs(start)
            if cycle:
                # Verify all edges are in remaining
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        valid = False
                        break
                if valid:
                    found_cycle = cycle
                    break
        
        if found_cycle:
            pieces.append(found_cycle)
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            # No more cycles
            for e in remaining:
                pieces.append(list(e))
            remaining = set()
            break
    
    for e in remaining:
        pieces.append(list(e))
    
    return pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            # Fallback: just output all edges as single-edge pieces
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        # Use the Eulerian decomposition approach
        graph_time = min(remaining_time * 0.8, 30.0)
        
        pieces = _decompose_eulerian(edges, n)
        
        # Validate: ensure pieces are edge-disjoint and cover all edges
        used_edges = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check for repeated vertices
                if len(set(piece)) < len(piece):
                    valid = False
                    break
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
        
        # Check coverage
        all_edges = set((min(u, v), max(u, v)) for u, v in edges)
        if not valid or used_edges != all_edges:
            # Fallback to greedy
            pieces = _decompose_greedy_cycles(edges, n, min(remaining_time * 0.5, 20.0))
        
        results.append(pieces)

    return results
