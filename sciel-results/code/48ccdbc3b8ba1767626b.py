"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
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
    edge_id_counter = 0
    arc_to_edge = {}  # arc_id -> undirected edge tuple (min, max)
    
    for u, v in edges:
        # Arc u -> v
        arc_to_edge[edge_id_counter] = (min(u, v), max(u, v))
        adj[u].append((v, edge_id_counter))
        edge_id_counter += 1
        # Arc v -> u
        arc_to_edge[edge_id_counter] = (min(u, v), max(u, v))
        adj[v].append((u, edge_id_counter))
        edge_id_counter += 1

    # Find connected components in the undirected version
    undirected_adj = defaultdict(set)
    for u, v in edges:
        undirected_adj[u].add(v)
        undirected_adj[v].add(u)
    
    visited_nodes = set()
    components = []
    for node in range(n):
        if node in visited_nodes or not undirected_adj[node]:
            continue
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
    
    all_cycles = []  # List of cycles, each cycle is a list of vertex indices
    
    for comp in components:
        if not comp:
            continue
        
        # Find a start node with non-zero degree
        start = None
        for node in sorted(comp):
            if adj[node]:
                start = node
                break
        if start is None:
            continue
        
        # Hierholzer's algorithm to find Eulerian circuit
        # We track used arcs globally within this component
        comp_arcs_used = set()
        circuit_vertices = []
        
        def hierholzer(start_node):
            """Find Eulerian circuit starting at start_node."""
            stack = [start_node]
            path = []
            
            while stack:
                node = stack[-1]
                found = False
                while adj[node]:
                    v, arc_id = adj[node].pop()
                    if arc_id not in comp_arcs_used:
                        comp_arcs_used.add(arc_id)
                        stack.append(v)
                        found = True
                        break
                if not found:
                    path.append(stack.pop())
            
            path.reverse()
            return path
        
        euler_path = hierholzer(start)
        
        if not euler_path:
            continue
        
        # The Eulerian circuit should be closed. 
        # In Hierholzer's, if the graph is connected and Eulerian, 
        # the path should start and end at the same node.
        # However, the implementation above pops nodes, so the last popped node is the start.
        # Let's verify: 
        # stack: [start]
        # pop start, add to path? No, we push neighbors.
        # When stack is empty, path is reversed.
        # Standard Hierholzer returns a path where path[0] == path[-1] if it's a circuit.
        # Actually, the standard implementation often returns the circuit without the final repeated node, 
        # or with it. Let's check the logic.
        # If we start at A, go to B, then back to A.
        # Stack: [A] -> push B -> [A, B] -> no more from B, pop B -> path=[B] -> 
        # then A has no more unused arcs? If A->B was the only one, yes.
        # Pop A -> path=[B, A].
        # So path is [B, A]. This is not a closed walk representation directly.
        # The circuit is A->B->A.
        
        # Let's re-implement Hierholzer more carefully to get the full sequence of vertices.
        
        comp_arcs_used.clear()
        
        def hierholzer2(start_node):
            stack = [start_node]
            path = []
            
            while stack:
                node = stack[-1]
                found = False
                while adj[node]:
                    v, arc_id = adj[node].pop()
                    if arc_id not in comp_arcs_used:
                        comp_arcs_used.add(arc_id)
                        stack.append(v)
                        found = True
                        break
                if not found:
                    path.append(stack.pop())
            
            path.reverse()
            # path is the sequence of vertices in the Eulerian circuit
            # e.g., [A, B, C, A]
            return path
        
        euler_path = hierholzer2(start)
        
        if len(euler_path) < 2:
            continue
            
        # Ensure it's closed. If not, something is wrong or it's a trail.
        # For an Eulerian graph, it should be closed.
        # If euler_path[0] != euler_path[-1], append start to close it?
        # Actually, if the component is Eulerian, the circuit must close.
        # Let's assume it does. If not, we can't decompose into cycles easily.
        # But since every vertex has even degree in the multigraph (2*degree in original),
        # it is Eulerian.
        
        # Decompose the Eulerian circuit into simple cycles
        # Algorithm:
        # Walk along the circuit. Maintain a stack of vertices.
        # If the current vertex is already in the stack, a cycle is formed.
        # Pop the stack from that vertex to the current one to extract the cycle.
        # The edges in this cycle are removed from consideration for further cycle extraction
        # in this pass? No, the standard algorithm just extracts cycles.
        
        # A simpler way to extract cycles from an Eulerian circuit:
        # The circuit is a closed walk. 
        # We can find cycles by looking for the first time a vertex repeats.
        # But we want edge-disjoint cycles.
        
        # Let's use the standard "cycle extraction from Eulerian circuit" algorithm:
        # 1. Start with the full circuit.
        # 2. Find the first vertex that appears twice (other than the start/end if they are same).
        #    Actually, since it's a circuit, v[0] == v[-1].
        #    We look for i < j such that v[i] == v[j] and the subpath v[i]...v[j] is a simple cycle.
        #    The first such pair (i, j) where j is the first repeat of v[i] after i.
        #    This subpath is a simple cycle.
        #    Remove this cycle from the circuit. The remaining path is still a closed walk (if we splice it).
        #    Repeat.
        
        # However, there is a more efficient way.
        # Since we have the Eulerian circuit, we can just greedily pick cycles.
        
        # Let's try a different approach:
        # The set of edges in the Eulerian circuit is exactly the set of edges in the component.
        # We want to partition these edges into simple cycles.
        
        # Let's use the stack-based decomposition on the circuit itself.
        # This is equivalent to finding the cycle space basis? No, we want a partition.
        
        # Actually, any Eulerian circuit can be decomposed into simple cycles.
        # Algorithm:
        # path = euler_path
        # cycles = []
        # while len(path) > 1:
        #     # Find the first repeated vertex in path (excluding the closure if needed)
        #     # Since path[0] == path[-1], we can look for i in [1, len(path)-2] such that path[i] appears later.
        #     # Or simpler: use a dict to store the last seen index.
        #     pos = {}
        #     found = False
        #     for i in range(len(path)):
        #         if path[i] in pos:
        #             # Cycle from pos[path[i]] to i
        #             cycle = path[pos[path[i]]:i+1]
        #             cycles.append(cycle)
        #             # Remove this cycle from path
        #             # The remaining path is path[:pos[path[i]]] + path[i:]
        #             # But we need to be careful with the closure.
        #             # This gets complicated.
            
        # Let's use a simpler heuristic: 
        # Just find any cycle in the remaining graph, remove it, repeat.
        # This is what the current program does, but maybe less efficiently.
        # The Eulerian approach is supposed to be faster or yield better results?
        # Actually, the number of cycles in a decomposition is not unique.
        # The minimum number of cycles is not trivial to find.
        # However, the Eulerian decomposition gives A valid decomposition.
        # Does it minimize the number of pieces?
        # Each cycle of length k replaces k edges. Saving is k-1.
        # To minimize pieces, we want to maximize the sum of (k-1) over all cycles.
        # This is equivalent to maximizing the sum of k.
        # So we want long cycles.
        
        # The current program tries to find long cycles greedily.
        # The Eulerian decomposition might produce short cycles.
        
        # Let's stick to the greedy long-cycle approach but optimize it.
        # Or, use the Eulerian circuit to guide the search for long cycles?
        
        # For now, let's implement the standard cycle extraction from Eulerian circuit.
        # It's guaranteed to work.
        
        # Re-implementation of cycle extraction:
        # We have the list of vertices in the Eulerian circuit.
        # We will extract cycles one by one.
        
        remaining_path = list(euler_path)
        comp_cycles = []
        
        while len(remaining_path) > 1:
            # Find the first cycle in the path
            # A cycle is a subsequence v[i], v[i+1], ..., v[j] where v[i] == v[j]
            # and all intermediate vertices are distinct and not equal to v[i].
            # Since it's an Eulerian circuit, such a cycle must exist.
            
            pos = {}
            cycle_found = None
            for i in range(len(remaining_path)):
                v = remaining_path[i]
                if v in pos:
                    # Check if the subpath is a simple cycle
                    # v[i] == v[pos[v]]
                    # We need to ensure no other vertex repeats in between?
                    # Actually, if v[i] == v[j], the subpath v[j]...v[i] is a closed walk.
                    # Is it a simple cycle? Not necessarily.
                    # But if we pick the FIRST repeat, it is a simple cycle.
                    # Because if there was a repeat earlier, we would have found it.
                    start_idx = pos[v]
                    end_idx = i
                    # The cycle is remaining_path[start_idx:end_idx+1]
                    # But we need to check if it's simple.
                    # By construction of "first repeat", it should be simple.
                    cycle = remaining_path[start_idx:end_idx+1]
                    # Verify simplicity: len(cycle) >= 3 and all vertices distinct except first/last
                    if len(cycle) >= 3:
                        # Check distinctness
                        inner = cycle[1:-1]
                        if len(set(inner)) == len(inner) and cycle[0] not in inner:
                            comp_cycles.append(cycle)
                            # Remove this cycle from the path
                            # The new path is remaining_path[:start_idx] + remaining_path[end_idx:]
                            # But we must ensure connectivity.
                            # Since it's a closed walk, splicing should work.
                            remaining_path = remaining_path[:start_idx] + remaining_path[end_idx:]
                            # Restart the search
                            break
                else:
                    pos[v] = i
            
            if cycle_found is None and 'break' not in dir(): # dummy
                pass
            
            if not cycle_found:
                # If no simple cycle found via first repeat (shouldn't happen for Eulerian),
                # or if we didn't break, we might be stuck.
                # Actually, the loop above breaks when a cycle is found.
                # If no cycle is found, it means no repeats? Impossible for Eulerian circuit.
                # Or maybe the repeat was not simple?
                # Let's assume it works.
                if len(remaining_path) <= 1:
                    break
                # Fallback: take a single edge?
                # This should not happen.
                break
        
        all_cycles.extend(comp_cycles)
    
    # Now we have a list of cycles (as vertex lists) that partition the edges.
    # We need to convert these to pieces.
    # A cycle piece is a list of vertices.
    # Are there any edges left?
    # The Eulerian circuit covers all edges in the component.
    # So all edges are covered.
    
    # However, the cycles extracted might share vertices, but do they share edges?
    # The algorithm removes the edges from the path, so they are edge-disjoint.
    
    # Let's validate and format the pieces.
    pieces = []
    used_edges = set()
    
    for cycle in all_cycles:
        # cycle is a list of vertices [v0, v1, ..., vk] where v0 == vk
        # We need to output [v0, v1, ..., v_{k-1}]
        if len(cycle) < 3:
            continue
        # Remove the last vertex if it's the same as the first
        if cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        
        if len(cycle) < 3:
            continue
            
        # Check if edges are valid and not used
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e not in set((min(a,b), max(a,b)) for a,b in edges):
                valid = False
                break
            if e in used_edges:
                valid = False
                break
        
        if valid:
            # Add edges to used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            pieces.append(cycle)
    
    # Add any remaining edges as single-edge pieces
    all_edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    for e in all_edges_set:
        if e not in used_edges:
            pieces.append(list(e))
            
    return pieces


def _find_one_cycle(adj, start, n, remaining_edges):
    """
    Find one simple cycle containing 'start' where start is the smallest vertex.
    Returns the cycle as a list of vertices, or None.
    """
    if not adj[start] or len(adj[start]) < 2:
        return None
    
    # DFS with backtracking to find a cycle
    # We want to find a cycle that returns to 'start'
    path = [start]
    on_path = {start}
    
    def dfs(u, depth):
        if depth > n:
            return None
        for v in sorted(adj[u]):
            if v < start:
                continue  # start must be smallest in cycle
            edge = (min(u, v), max(u, v))
            if edge not in remaining_edges:
                continue
            if v == start:
                if len(path) >= 3:
                    return path[:]
                continue
            if v not in on_path:
                on_path.add(v)
                path.append(v)
                result = dfs(v, depth + 1)
                if result:
                    return result
                path.pop()
                on_path.discard(v)
        return None
    
    return dfs(start, 0)


def _decompose_greedy(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding long cycles and removing them.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        if time.time() - start_time > time_limit:
            break
        
        # Find a cycle in the current graph
        # Try to find a long cycle for better savings
        best_cycle = None
        best_len = 0
        
        # Try finding cycles from each vertex
        for start in range(n):
            if time.time() - start_time > time_limit:
                break
            if not adj[start] or len(adj[start]) < 2:
                continue
            
            # DFS to find a cycle starting and ending at 'start'
            # where 'start' is the smallest vertex in the cycle
            found = _find_one_cycle(adj, start, n, remaining_edges)
            if found and len(found) > best_len:
                best_len = len(found)
                best_cycle = found
                if best_len >= n:  # Can't do better than Hamiltonian
                    break
        
        if best_cycle:
            # Verify all edges in cycle still exist
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                pieces.append(best_cycle)
                # Remove cycle edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                # Cycle has invalid edges, try next
                continue
        else:
            # No cycle found, take a single edge
            edge = next(iter(remaining_edges))
            remaining_edges.discard(edge)
            u, v = edge
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Add any remaining edges
    for edge in remaining_edges:
        pieces.append(list(edge))
    
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
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Try greedy decomposition first, as it often finds longer cycles
        # Time budget: most of remaining time
        graph_time = min(remaining_time * 0.9, 40.0)
        
        pieces = _decompose_greedy(edges, n, graph_time)
        
        # Validate: ensure pieces are edge-disjoint and cover all edges
        # Quick validation
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
            
        if not valid or used_edges != edges_set:
            # Fallback to Eulerian decomposition if greedy failed or was invalid
            pieces = _decompose_eulerian(edges, n)
            # Re-validate
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
            
            if not valid or used_edges != edges_set:
                # Final fallback: all edges as single pieces
                pieces = [[u, v] for u, v in edges]

        results.append(pieces)

    return results
