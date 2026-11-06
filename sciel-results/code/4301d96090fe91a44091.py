"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import heapq

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_paths_lower_bound(edges, n):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd = sum(1 for d in deg if d % 2 == 1)
        return max(odd // 2, 1) if odd > 0 else 1

    def decompose_into_trails(edges, n):
        """
        Decompose graph into minimum number of edge-disjoint trails.
        Uses Eulerian augmentation.
        """
        if not edges:
            return []
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        
        # Pair up odd vertices with dummy edges
        dummy_edges = []
        for i in range(0, len(odd_vertices), 2):
            u = odd_vertices[i]
            v = odd_vertices[i+1]
            dummy_edges.append((u, v))
        
        # Build augmented graph
        aug_edges = list(edges) + dummy_edges
        
        adj = defaultdict(list)
        for u, v in aug_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find Eulerian circuits
        used_edges = set()
        for u, v in aug_edges:
            used_edges.add((u, v))
            used_edges.add((v, u))
            
        trails = []
        
        def find_eulerian_circuit(start):
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                # Find unused edge
                next_v = None
                for u in adj[v]:
                    if (v, u) not in used_edges:
                        next_v = u
                        break
                if next_v is None:
                    circuit.append(stack.pop())
                else:
                    used_edges.remove((v, next_v))
                    used_edges.remove((next_v, v))
                    stack.append(next_v)
            return circuit
        
        # We need to ensure we process all components (graph is connected, so 1 component)
        # But dummy edges might create multiple? No, original is connected.
        # Start from any vertex with edges
        start_node = 0
        while not adj[start_node]:
            start_node += 1
            
        # Since graph is connected, one Eulerian circuit exists in augmented graph
        circuit = find_eulerian_circuit(start_node)
        
        # Split circuit into trails by removing dummy edges
        # Circuit is a list of vertices v0, v1, ..., vk = v0
        # Edges are (v_i, v_{i+1})
        
        # Identify dummy edges in the circuit
        dummy_set = set()
        for u, v in dummy_edges:
            dummy_set.add((u, v))
            dummy_set.add((v, u))
            
        current_trail = []
        current_start = None
        for i in range(len(circuit) - 1):
            u = circuit[i]
            v = circuit[i+1]
            edge = (u, v)
            if not current_trail:
                current_start = u
                current_trail = [u]
            current_trail.append(v)
            
            if edge in dummy_set:
                # End of a trail
                trails.append(current_trail)
                current_trail = []
                
        if current_trail:
            trails.append(current_trail)
            
        return trails

    def trail_to_simple_paths(trail):
        """
        Decompose a trail into simple paths.
        Greedy: if a vertex repeats, split.
        """
        if not trail:
            return []
            
        paths = []
        current_path = [trail[0]]
        visited_in_path = {trail[0]}
        
        for i in range(1, len(trail)):
            v = trail[i]
            if v in visited_in_path:
                # Cycle detected. 
                # The segment from first occurrence of v to current v is a cycle.
                # We cannot include the cycle in a simple path.
                # We must split.
                # Let's say current path is P.
                # We end the current path before the cycle starts?
                # Actually, the trail is a sequence of edges.
                # If we hit a visited vertex, the edge (prev, v) closes a loop or enters visited.
                # Since it's a trail, edges are unique.
                # If v is visited, it means we are returning to a vertex in the current path.
                # This forms a cycle.
                # We can't have a cycle in a simple path.
                # So we must break the path.
                
                # Find the index of the first occurrence of v in current_path
                # But we need to be careful. The cycle is the segment.
                # We can output the path up to the vertex before the cycle start?
                # Or we can just break here and start a new path?
                
                # Simple strategy:
                # End current path at the previous vertex.
                # Start new path at v.
                # But we need to cover the edge (prev, v).
                # If we end at prev, we haven't used (prev, v).
                # So we should end at prev, and the new path starts with (prev, v)?
                # No, new path starts at v? Then edge (prev, v) is lost.
                
                # Correct strategy:
                # The cycle is $v_{start} \dots v_{end}=v$.
                # The path before $v_{start}$ is $P_{pre}$.
                # The path after $v$ is $P_{post}$.
                # We can form:
                # 1. $P_{pre}$ (ends at $v_{start}$)
                # 2. The cycle $v_{start} \dots v$ decomposed into 2 paths.
                # 3. $P_{post}$ (starts at $v$)
                
                # This is complex to implement inline.
                # Simpler: Just break the trail into single edges if it gets too complex?
                # No, that increases count too much.
                
                # Let's try:
                # End current path at the vertex before the one that caused the repeat?
                # If trail is ... u, v (where v is in visited).
                # We can end path at u.
                # Then we have edge (u, v) left.
                # And the rest of the trail starting from v.
                # So:
                # 1. Pop u from current_path? No, u is the last one.
                # 2. current_path ends at u.
                # 3. Start new path with u, v?
                #    If we start new path with u, v, then u is repeated globally?
                #    Paths are edge-disjoint, but vertices can be shared?
                #    "Paths must be simple". Simple path: no repeated vertices *within* the path.
                #    Vertices CAN be shared between different paths.
                
                # So:
                # End current path at u.
                # Start new path at u, then v.
                # But u is already in current_path.
                # If we start new path at u, it's fine.
                # But we need to remove u from current_path?
                # No, current_path is ... u.
                # We keep current_path as is (ending at u).
                # New path starts with [u, v].
                # But edge (u, v) is used in the new path.
                # So we don't add v to current_path.
                
                # Let's refine:
                # Current path: [w, ..., u]
                # Next vertex: v (which is in {w, ..., u})
                # We output current_path.
                # Start new path: [u, v, ...]
                # But we must ensure the new path is simple.
                # Since v is in current_path, and we are starting a new path,
                # we just need to ensure no repeats in the new path.
                
                paths.append(current_path)
                current_path = [u, v]
                visited_in_path = {u, v}
            else:
                current_path.append(v)
                visited_in_path.add(v)
                
        if current_path:
            paths.append(current_path)
            
        return paths

    def merge_paths(paths, n, edges_set):
        """
        Try to merge paths to reduce count.
        """
        # This is a heuristic.
        # We can try to merge any two paths P1, P2 if P1.end == P2.start and union is simple.
        # For simplicity, we will just do a few rounds of random merging attempts.
        
        # For now, return paths as is.
        # A full merge implementation is complex.
        return paths

    def solve_instance(edges, n):
        trails = decompose_into_trails(edges, n)
        
        all_paths = []
        for trail in trails:
            sps = trail_to_simple_paths(trail)
            all_paths.extend(sps)
            
        # Optional: Try to merge
        # (Skipped for now to ensure correctness and speed)
        
        return all_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        if time.time() > deadline - 10:
            # Fallback
            results.append([[u, v] for u, v in edges])
            continue
            
        paths = solve_instance(edges, n)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
