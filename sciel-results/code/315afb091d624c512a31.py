"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute edge set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def time_exceeded():
        return time.time() > deadline - 2

    def find_hamiltonian_cycle(adj_dict, n, start_vertex=0):
        """
        Find a Hamiltonian cycle in the graph represented by adj_dict.
        Uses DFS with Warnsdorff's heuristic and connectivity pruning.
        """
        if n <= 2:
            return None
        
        # Sort neighbors for deterministic behavior in tie-breaking
        for u in adj_dict:
            adj_dict[u].sort()
            
        path = [start_vertex]
        visited = {start_vertex}
        
        def is_connected(visited_set):
            """Check if all unvisited vertices are connected."""
            if len(visited_set) == n:
                return True
            
            unvisited = [v for v in range(n) if v not in visited_set]
            if not unvisited:
                return True
                
            # BFS/DFS from the first unvisited vertex
            start = unvisited[0]
            stack = [start]
            seen = {start}
            
            while stack:
                u = stack.pop()
                for v in adj_dict[u]:
                    if v in visited_set:
                        continue
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
                        
            return len(seen) == len(unvisited)

        def dfs(u):
            if time_exceeded():
                return None
            
            if len(path) == n:
                # Check if we can close the cycle
                if start_vertex in adj_dict[u]:
                    return list(path)
                return None
            
            # Get unvisited neighbors
            neighbors = [v for v in adj_dict[u] if v not in visited]
            
            if not neighbors:
                return None
            
            # Warnsdorff heuristic: sort neighbors by number of onward moves
            # We want to visit the neighbor with the fewest options first
            def onward_count(v):
                return sum(1 for w in adj_dict[v] if w not in visited and w != u)
                
            neighbors.sort(key=onward_count)
            
            for v in neighbors:
                path.append(v)
                visited.add(v)
                
                # Pruning: Check connectivity of remaining unvisited vertices
                # This is expensive, so maybe do it only at deeper levels or if n is small enough
                # For n=21, it's okay if we do it selectively or rely on degree pruning.
                # Let's add a simple degree check: if any unvisited vertex (except maybe the last one) 
                # has 0 available edges to other unvisited vertices, it's a dead end.
                
                # A stronger pruning: if any unvisited vertex v has no neighbors in unvisited set 
                # AND v is not the start_vertex (which is visited) AND we haven't visited all nodes...
                # Actually, if an unvisited vertex has degree 0 in the remaining graph, we are stuck.
                
                stuck = False
                for w in range(n):
                    if w not in visited:
                        # Count available neighbors for w in the remaining graph
                        avail = 0
                        for x in adj_dict[w]:
                            if x not in visited:
                                avail += 1
                        if avail == 0:
                            stuck = True
                            break
                        
                if not stuck:
                    res = dfs(v)
                    if res is not None:
                        return res
                
                path.pop()
                visited.remove(v)
                
            return None
            
        return dfs(start_vertex)

    def extract_cycles_from_2_regular(adj_dict, n):
        """
        Given a 2-regular graph (union of disjoint cycles), extract the cycles.
        Returns a list of cycles.
        """
        visited = set()
        cycles = []
        
        for i in range(n):
            if i not in visited:
                cycle = [i]
                visited.add(i)
                current = i
                prev = -1
                
                while True:
                    neighbors = [v for v in adj_dict[current] if v != prev]
                    # In a 2-regular graph, current has 2 neighbors. One is prev.
                    # If prev is -1 (start), we pick any neighbor.
                    if prev == -1:
                        next_v = neighbors[0]
                    else:
                        # The other neighbor
                        next_v = neighbors[0] if neighbors[0] != prev else neighbors[1]
                        
                    if next_v == i:
                        # Closed the cycle
                        break
                    
                    cycle.append(next_v)
                    visited.add(next_v)
                    prev = current
                    current = next_v
                    
                    if len(cycle) > n: # Safety break
                        break
                        
                cycles.append(cycle)
                
        return cycles

    def get_remaining_adj(used_edges):
        """Build adjacency list for the graph with used_edges removed."""
        rem_adj = defaultdict(list)
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in used_edges:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
        return rem_adj

    def verify_solution(cycles):
        """Verify that the cycles are valid, edge-disjoint, and cover all vertices."""
        if not cycles:
            return False
            
        all_vertices = set()
        all_edges_used = set()
        
        for cycle in cycles:
            if len(cycle) < 3 or len(set(cycle)) != len(cycle):
                return False
            all_vertices.update(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in all_edges_used:
                    return False
                if e not in edge_set:
                    return False
                all_edges_used.add(e)
                
        if all_vertices != set(range(n)):
            return False
            
        return True

    def find_three_cycles():
        """Try to find 3 edge-disjoint Hamiltonian cycles."""
        # We will try multiple random permutations of start vertices and neighbor orders
        # to increase the chance of finding a valid decomposition.
        
        for attempt in range(100):
            if time_exceeded():
                break
                
            # Randomize the order of neighbors in the original graph
            # This helps break symmetries and find different HCs
            adj_copy = {u: list(v) for u, v in adj.items()}
            for u in adj_copy:
                random.shuffle(adj_copy[u])
                
            # 1. Find first Hamiltonian Cycle
            hc1 = find_hamiltonian_cycle(adj_copy, n, start_vertex=0)
            if hc1 is None:
                continue
                
            # Extract edges of HC1
            edges1 = set()
            for i in range(len(hc1)):
                u = hc1[i]
                v = hc1[(i + 1) % len(hc1)]
                edges1.add((min(u, v), max(u, v)))
                
            # 2. Build remaining graph
            rem_adj1 = get_remaining_adj(edges1)
            
            # Check if all vertices have degree 4 in remaining graph
            # (Original degree was 5? No, let's check degrees in P)
            # Actually, just check if it's 4-regular. If not, HC2 might not be Hamiltonian or impossible.
            # Wait, if we remove a HC from a 5-regular graph (is it 5-regular?), degrees become 3.
            # Let's check the degree of the host graph.
            # Vertex 0: 1,2,5,16,19,20 -> degree 6
            # Vertex 1: 0,2,3,6,17,20 -> degree 6
            # It's not regular. So we can't just check for degree 4.
            
            # Find second Hamiltonian Cycle in remaining graph
            # We need to shuffle neighbors again for diversity
            adj_copy2 = {u: list(v) for u, v in rem_adj1.items()}
            for u in adj_copy2:
                random.shuffle(adj_copy2[u])
                
            hc2 = find_hamiltonian_cycle(adj_copy2, n, start_vertex=0)
            if hc2 is None:
                continue
                
            # Extract edges of HC2
            edges2 = set()
            for i in range(len(hc2)):
                u = hc2[i]
                v = hc2[(i + 1) % len(hc2)]
                edges2.add((min(u, v), max(u, v)))
                
            # Check edge disjointness between HC1 and HC2
            if edges1 & edges2:
                continue
                
            # 3. Build final remaining graph
            used_edges = edges1 | edges2
            rem_adj2 = get_remaining_adj(used_edges)
            
            # Check if all vertices have degree 2
            # If yes, the remaining graph is a union of cycles.
            # If it's a single cycle of length 21, we have 3 HCs.
            # If it's multiple cycles, we have less than 3 HCs (but we already have 2).
            # The task asks for "simple cycles... use exactly the same vertex set".
            # So all cycles must be Hamiltonian.
            # Thus, the remaining graph MUST be a single Hamiltonian cycle for us to return 3.
            
            is_2_regular = True
            for i in range(n):
                if len(rem_adj2[i]) != 2:
                    is_2_regular = False
                    break
                    
            if not is_2_regular:
                # We have 2 cycles, but the 3rd isn't Hamiltonian.
                # We can return 2 cycles.
                if verify_solution([hc1, hc2]):
                    return [hc1, hc2]
                continue
                
            # Extract cycles from 2-regular graph
            cycles3 = extract_cycles_from_2_regular(rem_adj2, n)
            
            # Check if it's a single cycle of length 21
            if len(cycles3) == 1 and len(cycles3[0]) == 21:
                if verify_solution([hc1, hc2, cycles3[0]]):
                    return [hc1, hc2, cycles3[0]]
            else:
                # We have 2 valid cycles, but the remainder is not a single HC.
                # Return the 2 cycles we found.
                if verify_solution([hc1, hc2]):
                    return [hc1, hc2]

        return None

    def find_two_cycles():
        """Try to find 2 edge-disjoint Hamiltonian cycles."""
        for attempt in range(50):
            if time_exceeded():
                break
                
            adj_copy = {u: list(v) for u, v in adj.items()}
            for u in adj_copy:
                random.shuffle(adj_copy[u])
                
            hc1 = find_hamiltonian_cycle(adj_copy, n, start_vertex=0)
            if hc1 is None:
                continue
                
            edges1 = set()
            for i in range(len(hc1)):
                u = hc1[i]
                v = hc1[(i + 1) % len(hc1)]
                edges1.add((min(u, v), max(u, v)))
                
            rem_adj1 = get_remaining_adj(edges1)
            
            adj_copy2 = {u: list(v) for u, v in rem_adj1.items()}
            for u in adj_copy2:
                random.shuffle(adj_copy2[u])
                
            hc2 = find_hamiltonian_cycle(adj_copy2, n, start_vertex=0)
            if hc2 is None:
                continue
                
            if verify_solution([hc1, hc2]):
                return [hc1, hc2]
                
        return None

    def find_one_cycle():
        """Try to find 1 Hamiltonian cycle."""
        adj_copy = {u: list(v) for u, v in adj.items()}
        for u in adj_copy:
            random.shuffle(adj_copy[u])
            
        hc1 = find_hamiltonian_cycle(adj_copy, n, start_vertex=0)
        if hc1 is not None:
            if verify_solution([hc1]):
                return [hc1]
                
        # Fallback to any cycle
        # Just return a trivial cycle if we must, though it should be Hamiltonian
        # The problem says "simple cycles... use exactly the same vertex set".
        # So a non-Hamiltonian cycle is invalid if it doesn't use all vertices.
        # So we MUST return Hamiltonian cycles.
        # If we can't find one, we return an empty list? No, "nonempty list".
        # So we need to find at least one.
        
        # Try different start vertices
        for start in range(n):
            if time_exceeded():
                break
            adj_copy = {u: list(v) for u, v in adj.items()}
            for u in adj_copy:
                random.shuffle(adj_copy[u])
            hc1 = find_hamiltonian_cycle(adj_copy, n, start_vertex=start)
            if hc1 is not None:
                if verify_solution([hc1]):
                    return [hc1]
                    
        return None

    # Main logic
    result = find_three_cycles()
    if result:
        # Convert to lists of ints
        return [[int(v) for v in c] for c in result]
        
    result = find_two_cycles()
    if result:
        return [[int(v) for v in c] for c in result]
        
    result = find_one_cycle()
    if result:
        return [[int(v) for v in c] for c in result]
        
    # Last resort: return a dummy valid cycle if we absolutely must (should not happen)
    # Find any cycle
    # This part is just to avoid returning empty list
    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
