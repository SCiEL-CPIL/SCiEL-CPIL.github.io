import time
import random
from itertools import product

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency lists and sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
            
    def edge_key(u, v):
        return (min(u, v), max(u, v))

    # Function to find one Hamiltonian cycle avoiding used_edges
    def find_hamiltonian_cycle(used_edges, deadline, rng=None, start_vertex=0):
        # We start at start_vertex to reduce symmetry or allow restarts
        
        path = [start_vertex]
        visited = {start_vertex}
        
        # Recursion with backtracking
        # Optimization: Sort neighbors by degree in remaining graph (Warnsdorff's rule heuristic)
        
        def dfs(current, remaining_count):
            if time.time() > deadline:
                return None
            
            if remaining_count == 0:
                # Check if we can close the cycle to start (start_vertex)
                ek = edge_key(current, start_vertex)
                if ek not in used_edges and start_vertex in adj[current]:
                    return list(path)
                return None
            
            # Get unvisited neighbors
            candidates = []
            for nb in adj[current]:
                if nb not in visited:
                    ek = edge_key(current, nb)
                    if ek not in used_edges:
                        # Heuristic: count how many unvisited neighbors nb has
                        # Lower count is preferred (Warnsdorff)
                        deg = 0
                        for n2 in adj[nb]:
                            if n2 not in visited:
                                if edge_key(nb, n2) not in used_edges:
                                    deg += 1
                        candidates.append((deg, nb))
            
            if not candidates:
                return None
                
            # Sort by degree (fewest options first)
            if rng:
                # Add random tie-breaker
                candidates = [(deg, rng.random(), nb) for deg, nb in candidates]
                candidates.sort(key=lambda x: (x[0], x[1]))
                iter_candidates = [nb for _, _, nb in candidates]
            else:
                candidates.sort(key=lambda x: x[0])
                iter_candidates = [nb for _, nb in candidates]
            
            for nb in iter_candidates:
                path.append(nb)
                visited.add(nb)
                
                result = dfs(nb, remaining_count - 1)
                if result is not None:
                    return result
                    
                path.pop()
                visited.remove(nb)
                
            return None
        
        return dfs(start_vertex, n - 1)

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    used_edges = set()
    cycles = []
    
    # Theoretical max is floor(m/n) = 52/21 = 2? 
    # Wait, m=52, n=21. 52/21 = 2.47. So max 2 edge-disjoint Hamiltonian cycles?
    # Let's re-verify edge count.
    # 21 vertices, 5-regular graph. Sum of degrees = 21 * 5 = 105. m = 105 / 2 = 52.5.
    # Wait, 21 is odd. A 5-regular graph on an odd number of vertices cannot exist because sum of degrees must be even.
    # Let me count edges in P.
    # [0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20] -> 6 edges from 0?
    # Let's check degrees of 0.
    # 0: 1, 2, 5, 16, 19, 20. Degree 6.
    # So it's not 5-regular.
    # Let's count total edges.
    # The list has 52 entries?
    # Let's just run the algorithm and see how many we can get.
    # If max is 2, we should stop at 2. If max is 3, stop at 3.
    # The previous program found 3. So max >= 3.
    # m=52. 3 * 21 = 63 > 52. Impossible to have 3 edge-disjoint Hamiltonian cycles if m=52.
    # Wait. 3 cycles * 21 edges/cycle = 63 edges.
    # If m=52, we can have at most floor(52/21) = 2 edge-disjoint Hamiltonian cycles.
    # Why did the previous program claim 3?
    # Let's check the edge count of P again carefully.
    
    # 0: 1, 2, 5, 16, 19, 20 (6)
    # 1: 0, 2, 3, 6, 17, 20 (6)
    # 2: 0, 1, 3, 4, 7, 18 (6)
    # 3: 1, 2, 4, 5, 8, 19 (6)
    # 4: 2, 3, 5, 6, 9, 20 (6)
    # 5: 0, 3, 4, 6, 7, 10 (6)
    # 6: 1, 4, 5, 7, 8, 11 (6)
    # 7: 2, 5, 6, 8, 9, 12 (6)
    # 8: 3, 6, 7, 9, 10, 13 (6)
    # 9: 4, 7, 8, 10, 11, 14 (6)
    # 10: 5, 8, 9, 11, 12, 15 (6)
    # 11: 6, 9, 10, 12, 13, 16 (6)
    # 12: 7, 10, 11, 13, 14, 17 (6)
    # 13: 8, 11, 12, 14, 15, 18 (6)
    # 14: 9, 12, 13, 15, 16, 19 (6)
    # 15: 10, 13, 14, 16, 17, 20 (6)
    # 16: 0, 11, 14, 15, 17, 18 (6)
    # 17: 1, 12, 15, 16, 18, 19 (6)
    # 18: 2, 13, 16, 17, 19, 20 (6)
    # 19: 0, 3, 14, 17, 18, 20 (6)
    # 20: 0, 1, 4, 15, 18, 19 (6)
    
    # Every vertex has degree 6.
    # Sum of degrees = 21 * 6 = 126.
    # m = 126 / 2 = 63.
    # 63 edges.
    # 63 / 21 = 3.
    # So exactly 3 edge-disjoint Hamiltonian cycles are possible if the graph decomposes into 3 Hamiltonian cycles.
    # This is a 6-regular graph on 21 vertices.
    # The task is to find 3 such cycles.
    
    max_cycles = 3
    for attempt in range(10): # Increased attempts for robustness
        if time.time() > deadline:
            break
            
        if len(cycles) >= max_cycles:
            break
            
        # If we have found some cycles, we are looking for the next disjoint one
        # The search space shrinks as we add edges to used_edges
        
        # First attempt: Deterministic
        cycle = find_hamiltonian_cycle(used_edges, deadline, rng=None, start_vertex=0)
        
        if not cycle:
            # If deterministic failed, try randomized restarts with different start vertices
            found = False
            for i in range(20):
                if time.time() > deadline:
                    break
                rng = random.Random(i + attempt * 100) # Different seeds for each attempt
                start_v = i % n
                cycle = find_hamiltonian_cycle(used_edges, deadline, rng=rng, start_vertex=start_v)
                if cycle:
                    found = True
                    break
            
            if not found:
                # If still not found, we stop.
                break

        if cycle:
            cycles.append(cycle)
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
        else:
            break

    # Ensure result types are correct
    result = []
    for cycle in cycles:
        # Convert to list of ints
        result.append([int(v) for v in cycle])
        
    return result
# EVOLVE-BLOCK-END
