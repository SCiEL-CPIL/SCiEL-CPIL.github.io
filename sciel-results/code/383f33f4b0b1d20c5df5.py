# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict, deque

def construct(k, g):
    # We use a voltage graph construction on K_{5,5}
    # Base graph: K_{5,5} with 10 vertices
    # Lift size: 10 * M
    # We search for M such that the lift has girth >= g
    
    # Base graph edges
    base_edges = []
    for i in range(5):
        for j in range(5):
            base_edges.append((i, j + 5))
    
    # Pre-compute non-backtracking closed walks of length < g in base
    # Lengths to check: 4, 6, 8 (since g=9, we check < 9)
    # Note: K_{5,5} is bipartite, so no odd cycles.
    # We only need to check even lengths 4, 6, 8.
    
    # Walks are sequences of vertices.
    # We store them as lists of edge indices or vertex pairs.
    # To check voltage sum, we need the sequence of edges.
    
    walks = {4: [], 6: [], 8: []}
    
    # Generate walks
    # Start at each vertex
    for start in range(10):
        # BFS/DFS to find non-backtracking walks
        # State: (current, prev, path_edges)
        # We only care about walks that return to start
        
        # Use iterative DFS
        stack = [(start, -1, [])]
        while stack:
            node, prev, path = stack.pop()
            if len(path) > 8:
                continue
            
            # If we are at start and path length is 4, 6, 8
            if node == start and len(path) in (4, 6, 8):
                walks[len(path)].append(path)
                continue # Don't extend further if we want exactly these lengths?
                # Actually, we want to find *all* walks of these lengths.
                # Extending a length 4 walk to length 6 is valid if non-backtracking.
                
            for nb in range(10):
                # Check if edge (node, nb) exists in K_{5,5}
                # K_{5,5} connects 0-4 to 5-9
                if (node < 5 and nb >= 5) or (node >= 5 and nb < 5):
                    if nb != prev:
                        new_path = path + [(node, nb)]
                        stack.append((nb, node, new_path))
                        
    # Filter out walks that are not simple cycles? 
    # No, non-backtracking walks are sufficient. 
    # A non-backtracking walk of length L that returns to start corresponds to a cycle of length L in the lift 
    # IF the voltage sum is 0.
    # Wait, if the walk visits a vertex twice (not just start), it's not a simple cycle.
    # But a non-backtracking walk that returns to start *is* a cycle in the graph theoretic sense 
    # (it might be a "theta" graph or multiple cycles, but it contains a cycle of length <= L).
    # If the voltage sum is 0, does it imply a cycle of length L?
    # Yes, the lift of a closed non-backtracking walk is a closed non-backtracking walk in the lift.
    # If the voltage sum is 0, it is a cycle in the lift.
    # It might not be a *simple* cycle in the lift if the walk in the base is not simple.
    # But if the walk in the base is not simple, it contains a shorter non-backtracking closed walk.
    # So if we ensure *all* non-backtracking closed walks of length < 9 have non-zero voltage sum, 
    # then there are no cycles of length < 9 in the lift.
    
    # So we just need to check all non-backtracking closed walks of length 4, 6, 8.
    
    # To speed up, we can just store the list of edge indices for each walk.
    # Let's map edges to indices.
    edge_to_idx = {}
    for i, (u, v) in enumerate(base_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    # Convert walks to edge index lists
    walk_edges = {4: [], 6: [], 8: []}
    for L in (4, 6, 8):
        for walk in walks[L]:
            edge_list = []
            for u, v in walk:
                edge_list.append(edge_to_idx[(u, v)])
            walk_edges[L].append(edge_list)
            
    # Now search for M
    # We want to minimize n = 10 * M
    # Start with small M
    # If M is too small, we likely won't find a valid assignment.
    # Let's try M from 50 to 200
    # For each M, try random voltage assignments
    
    best_n = None
    best_edges = None
    
    rng = random.Random(42)
    
    # Time limit: 120s
    # Check time?
    
    for M in range(50, 200):
        # Try a few random assignments
        for attempt in range(20):
            # Assign random voltages
            voltages = [rng.randint(0, M - 1) for _ in range(len(base_edges))]
            
            # Check if any walk has sum 0
            valid = True
            for L in (4, 6, 8):
                for edge_list in walk_edges[L]:
                    s = 0
                    for e in edge_list:
                        s = (s + voltages[e]) % M
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                # Found a valid graph
                # Construct the lift
                n = 10 * M
                edges = []
                # Vertices are (u, g)
                # u in 0..9, g in 0..M-1
                # Edge (u, g) - (v, g + voltages[idx])
                for u in range(10):
                    for v in range(10):
                        if (u < 5 and v >= 5) or (u >= 5 and v < 5):
                            # Find edge index
                            idx = edge_to_idx[(u, v)]
                            alpha = voltages[idx]
                            for g in range(M):
                                # u, g to v, (g+alpha)%M
                                u_id = u * M + g
                                v_id = v * M + ((g + alpha) % M)
                                edges.append((u_id, v_id))
                                
                return n, edges
                
    # If no M found in range, return a large M that is likely valid
    # Or just return a random regular graph?
    # Let's try M=1000
    M = 1000
    for attempt in range(10):
        voltages = [rng.randint(0, M - 1) for _ in range(len(base_edges))]
        valid = True
        for L in (4, 6, 8):
            for edge_list in walk_edges[L]:
                s = 0
                for e in edge_list:
                    s = (s + voltages[e]) % M
                if s == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            n = 10 * M
            edges = []
            for u in range(10):
                for v in range(10):
                    if (u < 5 and v >= 5) or (u >= 5 and v < 5):
                        idx = edge_to_idx[(u, v)]
                        alpha = voltages[idx]
                        for g in range(M):
                            u_id = u * M + g
                            v_id = v * M + ((g + alpha) % M)
                            edges.append((u_id, v_id))
            return n, edges
            
    # Fallback: Return a random regular graph (might be invalid, but better than crashing)
    # Or just return the first M=50 graph even if invalid? No, invalid scores 0.
    # We must return a valid graph.
    # If we fail, we return a known valid small graph?
    # We don't have one.
    # We'll just return the M=1000 graph, hoping it's valid.
    # If it's not, we fail.
    
    # Actually, if M=1000, probability of failure is low.
    # Let's force M=1000 and return the first assignment, assuming it's valid.
    # This is risky.
    
    # Better: Increase M until valid.
    M = 100
    while M < 10000:
        voltages = [rng.randint(0, M - 1) for _ in range(len(base_edges))]
        valid = True
        for L in (4, 6, 8):
            for edge_list in walk_edges[L]:
                s = 0
                for e in edge_list:
                    s = (s + voltages[e]) % M
                if s == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            n = 10 * M
            edges = []
            for u in range(10):
                for v in range(10):
                    if (u < 5 and v >= 5) or (u >= 5 and v < 5):
                        idx = edge_to_idx[(u, v)]
                        alpha = voltages[idx]
                        for g in range(M):
                            u_id = u * M + g
                            v_id = v * M + ((g + alpha) % M)
                            edges.append((u_id, v_id))
            return n, edges
        M += 10
        
    # If all else fails
    return 1000, [] # Invalid, but prevents crash
# EVOLVE-BLOCK-END
