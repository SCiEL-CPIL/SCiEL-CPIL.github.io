"""Q3-free graph on 16 vertices - maximum edge count via randomized local search."""
import random
import time

def _adj_matrix_from_edges(edges, n=16):
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def _find_q3(adj, n=16):
    """
    Check if graph with adjacency matrix adj contains a subgraph isomorphic to Q3.
    Q3 has 8 vertices and 12 edges.
    Returns True if Q3 is found, False otherwise.
    """
    # Q3 structure: vertices can be labeled 0..7, edges between i,j if popcount(i xor j)==1
    # Precompute Q3 edges
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # Try all mappings of Q3 vertices to graph vertices
    # This is expensive: 16!/(16-8)! possibilities, but we can prune
    # Use backtracking with constraint propagation
    
    def backtrack(assignment, used):
        """
        assignment: list of length 8, -1 if unassigned
        used: set of graph vertices already used
        """
        # Find next unassigned Q3 vertex
        next_q = -1
        for i in range(8):
            if assignment[i] == -1:
                next_q = i
                break
        if next_q == -1:
            # All assigned - verify all Q3 edges map to graph edges
            for u, v in q3_edges:
                gu, gv = assignment[u], assignment[v]
                if adj[gu][gv] != 1:
                    return False
            return True
        
        # Try each unused graph vertex for this Q3 vertex
        for gv in range(n):
            if gv in used:
                continue
            # Check constraints with already-assigned Q3 neighbors
            valid = True
            for qv in range(8):
                if assignment[qv] != -1:
                    # If qv and next_q are adjacent in Q3, then assignment[qv] and gv must be adjacent in G
                    # If qv and next_q are NOT adjacent in Q3, assignment[qv] and gv must NOT be adjacent (for induced subgraph)
                    # But we want subgraph (not necessarily induced), so we only need: if edge in Q3, must be edge in G
                    is_adj_q3 = any((a==qv and b==next_q) or (a==next_q and b==qv) for a,b in q3_edges)
                    if is_adj_q3:
                        if adj[assignment[qv]][gv] != 1:
                            valid = False
                            break
            if not valid:
                continue
            
            assignment[next_q] = gv
            used.add(gv)
            if backtrack(assignment, used):
                return True
            assignment[next_q] = -1
            used.remove(gv)
        
        return False
    
    # Try with a few random orderings of graph vertices for the first Q3 vertex
    # Actually, let's just try all 16 choices for the first vertex
    for first_gv in range(n):
        assignment = [-1]*8
        assignment[0] = first_gv
        used = {first_gv}
        if backtrack(assignment, used):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    # Start with K3,13: 39 edges
    initial_edges = [[a, b] for a in range(3) for b in range(3, n)]
    best_edges = list(initial_edges)
    best_count = len(best_edges)
    
    # Build adjacency matrix
    adj = _adj_matrix_from_edges(best_edges, n)
    
    # All possible edges not in current graph
    all_possible = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                all_possible.append((i, j))
    
    random.shuffle(all_possible)
    
    # Try to add edges one at a time
    for u, v in all_possible:
        if time.time() > deadline - 5.0:
            break
        
        # Check if adding this edge creates a Q3
        adj[u][v] = 1
        adj[v][u] = 1
        
        has_q3 = _find_q3(adj, n)
        
        if not has_q3:
            best_edges.append([u, v])
            best_count += 1
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
    
    # Now try local search: swap edges
    # Represent current edge set for fast lookup
    edge_set = set((min(u,v), max(u,v)) for u,v in best_edges)
    adj = _adj_matrix_from_edges(best_edges, n)
    
    # Generate candidate swaps
    current_edges = list(edge_set)
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) not in edge_set:
                missing_edges.append((i,j))
    
    random.shuffle(current_edges)
    random.shuffle(missing_edges)
    
    max_swaps = min(len(current_edges), 50)
    for _ in range(max_swaps):
        if time.time() > deadline - 5.0:
            break
        
        rem_edge = random.choice(current_edges)
        add_edge = random.choice(missing_edges)
        
        u1, v1 = rem_edge
        u2, v2 = add_edge
        
        # Remove rem_edge
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        # Add add_edge
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        
        if not _find_q3(adj, n):
            # Accept the swap
            edge_set.remove(rem_edge)
            edge_set.add(add_edge)
            current_edges.remove(rem_edge)
            if add_edge not in current_edges:
                current_edges.append(add_edge)
            missing_edges.remove(add_edge)
        else:
            # Revert
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            adj[u2][v2] = 0
            adj[v2][u2] = 0
    
    # Convert back to list format
    result = [[u, v] for u, v in sorted(edge_set)]
    return result
