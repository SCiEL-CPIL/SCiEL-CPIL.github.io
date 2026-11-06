"""Find a dense Q3-free graph on 16 vertices via edge removal from K16."""
import time
import random
from itertools import combinations

# Deadline guard: must return well before 150s wall-clock.
_DEADLINE = 140.0  # safety margin

def _is_q3_free(adj, n=16):
    """Check if graph with adjacency matrix adj has no Q3 subgraph.
    A Q3 subgraph is a set of 8 vertices where each vertex has degree 3
    within the induced subgraph (i.e., the induced subgraph is exactly Q3).
    Actually, we need to check for a subgraph isomorphic to Q3, which means
    we need 8 vertices and 12 specific edges forming Q3.
    
    More precisely: a subgraph isomorphic to Q3 exists if there's a set of
    8 vertices and a bijection to {0,1,2,3,4,5,6,7} (cubing) such that
    for each pair of cube vertices that are adjacent in Q3, the corresponding
    graph vertices are adjacent.
    
    Q3 vertices: 000,001,010,011,100,101,110,111
    Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    """
    # Q3 adjacency: vertex i connects to i^1, i^2, i^4 (bit flips)
    q3_edges = []
    for i in range(8):
        for bit in [0, 1, 2]:
            j = i ^ (1 << bit)
            if i < j:
                q3_edges.append((i, j))
    
    # Try all 8-vertex subsets
    for subset in combinations(range(n), 8):
        # Check if there's an isomorphism from Q3 to induced subgraph
        # We need to find a permutation of subset that maps Q3 edges to graph edges
        # This is expensive; use a smarter check.
        
        # For each assignment of the 8 Q3 vertices to the 8 graph vertices,
        # check if all 12 Q3 edges are present.
        # There are 8! = 40320 permutations per subset, and C(16,8)=12870 subsets.
        # Total: 12870 * 40320 ≈ 519 million - too slow.
        
        # Better approach: use the fact that Q3 is bipartite with parts of size 4.
        # In Q3, vertices with even parity (000,011,101,110) form one part,
        # vertices with odd parity (001,010,111,100) form the other.
        # Each vertex has degree 3.
        
        # Alternative: check all 8-subsets, and for each, check if the induced
        # subgraph contains Q3 as a subgraph (not necessarily induced).
        # A subgraph isomorphic to Q3 means there exist 8 vertices and 12 edges
        # forming Q3. The 12 edges must be a subset of the induced subgraph edges.
        
        # For a given 8-vertex subset, the induced subgraph has some edges.
        # We need to check if Q3 is a subgraph of this induced subgraph.
        # This is equivalent to: is there a labeling of the 8 vertices such that
        # all 12 Q3 edges are present in the induced subgraph?
        
        # Let's use a backtracking approach for the isomorphism check.
        # Assign Q3 vertices 0..7 to graph vertices in the subset.
        
        # Precompute which pairs in the subset are connected
        subset_list = list(subset)
        conn = [[False]*8 for _ in range(8)]
        for a in range(8):
            for b in range(a+1, 8):
                if adj[subset_list[a]][subset_list[b]]:
                    conn[a][b] = True
                    conn[b][a] = True
        
        # Now check if Q3 is a subgraph of this 8-vertex graph.
        # Use backtracking: assign Q3 vertices to positions 0..7.
        # Prune: vertex 0 of Q3 must map to a vertex with degree >= 3 in the induced subgraph.
        
        # Count degrees in the induced subgraph
        deg = [0]*8
        for a in range(8):
            for b in range(8):
                if conn[a][b]:
                    deg[a] += 1
        
        # All vertices in Q3 have degree 3, so all mapped vertices must have deg >= 3
        if any(d < 3 for d in deg):
            continue
        
        # Backtracking search
        assignment = [-1]*8  # q3_vertex -> position in subset
        pos_assigned = [False]*8  # position -> assigned?
        
        def backtrack(qv):
            if qv == 8:
                # Check all Q3 edges
                for (u, v) in q3_edges:
                    pu = assignment[u]
                    pv = assignment[v]
                    if not conn[pu][pv]:
                        return False
                return True
            
            # Choose a position for qv
            # Symmetry breaking: if qv == 0, only try position 0 (fix vertex 0 of Q3 to position 0)
            # Actually, we can fix Q3 vertex 0 to position 0 to break symmetry.
            if qv == 0:
                candidates = [0]
            else:
                candidates = [p for p in range(8) if not pos_assigned[p]]
            
            for p in candidates:
                # Pruning: check that all already-assigned neighbors of qv in Q3
                # are connected to p in the graph
                ok = True
                for bit in [0, 1, 2]:
                    nb = qv ^ (1 << bit)
                    if nb < qv:  # already assigned
                        if not conn[p][assignment[nb]]:
                            ok = False
                            break
                if not ok:
                    continue
                
                # Also check: for unassigned neighbors, ensure there's capacity
                # (each unassigned neighbor needs to connect to p)
                # This is a soft constraint; skip for now.
                
                assignment[qv] = p
                pos_assigned[p] = True
                
                if backtrack(qv + 1):
                    return True
                
                assignment[qv] = -1
                pos_assigned[p] = False
            
            return False
        
        if backtrack(0):
            return False  # Found Q3
    
    return True  # No Q3 found


def _count_q3s(adj, n=16):
    """Count the number of Q3 subgraphs."""
    q3_edges = []
    for i in range(8):
        for bit in [0, 1, 2]:
            j = i ^ (1 << bit)
            if i < j:
                q3_edges.append((i, j))
    
    count = 0
    for subset in combinations(range(n), 8):
        subset_list = list(subset)
        conn = [[False]*8 for _ in range(8)]
        for a in range(8):
            for b in range(a+1, 8):
                if adj[subset_list[a]][subset_list[b]]:
                    conn[a][b] = True
                    conn[b][a] = True
        
        deg = [0]*8
        for a in range(8):
            for b in range(8):
                if conn[a][b]:
                    deg[a] += 1
        if any(d < 3 for d in deg):
            continue
        
        assignment = [-1]*8
        pos_assigned = [False]*8
        found = 0
        
        def backtrack(qv):
            nonlocal found
            if qv == 8:
                for (u, v) in q3_edges:
                    if not conn[assignment[u]][assignment[v]]:
                        return
                found += 1
                return
            
            if qv == 0:
                candidates = [0]
            else:
                candidates = [p for p in range(8) if not pos_assigned[p]]
            
            for p in candidates:
                ok = True
                for bit in [0, 1, 2]:
                    nb = qv ^ (1 << bit)
                    if nb < qv:
                        if not conn[p][assignment[nb]]:
                            ok = False
                            break
                if not ok:
                    continue
                
                assignment[qv] = p
                pos_assigned[p] = True
                backtrack(qv + 1)
                assignment[qv] = -1
                pos_assigned[p] = False
        
        backtrack(0)
        count += found
        if count > 0:
            break  # Early exit: we just need to know if any exist for removal
    return count


def construct():
    start_time = time.time()
    n = 16
    
    # Start with K16
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = True
            adj[j][i] = True
    
    edge_count = n*(n-1)//2  # 120
    
    # Get all edges as a list
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Greedy removal: repeatedly find a Q3 and remove an edge from it
    # Use randomization to escape local minima
    rng = random.Random(42)
    
    best_edges = list(all_edges)
    best_adj = [row[:] for row in adj]
    best_count = edge_count
    
    # Time budget for main search
    max_time = 130.0
    
    # First pass: greedy removal
    while True:
        if time.time() - start_time > max_time:
            break
        
        # Check if current graph is Q3-free
        if _is_q3_free(adj, n):
            break
        
        # Find a Q3 to break
        q3_edges = []
        for i in range(8):
            for bit in [0, 1, 2]:
                j = i ^ (1 << bit)
                if i < j:
                    q3_edges.append((i, j))
        
        found_q3 = None
        for subset in combinations(range(n), 8):
            if time.time() - start_time > max_time:
                break
            subset_list = list(subset)
            conn = [[False]*8 for _ in range(8)]
            for a in range(8):
                for b in range(a+1, 8):
                    if adj[subset_list[a]][subset_list[b]]:
                        conn[a][b] = True
                        conn[b][a] = True
            
            deg = [0]*8
            for a in range(8):
                for b in range(8):
                    if conn[a][b]:
                        deg[a] += 1
            if any(d < 3 for d in deg):
                continue
            
            assignment = [-1]*8
            pos_assigned = [False]*8
            
            def backtrack(qv):
                if qv == 8:
                    for (u, v) in q3_edges:
                        if not conn[assignment[u]][assignment[v]]:
                            return False
                    return True
                
                if qv == 0:
                    candidates = [0]
                else:
                    candidates = [p for p in range(8) if not pos_assigned[p]]
                
                for p in candidates:
                    ok = True
                    for bit in [0, 1, 2]:
                        nb = qv ^ (1 << bit)
                        if nb < qv:
                            if not conn[p][assignment[nb]]:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    assignment[qv] = p
                    pos_assigned[p] = True
                    if backtrack(qv + 1):
                        return True
                    assignment[qv] = -1
                    pos_assigned[p] = False
                
                return False
            
            if backtrack(0):
                # Found a Q3; get the actual graph edges
                q3_graph_edges = set()
                for (u, v) in q3_edges:
                    gu = subset_list[assignment[u]]
                    gv = subset_list[assignment[v]]
                    if gu > gv:
                        gu, gv = gv, gu
                    q3_graph_edges.add((gu, gv))
                found_q3 = list(q3_graph_edges)
                break
        
        if found_q3 is None:
            # No Q3 found (shouldn't happen if _is_q3_free said False)
            break
        
        # Randomly pick an edge from the Q3 to remove
        edge_to_remove = rng.choice(found_q3)
        u, v = edge_to_remove
        adj[u][v] = False
        adj[v][u] = False
        edge_count -= 1
        
        # Update best
        if edge_count > best_count:
            best_count = edge_count
            best_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            best_adj = [row[:] for row in adj]
    
    # Check if we have a valid Q3-free graph
    if not _is_q3_free(best_adj, n):
        # Fallback: use the seed solution
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i][j]]
