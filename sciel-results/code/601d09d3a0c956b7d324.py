<<<<<<< SEARCH
import random
import time

def _has_q3(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains Q3 as subgraph.
    Returns True if found, False otherwise."""
    # Q3: 8 vertices, edges between vertices differing in exactly 1 bit
    # Vertices: 0,1,2,3,4,5,6,7 (binary 000 to 111)
    # Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    q3_vertices = list(range(8))
    
    # Backtracking: assign Q3 vertices to graph vertices
    # Use symmetry breaking: fix Q3 vertex 0 to graph vertex 0
    # Actually, let's be smarter. Fix Q3 vertex 0 to each possible graph vertex.
    
    def backtrack(assign, remaining):
        if not remaining:
            return True
        # Pick the next Q3 vertex to assign (heuristic: most constrained)
        # For simplicity, just take the first remaining
        qv = remaining[0]
        rest = remaining[1:]
        
        # Try all unassigned graph vertices
        for gv in range(n):
            if gv in assign.values():
                continue
            # Check edges from qv to already-assigned Q3 vertices
            ok = True
            for (a, b) in q3_edges:
                if a == qv and b in assign:
                    if not adj[gv][assign[b]]:
                        ok = False
                        break
                elif b == qv and a in assign:
                    if not adj[gv][assign[a]]:
                        ok = False
                        break
            if not ok:
                continue
            assign[qv] = gv
            if backtrack(assign, rest):
                return True
            del assign[qv]
        return False
    
    # Try fixing Q3 vertex 0 to each graph vertex
    for start in range(n):
        assign = {0: start}
        if backtrack(assign, list(range(1, 8))):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    n = 16
    
    # Use numpy for faster adjacency operations
    import numpy as np
    
    # Start with a random graph with a reasonable density
    # Let's start around 55-65 edges
    random.seed(42)
    
    best_edges = None
    best_count = 0
    
    # Try multiple random starts
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start with random graph
        edge_set = set()
        # Randomly add edges with probability ~0.5
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.55:
                    edge_set.add((i, j))
        
        # Build adjacency
        adj = [[0]*n for _ in range(n)]
        for (u, v) in edge_set:
            adj[u][v] = 1
            adj[v][u] = 1
        
        # Phase 1: Remove edges to eliminate Q3
        iters = 0
        while time.time() < deadline - 10 and iters < 500:
            if not _has_q3(adj, n):
                break
            # Find a Q3 and remove a random edge from it
            # For now, just randomly remove an edge
            # Better: find which edge to remove
            # Simple approach: remove random edge, recheck
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.remove(e)
                adj[e[0]][e[1]] = 0
                adj[e[1]][e[0]] = 0
            iters += 1
        
        if time.time() > deadline - 5:
            break
        
        # Phase 2: Try to add edges
        if not _has_q3(adj, n):
            added = True
            while added and time.time() < deadline - 5:
                added = False
                # Try adding random missing edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
                random.shuffle(missing)
                for (u, v) in missing:
                    if time.time() > deadline - 5:
                        break
                    edge_set.add((u, v))
                    adj[u][v] = 1
                    adj[v][u] = 1
                    if not _has_q3(adj, n):
                        added = True
                    else:
                        edge_set.remove((u, v))
                        adj[u][v] = 0
                        adj[v][u] = 0
        
        # Update best
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = list(edge_set)
    
    # If we didn't find anything, fall back to K3,13
    if best_edges is None or best_count <= 39:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return [[u, v] for (u, v) in best_edges]
=======
import random
import time

# Precompute Q3 structure
Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
Q3_N = 8
N_VERTS = 16
TOTAL_EDGES = N_VERTS * (N_VERTS - 1) // 2  # 120

def _build_q3_embeddings():
    """Precompute all Q3 embeddings in K16. Each embedding is a tuple of 8 vertex indices."""
    embeddings = []
    from itertools import permutations
    
    # Q3 vertices: 0..7, edges where Hamming distance = 1
    # For each injection of {0..7} into {0..15}, check if it's a Q3 embedding
    # This is expensive, so use a smarter approach
    
    # Actually, let's generate all 8-tuples and check
    # 16P8 = 16!/8! which is huge. Instead, use backtracking with pruning.
    
    # Better approach: enumerate all 8-subsets of {0..15}, then for each,
    # check all 8! bijections to Q3 vertices
    # C(16,8) = 12870, times 8! = 40320 = ~519M, still too many
    
    # Even better: use the fact that Q3 has specific structure.
    # Let's just do a direct search during the algorithm.
    return embeddings

def _check_q3_with_bitsets(adj_bits):
    """Check if graph contains Q3 using bitset adjacency.
    adj_bits[i] is a bitmask of neighbors of vertex i.
    Returns True if Q3 found, False otherwise."""
    n = N_VERTS
    all_mask = (1 << n) - 1
    
    # Backtracking with bitsets
    # assign: list of length 8, assign[qv] = gv or -1
    assign = [-1] * Q3_N
    
    def backtrack(qv_idx):
        if qv_idx == Q3_N:
            return True
        
        qv = qv_idx
        # Get neighbors of qv in Q3
        q3_neighbors = set()
        for (a, b) in Q3_EDGES:
            if a == qv:
                q3_neighbors.add(b)
            elif b == qv:
                q3_neighbors.add(a)
        
        # Get already-assigned neighbors
        assigned_neighbors = {}
        for qn in q3_neighbors:
            if assign[qn] != -1:
                assigned_neighbors[qn] = assign[qn]
        
        # Candidate vertices: must be adjacent to all assigned Q3-neighbors
        candidates = all_mask
        for qn, gv in assigned_neighbors.items():
            candidates &= adj_bits[gv]
        
        # Also must not be already used
        used_mask = 0
        for i in range(qv_idx):
            if assign[i] != -1:
                used_mask |= (1 << assign[i])
        candidates &= ~used_mask
        
        # Try each candidate
        while candidates:
            gv = (candidates & -candidates).bit_length() - 1
            candidates &= candidates - 1
            
            # Check non-edges: for Q3 non-neighbors that are already assigned,
            # gv must NOT be adjacent to them
            ok = True
            for qn in range(Q3_N):
                if qn == qv or assign[qn] == -1:
                    continue
                # Check if qn is a neighbor of qv in Q3
                is_neighbor = False
                for (a, b) in Q3_EDGES:
                    if (a == qv and b == qn) or (a == qn and b == qv):
                        is_neighbor = True
                        break
                if not is_neighbor:
                    # Must NOT be adjacent
                    if (adj_bits[gv] >> assign[qn]) & 1:
                        ok = False
                        break
            
            if not ok:
                continue
            
            assign[qv] = gv
            if backtrack(qv_idx + 1):
                return True
            assign[qv] = -1
        
        return False
    
    # Symmetry breaking: fix Q3 vertex 0 to vertex 0
    assign[0] = 0
    if backtrack(1):
        return True
    
    # Try other starting vertices for Q3 vertex 0
    for start in range(1, n):
        assign[0] = start
        if backtrack(1):
            return True
        assign[0] = -1
    
    return False

def _count_q3_violations(adj_bits):
    """Count number of Q3 subgraphs (approximate via checking).
    Returns 0 if no Q3, else positive number."""
    if not _check_q3_with_bitsets(adj_bits):
        return 0
    return 1  # We just need to know if any exist

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = N_VERTS
    random.seed(42)
    
    best_edge_set = None
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
    
    def _edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
    
    def _build_adj_bits(edge_set):
        """Build bitset adjacency from edge set."""
        adj_bits = [0] * n
        for (u, v) in edge_set:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits
    
    def _is_q3_free(edge_set):
        """Check if graph is Q3-free using bitset adjacency."""
        adj_bits = _build_adj_bits(edge_set)
        return not _check_q3_with_bitsets(adj_bits)
    
    # Phase 1: Find initial Q3-free graph
    # Start with dense random graphs and remove edges
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Start with random dense graph
        edge_set = set()
        density = 0.55 + 0.05 * random.random()
        for (i, j) in all_edges:
            if random.random() < density:
                edge_set.add((i, j))
        
        # Remove edges until Q3-free
        max_removals = 50
        for _ in range(max_removals):
            if time.time() > deadline - 10:
                break
            if not _check_q3_with_bitsets(_build_adj_bits(edge_set)):
                break
            # Remove random edge
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.remove(e)
        
        if not _check_q3_with_bitsets(_build_adj_bits(edge_set)):
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edge_set = set(edge_set)
    
    # If we didn't find anything good, use fallback
    if best_edge_set is None or best_count <= 39:
        # K3,13: 3 * 13 = 39 edges
        edge_set = set()
        for a in range(3):
            for b in range(3, 16):
                edge_set.add((a, b))
        best_edge_set = edge_set
        best_count = 39
    
    # Phase 2: Tabu search to add edges
    edge_set = set(best_edge_set)
    current_count = len(edge_set)
    tabu_list = {}  # edge -> iteration when it becomes non-tabu
    tabu_tenure = 20
    best_found = current_count
    best_edge_set_final = set(edge_set)
    iteration = 0
    
    # Precompute missing edges
    missing_edges = [e for e in all_edges if e not in edge_set]
    
    while time.time() < deadline - 5:
        iteration += 1
        
        # Remove expired tabu entries
        if iteration % 10 == 0:
            tabu_list = {e: t for e, t in tabu_list.items() if t + tabu_tenure > iteration}
        
        # Try to add edges (primary objective)
        added = False
        # Shuffle missing edges for diversity
        random.shuffle(missing_edges)
        
        for (u, v) in missing_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in edge_set:
                continue
            if (u, v) in tabu_list and tabu_list[(u, v)] > iteration:
                continue
            
            # Try adding edge
            edge_set.add((u, v))
            adj_bits = _build_adj_bits(edge_set)
            
            if not _check_q3_with_bitsets(adj_bits):
                # Success! Keep the edge
                added = True
                current_count += 1
                if current_count > best_found:
                    best_found = current_count
                    best_edge_set_final = set(edge_set)
                # Remove from missing
                missing_edges.remove((u, v))
                break
            else:
                # Revert
                edge_set.remove((u, v))
                tabu_list[(u, v)] = iteration
        
        # If no edge could be added, try removing an edge to enable future additions
        if not added:
            # Remove a random edge to break symmetry
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.remove(e)
                current_count -= 1
                # Add to missing
                if e not in missing_edges:
                    missing_edges.append(e)
    
    # Use the best found
    if best_edge_set_final and len(best_edge_set_final) >= best_count:
        best_edge_set = best_edge_set_final
        best_count = len(best_edge_set_final)
    
    return [[u, v] for (u, v) in best_edge_set]
>>>>>>> REPLACE
