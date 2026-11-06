"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m_edges = len(all_edges)  # 120

    # Precompute all Q3 subgraphs (sets of 12 edges that form a Q3)
    # A Q3 is the 3-dimensional hypercube: 8 vertices, 12 edges, 3-regular, bipartite with parts of size 4.
    # We enumerate all 8-vertex subsets and check if they induce a Q3.
    
    def is_q3_induced(vertices, adj_dict):
        """Check if the induced subgraph on vertices is isomorphic to Q3."""
        if len(vertices) != 8:
            return False
        verts = list(vertices)
        # Build adjacency within these vertices
        sub_adj = {v: set() for v in verts}
        for u in verts:
            for w in adj_dict.get(u, set()):
                if w in sub_adj:
                    sub_adj[u].add(w)
        # Check 3-regular
        for v in verts:
            if len(sub_adj[v]) != 3:
                return False
        # Check bipartite with parts of size 4
        color = {}
        for start in verts:
            if start not in color:
                color[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop()
                    for w in sub_adj[u]:
                        if w not in color:
                            color[w] = 1 - color[u]
                            queue.append(w)
                        elif color[w] == color[u]:
                            return False
        part0 = [v for v in verts if color[v] == 0]
        part1 = [v for v in verts if color[v] == 1]
        if len(part0) != 4 or len(part1) != 4:
            return False
        # Count edges
        edge_count = sum(len(sub_adj[v]) for v in verts) // 2
        if edge_count != 12:
            return False
        # For 8-vertex 3-regular bipartite graphs with parts of size 4,
        # there are only a few. Let's verify it's Q3 by checking girth or just trust it.
        # Actually, let's do a quick check: Q3 has girth 4.
        # The only other 3-regular bipartite graph on 8+8 vertices with parts of size 4
        # would be K4,4 minus a perfect matching? No, K4,4 is 4-regular.
        # A 3-regular bipartite graph on 8 vertices (4+4) must be Q3.
        # Proof: The bipartite adjacency matrix is 4x4 with row/col sums 3.
        # This means each row has exactly one 0. The pattern of zeros determines the graph.
        # Up to isomorphism, there's essentially one such graph: Q3.
        # So we can trust this check.
        return True

    # Build full adjacency for K16
    full_adj = {v: set() for v in range(n)}
    for u, v in all_edges:
        full_adj[u].add(v)
        full_adj[v].add(u)

    q3_subgraphs = []
    for subset in itertools.combinations(range(n), 8):
        if is_q3_induced(subset, full_adj):
            # Extract the 12 edges
            edges_in_sub = []
            subset_set = set(subset)
            for u in subset:
                for w in full_adj[u]:
                    if w in subset_set and u < w:
                        edges_in_sub.append((u, w))
            q3_subgraphs.append(frozenset(edges_in_sub))
    
    # Deduplicate (shouldn't have duplicates since each 8-subset is unique)
    q3_subgraphs = list(set(q3_subgraphs))
    num_q3 = len(q3_subgraphs)
    
    # Build conflict structure: for each edge, which Q3s contain it
    edge_to_q3s = [[] for _ in range(m_edges)]
    for q3_idx, q3 in enumerate(q3_subgraphs):
        for e in q3:
            edge_to_q3s[edge_index[e]].append(q3_idx)
    
    # For fast checking: for each Q3, store its edges as a bitmask (120 bits)
    # Using Python integers as bitmasks
    q3_bitmasks = []
    for q3 in q3_subgraphs:
        mask = 0
        for e in q3:
            mask |= (1 << edge_index[e])
        q3_bitmasks.append(mask)
    
    # Strategy: Greedy construction with local search
    # Start with a random dense graph, then fix violations, then add edges back.
    
    best_edge_count = 0
    best_edges = None
    
    def count_edges(edge_mask):
        return bin(edge_mask).count('1')
    
    def check_q3_violation(edge_mask):
        """Check if any Q3 is fully contained in edge_mask. Returns the index of the first violating Q3 or -1."""
        for q3_idx, q3_mask in enumerate(q3_bitmasks):
            if (edge_mask & q3_mask) == q3_mask:
                return q3_idx
        return -1
    
    def count_q3_violations(edge_mask):
        """Count how many Q3s are fully contained in edge_mask."""
        count = 0
        for q3_mask in q3_bitmasks:
            if (edge_mask & q3_mask) == q3_mask:
                count += 1
        return count
    
    # Greedy: start with all edges, remove edges to eliminate Q3s
    def greedy_fix(edge_mask):
        """Iteratively remove edges to eliminate all Q3 violations."""
        violations = count_q3_violations(edge_mask)
        while violations > 0:
            # Find a violating Q3
            viol_idx = check_q3_violation(edge_mask)
            if viol_idx < 0:
                break
            # Remove one edge from this Q3. Choose the edge that is in the most other Q3s.
            q3 = q3_subgraphs[viol_idx]
            best_edge_to_remove = -1
            best_score = -1
            for e in q3:
                e_idx = edge_index[e]
                if (edge_mask >> e_idx) & 1:
                    # Count how many Q3s contain this edge
                    score = len(edge_to_q3s[e_idx])
                    if score > best_score:
                        best_score = score
                        best_edge_to_remove = e_idx
            if best_edge_to_remove >= 0:
                edge_mask &= ~(1 << best_edge_to_remove)
                violations -= 1  # approximate, will recount
            else:
                break
        return edge_mask
    
    # Local search: try to add edges back
    def local_search(edge_mask, time_limit):
        """Try to add edges to improve the solution."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each missing edge
            for i in range(m_edges):
                if time.time() > time_limit:
                    break
                if (edge_mask >> i) & 1:
                    continue
                # Check if adding edge i creates a Q3
                new_mask = edge_mask | (1 << i)
                # Quick check: does this create any Q3?
                creates_q3 = False
                for q3_idx in edge_to_q3s[i]:
                    q3_mask = q3_bitmasks[q3_idx]
                    if (new_mask & q3_mask) == q3_mask:
                        creates_q3 = True
                        break
                if not creates_q3:
                    edge_mask = new_mask
                    improved = True
        return edge_mask
    
    # Run multiple restarts with different random initial states
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start with a random dense graph
        if restart < 5:
            # First few: start with all edges and fix
            edge_mask = (1 << m_edges) - 1
        else:
            # Random start with varying densities
            density = random.uniform(0.3, 0.9)
            edge_mask = 0
            for i in range(m_edges):
                if random.random() < density:
                    edge_mask |= (1 << i)
        
        # Fix violations
        edge_mask = greedy_fix(edge_mask)
        
        # Local search to add edges back
        edge_mask = local_search(edge_mask, deadline - 5)
        
        # Verify no Q3 violations
        if count_q3_violations(edge_mask) == 0:
            ec = count_edges(edge_mask)
            if ec > best_edge_count:
                best_edge_count = ec
                best_edges = edge_mask
    
    # If we found a good solution, use it. Otherwise, fall back to a simple greedy.
    if best_edges is None:
        # Simple greedy: add edges one by one if they don't create a Q3
        best_edges = 0
        for i in range(m_edges):
            new_mask = best_edges | (1 << i)
            creates_q3 = False
            for q3_idx in edge_to_q3s[i]:
                q3_mask = q3_bitmasks[q3_idx]
                if (new_mask & q3_mask) == q3_mask:
                    creates_q3 = True
                    break
            if not creates_q3:
                best_edges = new_mask
    
    # Convert bitmask to list of edges
    result = []
    for i in range(m_edges):
        if (best_edges >> i) & 1:
            result.append(list(all_edges[i]))
    
    return result
# EVOLVE-BLOCK-END
