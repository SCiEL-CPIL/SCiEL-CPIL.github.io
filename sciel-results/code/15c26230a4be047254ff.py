import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 16
    all_vertices = set(range(n))
    
    # Q3 edge list (8 vertices, 12 edges)
    q3_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]
    
    def has_q3_with_edge(adj_mask, u, v):
        """Check if adding edge u-v creates a Q3. adj_mask is list of int bitmasks."""
        neu = [x for x in range(n) if x != u and x != v and (adj_mask[u] >> x) & 1]
        nev = [x for x in range(n) if x != v and x != u and (adj_mask[v] >> x) & 1]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        for pos2 in neu:
            for pos4 in neu:
                if pos2 == pos4:
                    continue
                for pos3 in nev:
                    if pos3 == pos2 or pos3 == pos4:
                        continue
                    for pos5 in nev:
                        if pos5 == pos2 or pos5 == pos3 or pos5 == pos4:
                            continue
                        
                        if not ((adj_mask[pos2] >> pos3) & 1): continue
                        if not ((adj_mask[pos4] >> pos5) & 1): continue
                        
                        excluded = {u, v, pos2, pos3, pos4, pos5}
                        cand_pos6 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos2] >> x) & 1) and ((adj_mask[pos4] >> x) & 1):
                                cand_pos6.append(x)
                        
                        cand_pos7 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos3] >> x) & 1) and ((adj_mask[pos5] >> x) & 1):
                                cand_pos7.append(x)
                        
                        for pos6 in cand_pos6:
                            for pos7 in cand_pos7:
                                if pos6 == pos7: continue
                                if ((adj_mask[pos6] >> pos7) & 1) and \
                                   ((adj_mask[pos3] >> pos7) & 1) and \
                                   ((adj_mask[pos5] >> pos7) & 1):
                                    return True
        return False
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def verify_no_q3(adj_mask):
        """Verify the graph has no Q3 subgraph."""
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    if has_q3_with_edge(adj_mask, u, v):
                        return False
        return True
    
    def build_adj_matrix(adj_mask):
        """Convert adj_mask list to numpy adjacency matrix."""
        A = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    A[i, j] = 1
                    A[j, i] = 1
        return A
    
    def count_4cycles(A):
        """Count 4-cycles using trace(A^4) = 8 * (number of 4-cycles) + other terms.
        Actually, trace(A^4) counts closed walks of length 4.
        For a simple graph, the number of 4-cycles C4 satisfies:
        trace(A^4) = 8*C4 + 4*sum(d(v)^2 - d(v))
        But we just use trace(A^4) as a proxy - minimizing it reduces 4-cycles."""
        A2 = A @ A
        A4 = A2 @ A2
        return int(np.trace(A4))
    
    def spectral_remove_edge(adj_mask, A):
        """Find the edge whose removal most reduces trace(A^4). Return (u,v) or None."""
        best_reduction = 0
        best_edge = None
        
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    # Removing edge (i,j) from A
                    # New A' = A - E where E has 1s at (i,j) and (j,i)
                    # trace((A-E)^4) = trace(A^4) - 4*trace(A^3*E) + 6*trace(A^2*E^2) - 4*trace(A*E^3) + trace(E^4)
                    # This is complex. Simpler: just compute the difference for candidate edges.
                    # For efficiency, precompute A2 and A3.
                    pass
        
        # More efficient approach: for each edge (i,j), the change in trace(A^4) when removing it
        # trace(A^4) = sum_{a,b,c,d} A[a,b]*A[b,c]*A[c,d]*A[d,a]
        # When we remove edge (i,j), we subtract all terms where A[i,j] or A[j,i] appears.
        # This is equivalent to: the number of closed walks of length 4 that use edge (i,j) at least once.
        
        # Simpler heuristic: for edge (i,j), compute how many 4-cycles it participates in.
        # A 4-cycle uses edge (i,j) if there exists a path of length 3 from i to j that doesn't use (i,j).
        # Number of such paths = (A^3)[i,j] - (number of length-3 paths that use edge (i,j))
        # Length-3 paths from i to j using edge (i,j): go i->j->x->j or i->x->j->j (latter impossible in simple graph)
        # Actually: i->j->x->j requires x != i,j and A[j,x]=1, A[x,j]=1. Count = deg(j) - 1 (excluding i)
        # But we also need to consider i->x->j->y... no, path is i to j in 3 steps.
        # Paths of length 3 from i to j: i->a->b->j where a,b are intermediate.
        # Those using edge (i,j): i->j->b->j (b adjacent to j, b!=i) OR i->a->j->j (impossible).
        # Wait, i->j->b->j: this uses edges (i,j), (j,b), (b,j). That's a valid walk.
        # Count: number of b != i,j with A[j,b]=1 = deg(j) - 1 (since i is adjacent to j)
        # Hmm, but this counts walks, and some might repeat vertices.
        
        # Let's just use a simpler proxy: for each edge (i,j), compute (A^3)[i,j].
        # Higher (A^3)[i,j] means more 3-paths between i and j, which means removing (i,j) breaks more 4-cycles.
        
        A3 = A @ A @ A
        best_score = -1
        best_edge = None
        
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    score = int(A3[i, j])
                    if score > best_score:
                        best_score = score
                        best_edge = (i, j)
        
        return best_edge
    
    best_edge_count = 0
    best_adj_mask = None
    
    random.seed(42)
    
    # Phase 1: Spectral heuristic - start from K16, remove edges to reduce 4-cycles
    # This gives a good initial Q3-sparse graph
    
    # Start with complete graph K16
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
    
    A = build_adj_matrix(adj_mask)
    
    # Iteratively remove the edge that participates in the most 4-cycles
    max_removals = 50  # Remove at most 50 edges from K16
    removals = 0
    
    while removals < max_removals and time.time() < deadline - 10:
        edge = spectral_remove_edge(adj_mask, A)
        if edge is None:
            break
        
        u, v = edge
        # Remove the edge
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        A[u, v] = 0
        A[v, u] = 0
        removals += 1
        
        # Check if current graph is Q3-free
        if verify_no_q3(adj_mask):
            ec = count_edges(adj_mask)
            if ec > best_edge_count:
                best_edge_count = ec
                best_adj_mask = adj_mask[:]
    
    # Phase 2: Greedy edge addition from the spectrally-reduced graph
    if best_adj_mask is not None:
        adj_mask = best_adj_mask[:]
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if has_q3_with_edge(adj_mask, u, v):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        ec = count_edges(adj_mask)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = adj_mask[:]
    
    # Phase 3: Multiple random restarts with different starting points
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        adj_mask = [0] * n
        
        if restart == 0:
            # K3,13 structure
            for u in range(3):
                for v in range(3, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        elif restart == 1:
            # K8,8
            for u in range(8):
                for v in range(8, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        elif restart == 2:
            # K4,4,4,4 complete 4-partite
            for i in range(n):
                for j in range(i+1, n):
                    if (i // 4) != (j // 4):
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
        else:
            # Random start
            density = random.uniform(0.3, 0.7)
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            for i, j in all_pairs:
                if random.random() < density:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Clean up: remove edges that create Q3
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_mask[u] >> v) & 1:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        if has_q3_with_edge(adj_mask, u, v):
                            changed = True
                        else:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
        
        # Greedy add
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if has_q3_with_edge(adj_mask, u, v):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        ec = count_edges(adj_mask)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = adj_mask[:]
    
    # Final verification
    if best_adj_mask is None:
        best_adj_mask = [0] * n
        for u in range(3):
            for v in range(3, 16):
                best_adj_mask[u] |= (1 << v)
                best_adj_mask[v] |= (1 << u)
    
    if not verify_no_q3(best_adj_mask):
        best_adj_mask = [0] * n
        for u in range(3):
            for v in range(3, 16):
                best_adj_mask[u] |= (1 << v)
                best_adj_mask[v] |= (1 << u)
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
