"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Identify edge indices for top cycle, bottom cycle, and matching
    # Top cycle: (i, (i+1)%m)
    # Bottom cycle: (m+i, m+(i+1)%m)
    # Matching: (i, m+i)
    
    top_edge_indices = []
    bottom_edge_indices = []
    match_edge_indices = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edge_indices.append(idx)
        elif u >= m and v >= m:
            bottom_edge_indices.append(idx)
        else:
            match_edge_indices.append(idx)
            
    # Sort indices by position to ensure consistent ordering
    # The input edges might not be sorted by vertex index, but the structure is fixed.
    # We need to map the logical position (0 to m-1) to the edge index.
    
    def get_edge_index_for_type(type_name, pos):
        # This helper is not strictly needed if we just iterate all combinations of "free" edges
        pass

    # Strategy: Exhaustive search over switching classes.
    # A switching class is determined by the signs of edges NOT in a spanning tree.
    # Let's pick a spanning tree:
    # - All matching edges (m edges)
    # - All top cycle edges except the last one (m-1 edges)
    # - All bottom cycle edges except the last one (m-1 edges)
    # Total tree edges: m + (m-1) + (m-1) = 3m - 2.
    # Total edges: 3m.
    # Number of non-tree edges (free edges): 2.
    # Wait, $2^{E-V+1} = 2^{3m - 2m + 1} = 2^{m+1}$.
    # My spanning tree count: 
    # V = 2m. Tree edges should be V-1 = 2m-1.
    # My proposed tree: m (match) + (m-1) (top) + (m-1) (bottom) = 3m - 2.
    # 3m - 2 is NOT 2m - 1. 
    # The graph has cyclomatic number $\mu = E - V + 1 = 3m - 2m + 1 = m + 1$.
    # So there are $m+1$ independent cycles. 
    # The number of switching classes is $2^{m+1}$.
    
    # To enumerate them, we need to identify $m+1$ "free" edges whose signs determine the class.
    # A standard basis for the cycle space of a prism:
    # 1. The top cycle (1 dimension)
    # 2. The bottom cycle (1 dimension)
    # 3. The "ladder" cycles formed by two matching edges and two cycle edges. 
    #    There are m such cycles, but they are dependent. 
    #    Specifically, the product of all m "ladder" cycle signs equals (top cycle sign) * (bottom cycle sign).
    #    So there are m-1 independent ladder cycles.
    # Total independent cycles: 1 (top) + 1 (bottom) + (m-1) (ladders) = m + 1.
    
    # We can fix the signs of the "tree" edges to +1 and vary the signs of the "free" edges.
    # Which edges are free?
    # Let's define the free edges as:
    # - Top edge (m-1, 0)  [closes the top cycle]
    # - Bottom edge (m+m-1, m+0) [closes the bottom cycle]
    # - Matching edges? No, matching edges are in the tree.
    # - We need m-1 more free edges from the "ladders".
    #   The ladders are formed by matching edge i, matching edge i+1, top edge i, bottom edge i.
    #   If we keep matching edges and top/bottom edges (except the closing ones) in the tree, 
    #   the "free" edges corresponding to the ladders would be... 
    #   Actually, it's easier to just pick a spanning tree and mark the rest as free.
    
    # Spanning Tree Selection:
    # 1. All matching edges (indices in match_edge_indices)
    # 2. Top cycle edges 0 to m-2 (indices in top_edge_indices[0:m-1])
    # 3. Bottom cycle edges 0 to m-2 (indices in bottom_edge_indices[0:m-1])
    
    # We need to ensure top_edge_indices and bottom_edge_indices are sorted by position.
    # The problem states edges are an ordered list, but doesn't specify the order within the list.
    # However, the structure is "two m-cycles... joined by matching".
    # Usually: 
    # 0..m-1: Top cycle (0-1, 1-2, ..., m-1-0)
    # m..2m-1: Bottom cycle
    # 2m..3m-1: Matching
    # Let's robustly identify the specific edges.
    
    # Map edge index to (type, pos)
    # pos is the "start" vertex in the cycle order.
    
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Determine position
            # If v == (u+1)%m, pos = u
            # If u == (v+1)%m, pos = v
            if v == (u + 1) % m:
                pos = u
            else:
                pos = v
            edge_map[idx] = ('top', pos)
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = v_local
            edge_map[idx] = ('bottom', pos)
        else:
            # Matching
            # u < m, v >= m OR u >= m, v < m
            if u < m:
                pos = u
            else:
                pos = v - m
            edge_map[idx] = ('match', pos)

    # Identify Free Edges
    # Tree edges:
    # - All matches
    # - Top edges with pos in 0..m-2
    # - Bottom edges with pos in 0..m-2
    
    free_edge_indices = []
    for idx in range(n_edges):
        etype, pos = edge_map[idx]
        if etype == 'match':
            continue # In tree
        if etype == 'top' and pos < m - 1:
            continue # In tree
        if etype == 'bottom' and pos < m - 1:
            continue # In tree
        # This edge is NOT in the tree
        free_edge_indices.append(idx)
        
    # There should be exactly m+1 free edges.
    # 1 top closing (pos m-1), 1 bottom closing (pos m-1), and... wait.
    # My tree definition:
    # Matches: m edges.
    # Top: m-1 edges.
    # Bottom: m-1 edges.
    # Total tree: 3m - 2.
    # Total edges: 3m.
    # Free: 2.
    # This contradicts $2^{m+1}$. Why?
    # Because a spanning tree of a graph with V vertices has V-1 edges.
    # V = 2m. Tree size = 2m - 1.
    # My "tree" has 3m - 2 edges. This is NOT a tree. It contains cycles!
    # The matching edges + top path + bottom path form a structure.
    # Let's build a proper spanning tree.
    # Start with vertex 0.
    # Add matching 0- m.
    # Add top 0-1, 1-2, ..., m-2 - (m-1). (m-1 edges)
    # Now we have visited 0..m-1 and m.
    # From m, add matching m - (m+1)? No, matching is i - (i+m).
    # We have visited 0, ..., m-1, m.
    # We need to visit m+1, ..., 2m-1.
    # We can use bottom cycle edges: m - (m+1), (m+1) - (m+2), ..., (2m-2) - (2m-1).
    # That's m-1 edges.
    # Total edges in this tree: 1 (match 0) + (m-1) (top) + (m-1) (bottom) = 2m - 1.
    # Correct.
    
    # So, which edges are in THIS tree?
    # 1. Match edge for pos 0.
    # 2. Top edges for pos 0 to m-2.
    # 3. Bottom edges for pos 0 to m-2.
    
    # Free edges are:
    # 1. Match edges for pos 1 to m-1. (m-1 edges)
    # 2. Top edge for pos m-1. (1 edge)
    # 3. Bottom edge for pos m-1. (1 edge)
    # Total free: m - 1 + 1 + 1 = m + 1.
    # This matches $2^{m+1}$.
    
    free_indices = []
    for idx in range(n_edges):
        etype, pos = edge_map[idx]
        is_in_tree = False
        if etype == 'match' and pos == 0:
            is_in_tree = True
        elif etype == 'top' and pos < m - 1:
            is_in_tree = True
        elif etype == 'bottom' and pos < m - 1:
            is_in_tree = True
            
        if not is_in_tree:
            free_indices.append(idx)
            
    # Sort free_indices for deterministic iteration
    free_indices.sort()
    
    n_free = len(free_indices)
    # n_free should be m + 1
    
    def spectral_radius(signs):
        A = np.zeros((n_vertices, n_vertices))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        try:
            vals = np.linalg.eigvalsh(A)
            return np.max(np.abs(vals))
        except:
            return float('inf')

    def is_balanced(signs):
        adj = [[] for _ in range(n_vertices)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        sigma = [0] * n_vertices
        for i in range(n_vertices):
            if sigma[i] == 0:
                sigma[i] = 1
                queue = deque([i])
                while queue:
                    u = queue.popleft()
                    for v, s in adj[u]:
                        expected_sigma_v = sigma[u] * s
                        if sigma[v] == 0:
                            sigma[v] = expected_sigma_v
                            queue.append(v)
                        elif sigma[v] != expected_sigma_v:
                            return False
        return True

    best_signs = None
    best_radius = float('inf')
    
    # Iterate over all 2^(m+1) switching classes
    # We fix tree edges to +1, and vary free edges.
    # Note: This generates one representative per class.
    # The spectral radius is invariant under switching, so we just need to evaluate these representatives.
    
    num_classes = 1 << n_free
    
    for mask in range(num_classes):
        if time.time() > DEADLINE:
            break
            
        signs = np.ones(n_edges, dtype=int)
        
        # Set free edges based on mask
        for i, idx in enumerate(free_indices):
            if mask & (1 << i):
                signs[idx] = -1
            else:
                signs[idx] = 1
        
        # Tree edges are already +1 (initialized)
        
        # Check validity (connected lift = unbalanced signed graph)
        if is_balanced(signs):
            continue
            
        r = spectral_radius(signs)
        
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    # Fallback if no valid unbalanced graph found (should not happen for m>=4)
    if best_signs is None:
        # Random fallback
        rng = np.random.default_rng(3)
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=n_edges)
            if not is_balanced(signs):
                r = spectral_radius(signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
                if best_signs is not None:
                    break
                    
    if best_signs is None:
        # Absolute fallback: all +1 except one -1
        best_signs = np.ones(n_edges, dtype=int)
        best_signs[0] = -1
        
    return best_signs
# EVOLVE-BLOCK-END
