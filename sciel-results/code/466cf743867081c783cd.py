import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Identify edge types and indices
    # Vertices: 0..m-1 (top), m..2m-1 (bottom)
    # Edges:
    # 1. Top cycle: (i, i+1) for i=0..m-2, and (m-1, 0)
    # 2. Bottom cycle: (m+i, m+i+1) for i=0..m-2, and (m+m-1, m)
    # 3. Matching: (i, m+i) for i=0..m-1
    
    # We need to map edge indices to types.
    # Let's just iterate through edges and classify them.
    
    # Define the tree edges (indices)
    # Tree:
    # - All matching edges: (i, m+i)
    # - Top cycle edges except (m-1, 0): (i, i+1) for i=0..m-2
    
    tree_edge_indices = []
    chord_edge_indices = []
    
    # We need to know which edge is which.
    # The problem statement says "ordered list of (u, v)".
    # Usually, for such tasks, the order might be specific, but we should be robust.
    # Let's assume we can identify them by vertex pairs.
    
    # Create a set of vertex pairs for quick lookup
    # Note: edges are undirected, so (u,v) is same as (v,u).
    # Normalize: (min(u,v), max(u,v))
    
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_map[key] = i

    # Identify tree edges
    # 1. Matching: (i, m+i)
    for i in range(m):
        u, v = i, m + i
        key = (min(u, v), max(u, v))
        if key in edge_map:
            tree_edge_indices.append(edge_map[key])
        else:
            # Fallback if edge order is different? 
            # The contract says "joined by the matching (i, i + m)".
            # So this should exist.
            pass

    # 2. Top cycle: (i, i+1) for i=0..m-2
    for i in range(m - 1):
        u, v = i, i + 1
        key = (min(u, v), max(u, v))
        if key in edge_map:
            tree_edge_indices.append(edge_map[key])
        else:
            pass

    # Chords are all other edges
    all_indices = set(range(n_edges))
    tree_set = set(tree_edge_indices)
    chord_indices = [i for i in all_indices if i not in tree_set]
    
    # There should be m+1 chords:
    # 1. Top edge (m-1, 0)
    # 2. Bottom edges (m+i, m+i+1) for i=0..m-1 (wrapping around)
    
    n_chords = len(chord_indices)
    # n_chords should be m+1
    
    # Precompute base sign vector for tree edges (+1)
    # We will vary chord signs.
    
    def is_unbalanced(signs):
        # signs: array of length n_edges
        # Check if there is a cycle with product -1
        # BFS with potentials
        potential = np.zeros(n_vertices, dtype=int)
        potential[0] = 1
        visited = np.zeros(n_vertices, dtype=bool)
        visited[0] = True
        queue = [0]
        
        # Build adjacency list with signs for this specific signing
        # This is slightly expensive if done inside the loop, but n is small.
        # We can optimize by precomputing adjacency structure (u, v, idx)
        
        adj = [[] for _ in range(n_vertices)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        while queue:
            u = queue.pop(0)
            for v, s in adj[u]:
                expected_v_potential = potential[u] * s
                if not visited[v]:
                    potential[v] = expected_v_potential
                    visited[v] = True
                    queue.append(v)
                else:
                    # Check consistency
                    if potential[v] != expected_v_potential:
                        return True # Unbalanced
        
        return False # Balanced

    def spectral_radius(signs):
        A = np.zeros((n_vertices, n_vertices))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigs = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigs))

    best_signs = None
    best_radius = float('inf')
    
    # Enumerate 2^(m+1) combinations of chord signs
    # m is at most 9, so m+1 is at most 10. 2^10 = 1024.
    # This is very fast.
    
    for k in range(2 ** n_chords):
        if time.time() - start_time > DEADLINE:
            break
            
        # Decode k into signs for chords
        # Bit 0 corresponds to chord_indices[0], etc.
        signs = np.ones(n_edges, dtype=int)
        
        for j in range(n_chords):
            idx = chord_indices[j]
            if (k >> j) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Tree edges are already +1 from initialization
        
        # Check validity (unbalanced)
        if not is_unbalanced(signs):
            continue
            
        # Compute radius
        radius = spectral_radius(signs)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    if best_signs is None:
        # Fallback: should not happen, but provide a valid unbalanced signing
        # e.g. set one chord to -1
        signs = np.ones(n_edges, dtype=int)
        if n_chords > 0:
            signs[chord_indices[0]] = -1
        else:
            # If no chords (should not happen for m>=2), flip a tree edge
            signs[0] = -1
        best_signs = signs

    return best_signs
