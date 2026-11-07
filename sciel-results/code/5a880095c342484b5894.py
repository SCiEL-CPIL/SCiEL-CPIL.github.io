import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    
    # Map edge tuple to index
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    # Define the spanning tree and chords
    # Tree edges:
    # Top path: (0,1), (1,2), ..., (m-2, m-1)
    # Bottom path: (m, m+1), ..., (2m-2, 2m-1)
    # Matching: (0, m)
    
    tree_edges = []
    for i in range(m - 1):
        tree_edges.append((i, i + 1))
        tree_edges.append((m + i, m + i + 1))
    tree_edges.append((0, m))
    
    # Chord edges:
    # Top closure: (m-1, 0)
    # Bottom closure: (2m-1, m)
    # Matching: (1, m+1), ..., (m-1, 2m-1)
    
    chord_edges = []
    chord_edges.append((m - 1, 0))
    chord_edges.append((2 * m - 1, m))
    for i in range(1, m):
        chord_edges.append((i, m + i))
        
    # Verify we have all edges
    all_logical_edges = tree_edges + chord_edges
    assert len(all_logical_edges) == len(edges), f"Edge count mismatch: {len(all_logical_edges)} vs {len(edges)}"
    
    # Get indices for tree and chords
    tree_indices = [edge_map[e] for e in tree_edges]
    chord_indices = [edge_map[e] for e in chord_edges]
    
    num_chords = len(chord_indices)
    # num_chords should be m + 1
    
    best_radius = float('inf')
    best_signs = None
    
    # Iterate over all 2^(m+1) switching classes
    # We fix tree edges to +1. The chords vary.
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        signs = np.ones(len(edges), dtype=int)
        
        # Set tree edges to +1 (already done by np.ones)
        
        # Set chord edges based on mask
        for i, idx in enumerate(chord_indices):
            if (mask >> i) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Check validity and compute radius
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            # Invalid lift (e.g., disconnected)
            continue
            
    if best_signs is None:
        # Fallback: return random signs if no valid lift found (should not happen)
        rng = np.random.default_rng(3)
        best_signs = rng.choice([-1, 1], size=len(edges))
        
    return best_signs
