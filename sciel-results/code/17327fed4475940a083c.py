import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START

def _generate_supports():
    """Generate all 4-subsets of range(D) as tuples."""
    supports = list(itertools.combinations(range(D), 4))
    return supports

def _build_support_conflict_graph(supports):
    """Build adjacency matrix where edge exists if intersection size is 3."""
    n = len(supports)
    # Precompute sets for faster intersection
    supp_sets = [set(s) for s in supports]
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if len(supp_sets[i].intersection(supp_sets[j])) == 3:
                adj[i, j] = True
                adj[j, i] = True
    return adj

def _find_mis_supports(adj, time_limit=5.0):
    """Find Maximum Independent Set in the support conflict graph.
    Since N=330, we can use a robust greedy + local search."""
    n = adj.shape[0]
    start_time = time.time()
    
    best_size = 0
    best_set = set()
    
    # Multiple restarts with different random orderings
    rng = np.random.default_rng(42)
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        alive = np.ones(n, dtype=bool)
        selected = []
        
        # Greedy: pick node with min degree among alive, or random
        # For MIS, picking min degree is often good.
        # But let's try random permutation first, then refine.
        order = rng.permutation(n)
        
        for idx in order:
            if not alive[idx]:
                continue
            selected.append(idx)
            # Remove neighbors
            neighbors = np.where(adj[idx])[0]
            to_remove = np.unique(np.concatenate(([idx], neighbors)))
            alive[to_remove] = False
            
        current_size = len(selected)
        
        # Local Search: 1-swap
        # Try to swap one selected node with an unselected one to increase size?
        # No, 1-swap doesn't change size.
        # We need to try to add nodes.
        # Standard MIS local search:
        # 1. Try to add any unselected node that doesn't conflict with selected.
        # 2. If none, try to swap: remove 1, add 2? (2-swap)
        
        selected_set = set(selected)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Check if we can add any node
            # A node u can be added if it has no neighbors in selected_set
            # Precompute neighbors for speed?
            # N is small (330), so O(N^2) check is fine.
            
            added_any = False
            for u in range(n):
                if u in selected_set:
                    continue
                # Check conflicts with selected
                conflict = False
                for v in selected_set:
                    if adj[u, v]:
                        conflict = True
                        break
                if not conflict:
                    selected_set.add(u)
                    added_any = True
                    improved = True
                    # Break and restart loop? Or continue?
                    # Continue to add more if possible
                    # But adding one might block others. 
                    # Greedy add is fine.
            
            if not added_any:
                # Try 2-swap: remove one, see if we can add two
                # This is expensive. 
                # Given N=330, the greedy + add is likely very good.
                # Let's try a simple 1-removal-2-addition heuristic if time permits
                pass
        
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_set = set(selected_set)
            
    return list(best_set)

def _construct_vectors_from_supports(selected_supports):
    """Construct the 16 vectors for each selected support."""
    vectors = []
    for support in selected_supports:
        # Support is a tuple of 4 indices
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _generate_w1_vectors():
    """Generate the 22 weight-1 vectors (norm 2)."""
    vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _check_validity(vectors):
    """Verify the condition: min pairwise distance >= max norm."""
    if len(vectors) == 0:
        return 0
    if len(vectors) == 1:
        return 1
        
    norms_sq = np.sum(vectors ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    n = len(vectors)
    # Check all pairs
    # To be fast, we can use matrix multiplication
    # dist_sq = ||u||^2 + ||v||^2 - 2 u.v
    # We need min dist_sq >= max_norm_sq
    # i.e., ||u||^2 + ||v||^2 - 2 u.v >= max_norm_sq
    
    # Since all norms are likely 4 (for W4) or 4 (for W1), max_norm_sq = 4.
    # So we need ||u||^2 + ||v||^2 - 2 u.v >= 4.
    # If norms are 4, then 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
    
    dots = vectors @ vectors.T
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dots[i,j]
    # We only need to check off-diagonal
    # Let's compute min dist_sq
    
    # Efficient check:
    # min_dist_sq = min_{i!=j} (norms_sq[i] + norms_sq[j] - 2*dots[i,j])
    
    # Since N is ~500, N^2 is 250,000. Very fast.
    min_dist_sq = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = norms_sq[i] + norms_sq[j] - 2 * dots[i, j]
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                if min_dist_sq < max_norm_sq:
                    return 0 # Invalid
                    
    return n

def construct():
    start_time = time.time()
    
    # 1. Generate supports
    supports = _generate_supports()
    
    # 2. Build conflict graph for supports (overlap 3)
    adj = _build_support_conflict_graph(supports)
    
    # 3. Find MIS of supports
    selected_supports = _find_mis_supports(adj, time_limit=10.0)
    
    # 4. Construct W4 vectors
    w4_vectors = _construct_vectors_from_supports(selected_supports)
    
    # 5. Construct W1 vectors
    w1_vectors = _generate_w1_vectors()
    
    # 6. Combine
    final_vectors = np.vstack([w4_vectors, w1_vectors])
    
    # 7. Verify
    count = _check_validity(final_vectors)
    
    if count == 0:
        # Fallback if something went wrong (should not happen)
        # Return a small valid set
        return _generate_w1_vectors()
        
    return final_vectors

# EVOLVE-BLOCK-END
