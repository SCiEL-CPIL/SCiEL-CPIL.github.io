import itertools
import random
from collections import deque

def _edge_index(n, i, j):
    """Map pair (i,j) with i<j to lexicographic index."""
    return i * (2 * n - i - 3) // 2 + (j - i - 1)

def _has_k222(edges, n):
    """
    Check if the graph on n vertices with edge set 'edges' (list of (i,j) tuples)
    contains a K2,2,2 subgraph.
    K2,2,2 is the complete tripartite graph with parts of size 2.
    Equivalent to: exist disjoint sets A, B, C of size 2 such that
    all edges between A and B, between A and C, and between B and C are present.
    
    We iterate over all pairs of vertices (a1, a2) as part A, then for each
    pair (b1, b2) as part B, check if there exists a pair (c1, c2) as part C
    such that all cross-edges exist.
    
    For efficiency: for fixed A and B, the candidate C vertices must be
    adjacent to all of A and all of B. So C must be chosen from the common
    neighborhood of {a1, a2, b1, b2}. If that common neighborhood has size >= 2,
    we have a K2,2,2.
    """
    # Build adjacency matrix for quick lookup
    adj = [[False] * n for _ in range(n)]
    for (i, j) in edges:
        adj[i][j] = True
        adj[j][i] = True
    
    # Precompute for each vertex, the list of vertices it's adjacent to
    # For each pair (a1, a2), compute intersection of neighbors
    # Actually, for each 4-tuple (a1, a2, b1, b2) with all distinct,
    # check if N(a1) ∩ N(a2) ∩ N(b1) ∩ N(b2) has size >= 2 (excluding a1,a2,b1,b2 themselves)
    
    # More efficient: iterate over all 2-subsets for A and B
    vertices = list(range(n))
    
    # For each pair (a1, a2), precompute common neighbors
    # common[a1][a2] = set of vertices adjacent to both a1 and a2
    # This is O(n^3) memory which for n=30 is 27000 sets, fine.
    
    # Actually let's just do the direct check with bitmasks for speed
    # Use integer bitmasks for adjacency
    adj_mask = [0] * n
    for (i, j) in edges:
        adj_mask[i] |= (1 << j)
        adj_mask[j] |= (1 << i)
    
    # For each pair (a1, a2), compute common neighbor mask
    # Then for each pair (b1, b2) disjoint from (a1,a2), check if
    # common_mask[a1][a2] & adj_mask[b1] & adj_mask[b2] has at least 2 bits set
    # (excluding a1, a2, b1, b2)
    
    pair_common = {}
    for a1 in range(n):
        for a2 in range(a1 + 1, n):
            pair_common[(a1, a2)] = adj_mask[a1] & adj_mask[a2]
    
    # Now iterate over all pairs of pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for idx_a, (a1, a2) in enumerate(all_pairs):
        common_a = pair_common[(a1, a2)]
        # Remove a1, a2 from common_a
        common_a &= ~(1 << a1) & ~(1 << a2)
        
        for idx_b, (b1, b2) in enumerate(all_pairs):
            if idx_b <= idx_a:
                continue
            # Check if b1, b2 are disjoint from a1, a2
            if b1 == a1 or b1 == a2 or b2 == a1 or b2 == a2:
                continue
            # Check if b1 and b2 are adjacent to each other? No, K2,2,2 requires
            # edges between B and C, not within B. Wait, K2,2,2 is complete tripartite.
            # Parts A, B, C. Edges exist between any vertex in A and any in B,
            # any in A and any in C, any in B and any in C. No edges within a part.
            # So we don't require edge (b1, b2) to exist. We just need b1, b2 to be
            # in the common neighborhood of a1, a2, and then find c1, c2 in the
            # common neighborhood of all four.
            
            # Candidates for C: vertices adjacent to a1, a2, b1, b2
            # = common_a & adj_mask[b1] & adj_mask[b2]
            # But we need to exclude a1, a2, b1, b2 from the candidate set
            candidate_mask = common_a & adj_mask[b1] & adj_mask[b2]
            # Exclude a1, a2, b1, b2
            candidate_mask &= ~(1 << a1) & ~(1 << a2) & ~(1 << b1) & ~(1 << b2)
            
            # Count bits
            if bin(candidate_mask).count('1') >= 2:
                return True
    
    return False

def _is_valid_coloring(n, colors):
    """Check that neither color class contains K2,2,2."""
    edges0 = []
    edges1 = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                edges0.append((i, j))
            else:
                edges1.append((i, j))
            idx += 1
    
    if _has_k222(edges0, n):
        return False
    if _has_k222(edges1, n):
        return False
    return True

def _paley_coloring(n):
    """Generate a Paley-type coloring for prime n."""
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _try_improve(n, colors, max_iters=5000):
    """Try to improve a coloring via local search. Returns improved colors or None."""
    colors = list(colors)
    current_valid = _is_valid_coloring(n, colors)
    if not current_valid:
        return None
    
    # We can't really "improve" n here since n is fixed. 
    # This function is a placeholder for potential future use.
    return colors

def construct():
    # Start with the known valid n=29 Paley coloring
    n = 29
    colors = _paley_coloring(n)
    
    # Verify it's valid
    if not _is_valid_coloring(n, colors):
        # Should not happen, but just in case
        raise ValueError("Paley(29) coloring is invalid!")
    
    return {'n': n, 'colors': colors}
