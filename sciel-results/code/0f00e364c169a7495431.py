"""Greedy randomized C4-free 5-coloring construction with deadline guard."""
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit
    
    best_n = 11
    best_colors = None
    # Default: K11 cyclic coloring
    n0 = 11
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    best_n = n0
    
    # Try greedy construction for increasing n
    for target_n in range(12, 29):
        if time.time() > DEADLINE - 10:
            break
        
        # Multiple random restarts for this n
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
            
            result = _greedy_construct(target_n, restart)
            if result is not None:
                n_found, colors_found = result
                if n_found > best_n:
                    best_n = n_found
                    best_colors = colors_found
    
    return {'n': best_n, 'colors': best_colors}


def _greedy_construct(n, seed):
    """
    Greedy randomized construction: assign colors to edges one by one,
    maintaining counts of common neighbors per color to avoid C4.
    
    Returns (n, colors) if a valid C4-free 5-coloring of K_n is found, else None.
    """
    rng = random.Random(seed * 12345 + n)
    
    # Edge index: (i, j) -> i*n + j (only for i < j)
    # We'll process edges in a randomized order for diversity
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    
    # colors[edge_index] = color assigned, -1 for unassigned
    color_of = {}  # (i,j) -> color
    
    # For each color c, track common neighbor counts:
    # cn[c][u][v] = number of vertices w such that (u,w) and (v,w) are both color c
    # This is expensive for large n. Instead, track for each color c and each pair (u,v):
    # how many common neighbors they have in color c.
    
    # More efficient: for each color c, maintain a set/dict of edges colored c.
    # When considering coloring edge (u,v) with color c, we need to check if
    # there exists a vertex w such that both (u,w) and (v,w) are already color c.
    # This would create a C4 (u,w,v,?) - actually it creates a triangle, not C4.
    
    # Wait - C4 means a cycle of length 4. A C4 is u-v-w-x-u where all edges same color.
    # Two vertices u,v have a common neighbor w in color c means edge (u,w) and (v,w) are color c.
    # If u and v have TWO common neighbors w1, w2 in color c, then u-w1-v-w2-u is a C4.
    
    # So the condition is: no two vertices share two or more common neighbors in the same color.
    # Equivalently: for each color c, the graph formed by edges of color c is C4-free.
    
    # Strategy: when assigning color c to edge (u,v), check if u and v already have
    # two common neighbors in color c. If they have exactly one, that's fine (one more would make C4).
    # If they have two or more, assigning this edge would create a C4.
    
    # Actually, adding edge (u,v) in color c creates a C4 if there exist w1 != w2 such that
    # (u,w1), (v,w1), (u,w2), (v,w2) are all color c. This is equivalent to u and v having
    # at least 2 common neighbors in color c BEFORE adding the edge... no wait.
    
    # Let me reconsider. A C4 in color c is four distinct vertices a,b,c,d with edges
    # (a,b), (b,c), (c,d), (d,a) all in color c. 
    # 
    # When we add edge (u,v) in color c, it creates a C4 if there exist w1, w2 (distinct from u,v, w1!=w2)
    # such that (u,w1), (w1,v), (v,w2), (w2,u) are all in color c. But (w1,v) and (v,w2) are the same as
    # (v,w1) and (v,w2). So we need: (u,w1), (v,w1), (v,w2), (u,w2) all in color c.
    # This means w1 and w2 are both common neighbors of u and v in color c.
    # So if u and v already have >= 2 common neighbors in color c, adding (u,v) in color c creates C4.
    # If they have exactly 1 common neighbor w in color c, adding (u,v) doesn't immediately create a C4
    # through that w alone, but it increases their common neighbor count to 2, which is dangerous for future edges.
    
    # So the check is: before assigning color c to (u,v), count common neighbors of u and v in color c.
    # If count >= 2, this assignment is forbidden.
    
    # Data structure: for each color c, for each vertex v, maintain the set of neighbors in color c.
    # Then common neighbors of u,v in color c = |neighbors_c[u] ∩ neighbors_c[v]|
    
    neighbors = [defaultdict(set) for _ in range(5)]  # neighbors[c][v] = set of u where (min(u,v),max(u,v)) in color c
    
    def common_neighbor_count(c, u, v):
        """Count common neighbors of u and v in color c."""
        su = neighbors[c][u]
        sv = neighbors[c][v]
        # Choose smaller set to iterate
        if len(su) <= len(sv):
            return sum(1 for x in su if x in sv)
        else:
            return sum(1 for x in sv if x in su)
    
    def can_assign(c, u, v):
        """Check if assigning color c to edge (u,v) is valid (doesn't create C4)."""
        return common_neighbor_count(c, u, v) < 2
    
    def assign(c, u, v):
        """Assign color c to edge (u,v)."""
        neighbors[c][u].add(v)
        neighbors[c][v].add(u)
        color_of[(min(u,v), max(u,v))] = c
    
    # Score function: prefer assignments that minimize creation of "forbidden pairs"
    # A forbidden pair is a pair (u,v) with exactly 1 common neighbor in some color c.
    # Assigning edge (u,v) in color c increases common neighbor counts for pairs involving u and v.
    
    # For simplicity, use a greedy approach: for each edge, pick the color that minimizes
    # the maximum increase in common neighbor counts, or just pick a valid color randomly.
    
    def score_assignment(c, u, v):
        """
        Score how bad it is to assign color c to (u,v).
        Lower is better.
        Consider: after assignment, for each neighbor w of u in color c, the pair (w,v) gains
        a common neighbor (u). Similarly for each neighbor w of v in color c, pair (u,w) gains
        a common neighbor (v).
        """
        # After adding (u,v) in color c:
        # For each w in neighbors[c][u]: pair (w,v) now has one more common neighbor (u) in color c
        # For each w in neighbors[c][v]: pair (u,w) now has one more common neighbor (v) in color c
        
        # Count how many of these pairs already have 1 common neighbor (would become 2 = bad)
        bad = 0
        su = neighbors[c][u]
        sv = neighbors[c][v]
        
        for w in su:
            if w == v:
                continue
            # Pair (w, v) in color c: how many common neighbors do they currently have?
            # They share u as a common neighbor (since w~u and v~u in color c)
            # After adding (u,v), they still share u. Do they share others?
            # The new edge (u,v) doesn't directly affect common neighbors of (w,v) unless w~v in color c.
            # Actually, common neighbors of w and v in color c are vertices x such that w~x and v~x in color c.
            # u is one such vertex (w~u, v~u). Adding (u,v) doesn't change this set.
            # Wait, I'm confusing myself.
            pass
        
        # Simpler heuristic: just count current common neighbors and prefer colors where
        # u and v have fewer common neighbors (less risk).
        cn = common_neighbor_count(c, u, v)
        # Also consider: how many "tight" pairs would be created
        # A pair (a,b) is tight if they have exactly 1 common neighbor in color c.
        # Adding (u,v) makes (w,v) potentially tighter for w in neighbors[c][u], etc.
        
        # Use a simple penalty: cn * 10 + number of neighbors in color c
        penalty = cn * 100
        penalty += len(su) + len(sv)
        return penalty
    
    # Process edges
    assigned_count = 0
    total_edges = len(edges)
    
    for idx, (u, v) in enumerate(edges):
        if time.time() > DEADLINE - 3:
            return None
        
        # Find valid colors
        valid_colors = []
        for c in range(5):
            if can_assign(c, u, v):
                valid_colors.append(c)
        
        if not valid_colors:
            return None  # Dead end
        
        if len(valid_colors) == 1:
            c = valid_colors[0]
        else:
            # Score each valid color
            scores = [(score_assignment(c, u, v), c) for c in valid_colors]
            scores.sort()
            # Pick from top few with some randomness
            top_k = min(3, len(scores))
            c = scores[rng.randint(0, top_k - 1)][1]
        
        assign(c, u, v)
        assigned_count += 1
    
    # Verify the coloring is valid
    if not _verify_c4_free(n, color_of):
        return None
    
    # Convert to list in lexicographic order
    colors_list = []
    for i in range(n):
        for j in range(i + 1, n):
            colors_list.append(color_of[(i, j)])
    
    return (n, colors_list)


def _verify_c4_free(n, color_of):
    """Verify that no color class contains a C4."""
    # For each color, build adjacency and check for C4
    for c in range(5):
        # Build adjacency list for color c
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if color_of.get((i, j), -1) == c:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Check for C4: for each pair (u,v), count common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
    return True


# Keep the original interface
def _construct_cyclic():
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _check_c4(n, colors):
    """Check if coloring has no C4 in any color class."""
    # Reconstruct color_of
    color_of = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            color_of[(i, j)] = colors[idx]
            idx += 1
    return _verify_c4_free(n, color_of)

def _get_idx(i, j, n):
    """Get index of edge (i,j) in the flat color list."""
    # Number of edges before row i: i*(2*n - i - 1) // 2
    # Within row i, edge (i,j) is at position j - i - 1
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(result):
    """Validate a coloring result."""
    n = result['n']
    colors = result['colors']
    expected_len = n * (n - 1) // 2
    if len(colors) != expected_len:
        return False
    if not all(0 <= c <= 4 for c in colors):
        return False
    return _check_c4(n, colors)
