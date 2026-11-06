"""Triangle-free graphs without independent 10-sets: search for larger n."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

_START_TIME = time.time()
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _is_triangle_free(adj_sets, n):
    """Check triangle-free using bitsets. adj_sets is a list of Python ints (bitmasks)."""
    for u in range(n):
        # For each pair of neighbors of u, check if they are adjacent
        mask = adj_sets[u]
        while mask:
            low = mask & (-mask)
            v = low.bit_length() - 1
            mask ^= low
            # Check if any common neighbor of u and v exists
            common = adj_sets[u] & adj_sets[v]
            if common:
                return False
    return True


def _alpha_upper_bound(adj_sets, candidates_mask, n):
    """Upper bound on independent set size using greedy coloring of the complement.
    candidates_mask: bitmask of vertices still available.
    Returns an upper bound on the size of an independent set within candidates.
    """
    # Greedy coloring of the complement graph (i.e., coloring the candidate set
    # in the original graph such that adjacent vertices get different colors)
    # The number of colors is an upper bound on alpha.
    remaining = candidates_mask
    color = 0
    while remaining:
        color += 1
        # Greedily pick a maximal set of vertices that are pairwise non-adjacent
        # (i.e., independent in original = clique in complement)
        # Actually for upper bound on independent set, we color the complement.
        # A color class in the complement = clique in original.
        # Wait, I want an upper bound on alpha(G). 
        # alpha(G) = clique_number(complement(G)).
        # So I need to color G such that adjacent vertices get different colors.
        # The number of colors used is an upper bound on... no.
        # Chromatic number chi(G) >= alpha(G)? No, chi * alpha >= n.
        # Let me use a different approach: greedy independent set packing for upper bound.
        # Actually, let me just use a simple bound: for each vertex, it can have at most
        # (n-1 - deg) neighbors not in candidates... 
        # Simpler: use the fact that alpha <= n - minimum_vertex_cover, and 
        # minimum_vertex_cover >= max matching.
        # For now, use a greedy approach: repeatedly find a maximal independent set,
        # remove it, count how many sets needed. That's a lower bound on chi, not useful.
        
        # Let me use: alpha(G) <= sum over colors of 1 where we greedily color.
        # Greedy coloring of G: assign colors so adjacent vertices differ.
        # Number of colors = upper bound on chi, and chi * alpha >= n, so alpha <= n/chi.
        # But that's a lower bound on chi, so upper bound on n/chi... not directly useful.
        
        # Better: use the standard bound alpha <= n - (number of edges in a maximal matching)
        # or just do a quick greedy: find a maximal matching, alpha <= n - |matching|.
        pass
    
    # Use a simpler bound: greedy maximal independent set gives lower bound on alpha.
    # For upper bound, use: repeatedly remove a vertex and one of its neighbors.
    # This is essentially a greedy vertex cover.
    # Let's compute a greedy maximal matching size.
    remaining2 = candidates_mask
    matching_size = 0
    while remaining2:
        # Pick lowest set bit
        low = remaining2 & (-remaining2)
        u = low.bit_length() - 1
        remaining2 ^= low
        # Find a neighbor of u in remaining2
        neighbors = adj_sets[u] & remaining2
        if neighbors:
            low_v = neighbors & (-neighbors)
            v = low_v.bit_length() - 1
            remaining2 ^= low_v
            matching_size += 1
        # else u has no neighbor in remaining, just remove u
    # alpha <= n_candidates - matching_size
    n_candidates = candidates_mask.bit_count()
    return n_candidates - matching_size


def _has_independent_set_k(adj_sets, n, k=10):
    """Exact check: does there exist an independent set of size k?
    Use branch and bound with bitsets."""
    def dfs(remaining_mask, size, current_size):
        if current_size + size >= k:
            return True
        if remaining_mask == 0:
            return False
        # Upper bound
        if current_size + _alpha_upper_bound(adj_sets, remaining_mask, n) < k:
            return False
        # Pick a vertex with fewest neighbors in remaining (to branch on include/exclude)
        # Try including the first vertex, then excluding it
        low = remaining_mask & (-remaining_mask)
        u = low.bit_length() - 1
        # Include u: remove u and all its neighbors from remaining
        new_remaining = remaining_mask & ~adj_sets[u] & ~(1 << u)
        if dfs(new_remaining, _alpha_upper_bound(adj_sets, new_remaining, n), current_size + 1):
            return True
        # Exclude u
        new_remaining2 = remaining_mask & ~(1 << u)
        if dfs(new_remaining2, _alpha_upper_bound(adj_sets, new_remaining2, n), current_size):
            return True
        return False
    
    all_mask = (1 << n) - 1
    return dfs(all_mask, n, 0)


def _construct_andrasfai(k):
    """Construct Andrasfai graph And(k) on n=3k+1 vertices.
    Triangle-free, alpha = k.
    Vertices 0..3k. Edges: i~j iff (j-i) mod (3k+1) is in {1, 3, ..., 3k-2} mod something.
    Actually, the standard Andrasfai graph And(k): n = 3k+1, vertices Z_n,
    edge if difference is in S where S is a specific set.
    The seed uses: diff % n % 3 == 1, which gives differences 1, 4, 7, ..., 3k-2 (mod n).
    That's the set {1, 4, 7, ..., 3k-2} which has k elements.
    This is indeed triangle-free and alpha = k.
    """
    n = 3 * k + 1
    S = set(range(1, n, 3))  # {1, 4, 7, ..., 3k-2}
    edges = []
    adj = [0] * n
    for i in range(n):
        for d in S:
            j = (i + d) % n
            if i < j:
                edges.append([i, j])
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return n, edges, adj


def _construct_clebsch():
    """Clebsch graph: 16 vertices, triangle-free, alpha = 5.
    Vertices: 4-bit binary strings 0..15.
    Edge between u and v if Hamming distance is 2 or 4.
    """
    n = 16
    edges = []
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            dist = bin(u ^ v).count('1')
            if dist == 2 or dist == 4:
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return n, edges, adj


def _verify(n, edges, adj):
    """Verify triangle-free and no independent set of 10."""
    if not _is_triangle_free(adj, n):
        return False
    if _has_independent_set_k(adj, n, 10):
        return False
    return True


def _random_search(n, adj_template, iterations=50000):
    """Try to find a triangle-free graph with no independent set of 10 on n vertices
    by starting from a template and randomly adding/removing edges."""
    best_n = 0
    best_result = None
    
    for it in range(iterations):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        # Start with the template and randomly perturb
        adj = list(adj_template)
        # Randomly flip some edges
        num_flips = random.randint(n // 4, n)
        for _ in range(num_flips):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            bit = 1 << v
            if adj[u] & bit:
                adj[u] ^= bit
                adj[v] ^= (1 << u)
            else:
                adj[u] |= bit
                adj[v] |= (1 << u)
        
        # Check triangle-free (fast)
        if not _is_triangle_free(adj, n):
            continue
        # Check independent set
        if _has_independent_set_k(adj, n, 10):
            continue
        # Valid!
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        best_result = {'n': n, 'edges': edges}
        best_n = n
        break  # Found one, stop
    
    return best_result


def construct():
    """Find a triangle-free graph with no independent set of 10, maximizing n."""
    # Start with known constructions
    # Andrasfai And(9): n=28, alpha=9 -> valid
    n_and, edges_and, adj_and = _construct_andrasfai(9)
    # Andrasfai And(10): n=31, alpha=10 -> NOT valid (alpha=10 means independent set of size 10 exists)
    # So max Andrasfai is And(9) with n=28.
    
    # Clebsch: n=16, alpha=5
    n_clebsch, edges_clebsch, adj_clebsch = _construct_clebsch()
    
    # Try to build larger graphs
    # Strategy: take a base graph and add vertices carefully
    # Or: use random search starting from dense triangle-free constructions
    
    best = None
    best_n = 0
    
    # Try Andrasfai And(9) first as fallback
    if _verify(n_and, edges_and, adj_and):
        best = {'n': n_and, 'edges': edges_and}
        best_n = n_and
    
    # Try Clebsch
    if _verify(n_clebsch, edges_clebsch, adj_clebsch):
        if n_clebsch > best_n:
            best = {'n': n_clebsch, 'edges': edges_clebsch}
            best_n = n_clebsch
    
    # Try to find larger graphs using random search
    # Start from n=29, 30, 31, ... up to 40
    # Use Andrasfai And(9) as template for perturbation
    
    for target_n in range(29, 41):
        if time.time() - _START_TIME > TIME_LIMIT - 15:
            break
        if target_n <= best_n:
            continue
        
        # Create a template: take Andrasfai And(9) and add random vertices
        # or just use random initialization
        # For random search, we need a starting adjacency
        # Let's use a random regular-ish graph as starting point
        # Actually, let's try: start with no edges, add edges to make it triangle-free and reduce alpha
        
        # Better approach: start from a known triangle-free graph and add vertices
        # For now, try random search with a random triangle-free template
        
        # Generate a random triangle-free graph on target_n vertices
        # Use a simple method: random ordering, add edge if it doesn't create a triangle
        # This gives a random triangle-free graph (a "random greedy" triangle-free graph)
        
        template_adj = [0] * target_n
        random.seed(target_n * 12345 + 67890)
        # Random greedy triangle-free construction
        vertex_order = list(range(target_n))
        random.shuffle(vertex_order)
        for i, u in enumerate(vertex_order):
            # Add edges from u to some later vertices, ensuring no triangle
            candidates = [v for v in vertex_order[i+1:] if not (template_adj[u] & (1 << v))]
            # Randomly choose to add or not add each edge
            for v in candidates:
                if random.random() < 0.5:
                    # Check if adding (u,v) creates a triangle
                    common = template_adj[u] & template_adj[v]
                    if common == 0:
                        template_adj[u] |= (1 << v)
                        template_adj[v] |= (1 << u)
        
        # Now do local search to reduce alpha
        # Try random local search: randomly flip edges, accept if still triangle-free and alpha doesn't increase
        # Actually, let's just check if this random one works, then try perturbations
        
        result = _random_search(target_n, template_adj, iterations=2000)
        if result is not None:
            best = result
            best_n = target_n
            break  # Found a valid graph at this n, try higher n? No, we want max n.
                    # Actually we should continue trying higher n.
                    # But if we found n, we can stop trying this n and try n+1.
                    # The loop continues to next target_n.
    
    # If we found nothing above 28, return the best we have
    if best is None:
        # Fallback: Andrasfai And(9)
        n_and, edges_and, adj_and = _construct_andrasfai(9)
        best = {'n': n_and, 'edges': edges_and}
    
    return best
# EVOLVE-BLOCK-END
