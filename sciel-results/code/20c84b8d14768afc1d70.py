import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy:
    1. Try LUW construction with many rule variants (fast, deterministic).
    2. Try hash-consing with improved boundary matching.
    3. Verify girth exactly and return the best valid graph found.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_n = float('inf')
    best_edges = None

    if k == 3 and g == 16:
        # --- Strategy 1: LUW with many rule variants ---
        for d in [7, 8, 9, 10, 11]:
            if time.time() > deadline - 20:
                break
            for rule in _generate_rule_variants(k, d):
                if time.time() > deadline - 15:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            if n_luw <= 936:
                                return n_luw, edges_luw
                except Exception:
                    continue

        # --- Strategy 2: Hash-consing with improved boundary matching ---
        if time.time() < deadline - 10:
            try:
                n_hc, edges_hc = _construct_hash_consolidated(k, g)
                if n_hc > 0 and _is_valid_3reg(n_hc, edges_hc, k):
                    if _has_girth_at_least_fast(n_hc, edges_hc, g):
                        if n_hc < best_n:
                            best_n = n_hc
                            best_edges = edges_hc
            except Exception:
                pass

        if best_n != float('inf'):
            return best_n, best_edges

    # --- Generic fallback for other (k, g) ---
    min_d = max(3, g // 2 - 2)
    max_d = min(g + 5, 12)

    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        for rule in _generate_rule_variants(k, d):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue

    # Last resort
    n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, _generate_rule_variants(k, max_d + 3)[0])
    return n_luw, edges_luw


def _construct_hash_consolidated(k, g):
    """
    Hash-consing construction: build k-regular tree to depth radius,
    then match boundary vertices using a structure-aware pairing that
    maximizes tree-distance between matched vertices.
    """
    radius = (g - 1) // 2  # For g=16, radius=7
    q = k

    # Build k-regular tree of depth radius
    vertices = {}
    edges = []
    vertex_list = []

    def get_vertex(path):
        if path not in vertices:
            vertices[path] = len(vertex_list)
            vertex_list.append(path)
        return vertices[path]

    root_path = ()
    get_vertex(root_path)

    queue = deque([root_path])
    while queue:
        path = queue.popleft()
        depth = len(path)
        if depth >= radius:
            continue
        if depth == 0:
            available = list(range(k))
        else:
            last_dir = path[-1]
            available = [d for d in range(k) if d != last_dir]

        for d in available:
            new_path = path + (d,)
            idx = get_vertex(new_path)
            parent_idx = vertices[path]
            edges.append((parent_idx, idx))
            queue.append(new_path)

    n_tree = len(vertex_list)

    boundary_paths = [p for p in vertex_list if len(p) == radius]
    n_boundary = len(boundary_paths)

    if n_boundary == 0:
        return n_tree, edges

    # For each boundary vertex, compute its "signature" based on the path structure.
    # Two boundary vertices that are far apart in the tree (longest common prefix is short)
    # can be safely connected without creating short cycles.
    #
    # Strategy: For k=3, each boundary vertex needs 2 more edges.
    # Pair boundary vertices using a "complement" scheme:
    # For a boundary vertex with path (d1, d2, ..., d7), connect to vertices
    # whose paths differ in many early positions.
    #
    # More concretely: use a bit-reversal or XOR-based matching on the path codes.

    if k == 3:
        # Encode each boundary path as an integer
        # Path (d1,...,d7) where each di in {0,1,2}
        # Encode as base-3 number: code = d1*3^6 + d2*3^5 + ... + d7*3^0
        # But we need to handle the non-backtracking constraint properly.
        # Actually, the path is already valid (non-backtracking by construction).
        
        # Better approach: use the path tuple directly for distance computation.
        # Two boundary vertices at depth r have tree-distance = 2*(r - lcp_length)
        # where lcp_length is the length of their longest common prefix.
        
        # To maximize minimum distance, we want to pair vertices with short LCPs.
        # A simple approach: sort by path, then pair i with (i + n_boundary//2) mod n_boundary
        # in the sorted order. This tends to pair vertices from different "branches".
        
        # Even better: use a recursive "complement" matching.
        # For k=3, radius=7: boundary has 3*2^6 = 192 vertices.
        # The first step d1 in {0,1,2} gives 3 groups of 64.
        # Within each group, the remaining path is in a 2-ary tree of depth 6.
        
        # Match vertices from different first-step groups:
        # Group 0 (d1=0) with Group 1 (d1=1), Group 1 with Group 2, Group 2 with Group 0.
        # Within each cross-group pair, use a bit-reversal matching on the remaining 6 bits.
        
        # Let's implement this:
        groups = {0: [], 1: [], 2: []}
        for i, p in enumerate(boundary_paths):
            groups[p[0]].append((i, p))
        
        # For each group, sort by the remaining path (positions 1-6)
        for g_idx in range(3):
            groups[g_idx].sort(key=lambda x: x[1][1:])
        
        # Match group 0 with group 1, group 1 with group 2, group 2 with group 0
        # Each boundary vertex needs 2 edges, so each group of 64 connects to two other groups.
        # Group 0: each vertex connects to one in group 1 and one in group 2.
        
        added_edges = set()
        n_per_group = len(groups[0])  # Should be 64
        
        # For group 0 -> group 1: use bit-reversal on the 6-bit suffix
        def bit_reverse_6(bits):
            """Reverse 6 bits encoded as integer 0-63"""
            r = 0
            for i in range(6):
                r = (r << 1) | (bits & 1)
                bits >>= 1
            return r
        
        # Encode suffix (positions 1-6) as a 6-bit number
        # Each position has 2 choices (non-backtracking), so encode as binary
        def encode_suffix(path):
            # path[1:] has 6 elements, each in {0,1,2} but constrained by non-backtracking
            # Map to 0 or 1 based on whether it's less or greater than the previous
            # Actually, simpler: just use the tuple directly for sorting, and use index for matching
            pass
        
        # Simpler: just use index-based matching with a twist
        # For group a matching to group b:
        # Vertex at index i in group a matches vertex at index (i * 37 + 13) % n_per_group in group b
        # (using a permutation that mixes up the indices)
        
        def match_groups(group_a, group_b, offset=0):
            """Match vertices in group_a to vertices in group_b"""
            for i, (idx_a, path_a) in enumerate(group_a):
                j = (i * 37 + 13 + offset) % n_per_group
                idx_b, path_b = group_b[j]
                global_a = vertices[path_a]
                global_b = vertices[path_b]
                edge = (min(global_a, global_b), max(global_a, global_b))
                if global_a != global_b and edge not in added_edges:
                    added_edges.add(edge)
                    edges.append(edge)
        
        # Group 0 connects to Group 1 and Group 2
        match_groups(groups[0], groups[1], offset=0)
        match_groups(groups[0], groups[2], offset=1)
        match_groups(groups[1], groups[2], offset=2)
        
        # Verify all boundary vertices have degree k
        # (This should work out if the matching is bijective per pair)
        
    return n_tree, edges


def _is_valid_3reg(n, edges, k):
    """Check that the graph is k-regular and simple."""
    if len(edges) != n * k // 2:
        return False

    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u in adj[v]:
            return False
        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    return True


def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    More variants with different patterns to find smaller valid graphs.
    """
    variants = []
    seen = set()
    
    def make_rule(base_rules, pattern_type, seed_shift=0):
        """Create a rule dict from base rules and pattern type."""
        rule = {}
        for c in range(2, max_d + 1):
            if c in base_rules:
                rule[c] = base_rules[c]
            else:
                if pattern_type == 0:
                    r, c0 = (c - 5 + seed_shift) % 4, c - (c - 5 + seed_shift) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 1:
                    r, c0 = (c - 4 + seed_shift) % 4, c - (c - 4 + seed_shift) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 2:
                    r, c0 = (c - 5 + seed_shift) % 3, c - (c - 5 + seed_shift) % 3
                    rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 3:
                    r, c0 = (c - 5 + seed_shift) % 5, c - (c - 5 + seed_shift) % 5
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 1, 1)][r % 5]
                elif pattern_type == 4:
                    r, c0 = (c - 6 + seed_shift) % 6, c - (c - 6 + seed_shift) % 6
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 1, 1), (c0 + 2, 1)][r % 6]
                elif pattern_type == 5:
                    # Fibonacci-like
                    rule[c] = (1, c - 1) if c % 2 == 0 else (1, c - 2)
                else:
                    rule[c] = (1, 1)
        return rule
    
    # Generate many variants
    base_rule_sets = [
        {2: (1, 1), 3: (2, 1), 4: (1, 2)},
        {2: (1, 1), 3: (1, 2), 4: (2, 1)},
        {2: (1, 1), 3: (1, 1), 4: (2, 1)},
        {2: (1, 1), 3: (1, 2), 4: (1, 1)},
        {2: (1, 1), 3: (2, 1), 4: (2, 1)},
        {2: (1, 1), 3: (1, 1), 4: (1, 2)},
    ]
    
    for base in base_rule_sets:
        for ptype in range(6):
            for shift in range(3):
                rule = make_rule(base, ptype, shift)
                key = tuple(sorted(rule.items()))
                if key not in seen:
                    seen.add(key)
                    variants.append(rule)
    
    return variants


def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with given degree k, depth d, and rule.
    """
    q = k if k >= 3 else 3

    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END
