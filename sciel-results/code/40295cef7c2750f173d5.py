<<<<<<< SEARCH
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Voltage graph lift of Petersen graph with random voltages
        result = _petersen_voltage_lift(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: LUW fallback - guaranteed valid construction
        # LUW CD(d,3) has girth >= 2d+2, so d=7 gives girth >= 16
        # n = 2 * 3^(d+1) = 2 * 3^8 = 13122
        if not check_deadline():
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, 7)
                if n_luw is not None:
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)
=======
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Search for 3-regular circulant graphs with girth >= g
        # This is a structural change: direct random search in a specific family
        # with rigorous non-backtracking verification.
        result = _circulant_search(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: LUW fallback - guaranteed valid construction
        # LUW CD(d,3) has girth >= 2d+2, so d=7 gives girth >= 16
        # n = 2 * 3^(d+1) = 2 * 3^8 = 13122
        if not check_deadline():
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, 7)
                if n_luw is not None:
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _circulant_search(k, g, deadline):
    """
    Search for a k-regular circulant graph with girth >= g.
    For k=3, n must be even. Offsets are {s1, s2, s3} with no overlaps and s_i != 0, n/2.
    """
    if k != 3:
        return None
        
    # Moore bound for n(3,16) is 512.
    # We search n from 512 upwards.
    # We use a random sampling of offset triples.
    
    # Pre-compute some constants for speed
    # A 3-regular circulant graph is defined by offsets (a, b, c)
    # such that {a, -a, b, -b, c, -c} are distinct and non-zero mod n.
    
    min_n = 512
    max_n = 1000  # Limit search range to keep runtime low
    
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            break
            
        # Number of random trials per n
        # We can afford many trials if n is small, fewer if large
        trials = 500 if n < 600 else 200
        
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
                
            # Generate random offsets
            # Valid offsets are in {1, ..., n-1} \ {n/2}
            # We need 3 distinct "pairs" {s, -s}
            # So we pick 3 distinct values from 1 to n//2 - 1
            
            half_n = n // 2
            if half_n < 4:
                continue
                
            # Pick 3 distinct offsets from 1 to half_n - 1
            # Note: offset n/2 is a self-loop in terms of undirected edge? 
            # No, in circulant C_n(a), edge (i, i+a). If a=n/2, (i, i+n/2) is valid.
            # But for 3-regular, we need 3 distinct "directions".
            # If we pick offsets a, b, c from 1..n-1, we must ensure:
            # a != -a mod n (so a != n/2)
            # a != b, a != -b, etc.
            
            # Simpler: Pick 3 distinct values from 1 to n-1, then normalize to canonical form
            # and check distinctness of the set of undirected steps.
            
            # Let's just pick 3 distinct integers from 1 to n-1
            # and verify they form a valid 3-regular simple graph.
            
            offsets = random.sample(range(1, n), 3)
            
            # Check validity of offsets for simple 3-regular graph
            # The set of steps is S = {o1, -o1, o2, -o2, o3, -o3} mod n
            # It must have size 6 (no overlaps, no 0)
            steps = set()
            valid = True
            for o in offsets:
                if o == 0 or o == n//2: # o == n/2 implies -o == o, size reduces
                    valid = False
                    break
                steps.add(o % n)
                steps.add((-o) % n)
            
            if not valid or len(steps) != 6:
                continue
                
            # Build adjacency list for girth check
            # We don't build full edge list yet, just check girth
            if _verify_circulant_girth(n, offsets, g):
                # Construct edges
                edges = []
                for u in range(n):
                    for o in offsets:
                        v = (u + o) % n
                        if u < v:
                            edges.append((u, v))
                        # else: edge (v, u) will be added when v is processed?
                        # No, we iterate u from 0 to n-1.
                        # If u < v, add. If u > v, it was added when v was u.
                        # If u == v, loop (not possible as o != 0).
                
                # Verify degree
                # Each u connects to u+o1, u+o2, u+o3.
                # Total edges = n * 3 / 2.
                if len(edges) == n * 3 // 2:
                    return n, edges
                    
    return None


def _verify_circulant_girth(n, offsets, g):
    """
    Verify girth >= g for a 3-regular circulant graph defined by n and offsets.
    Uses non-backtracking BFS from a single vertex (0) is not sufficient for all graphs,
    but for vertex-transitive graphs like circulants, if there is a short cycle,
    it can be rotated to pass through vertex 0? 
    Actually, yes, circulant graphs are vertex-transitive. 
    So checking from vertex 0 is sufficient to find the girth.
    However, the cycle might not be "centered" at 0 in a way that standard BFS detects easily,
    but Non-Backtracking Walks from 0 will detect any cycle that can be shifted to start at 0.
    Since the graph is vertex-transitive, every cycle is equivalent to a cycle passing through 0.
    So we only need to check from vertex 0.
    """
    # Adjacency
    adj = [[] for _ in range(n)]
    for u in range(n):
        for o in offsets:
            v = (u + o) % n
            adj[u].append(v)
            
    # Non-backtracking BFS from 0
    # State: (current_node, prev_node, depth)
    # We want to find if we return to 0 with depth < g.
    # Actually, we need to detect *any* cycle.
    # In a vertex transitive graph, if girth < g, there is a cycle of length L < g.
    # We can rotate it so 0 is on the cycle.
    # Then we can start a non-backtracking walk from 0, go around the cycle, and return to 0.
    # So checking for a closed non-backtracking walk of length < g starting and ending at 0 is sufficient.
    
    # We use a set to store visited states (node, prev) to avoid infinite loops
    # and to prune.
    # But we need to track depth to ensure we stop at g-1.
    
    visited = set()
    queue = deque()
    
    # Start at 0, prev = -1 (none)
    # We expand up to depth g-1.
    # If we reach (0, prev) with depth L < g and L >= 2, we found a cycle.
    # Note: (0, -1) is the start. We shouldn't count immediate return as cycle?
    # A cycle of length 2 would be 0-1-0.
    # Walk: 0 -> 1 (prev 0) -> 0 (prev 1). Depth 2.
    # Is (0, 1) a valid state? Yes.
    # When we pop (0, 1, 2), we check if node == 0 and depth >= 2.
    
    for neighbor in adj[0]:
        queue.append((neighbor, 0, 1))
        visited.add((neighbor, 0))
        
    while queue:
        u, prev, depth = queue.popleft()
        
        if depth >= g - 1:
            continue
            
        for v in adj[u]:
            if v == prev:
                continue
                
            new_state = (v, u)
            if new_state in visited:
                continue
                
            visited.add(new_state)
            new_depth = depth + 1
            
            if v == 0 and new_depth >= 2:
                # Found a cycle of length new_depth
                return False
                
            if new_depth < g - 1:
                queue.append((v, u, new_depth))
                
    return True
>>>>>>> REPLACE

