"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START

def _has_triangle(n, edges):
    """Check for a triangle using adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return True
    return False


def _max_independent_set_size(n, edges):
    """Exact branch and bound for maximum independent set size."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best = [0]
    
    def dfs(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(candidates) <= best[0]:
            return
        
        # Greedy upper bound: pick a vertex, its neighbors can't be in same independent set
        # Use a simple bound: current_size + len(candidates)
        # Better: use coloring bound or just proceed
        
        # Pick vertex with highest degree in the induced subgraph on candidates
        cand_list = list(candidates)
        if not cand_list:
            return
        
        # Heuristic: pick vertex with max degree within candidates
        max_deg = -1
        max_v = cand_list[0]
        for v in cand_list:
            deg = sum(1 for u in cand_list if u in adj[v])
            if deg > max_deg:
                max_deg = deg
                max_v = v
        
        # Branch 1: include max_v
        new_candidates = [v for v in cand_list if v not in adj[max_v] and v != max_v]
        dfs(set(new_candidates), current_size + 1)
        
        # Branch 2: exclude max_v
        new_candidates = [v for v in cand_list if v != max_v]
        dfs(set(new_candidates), current_size)
    
    dfs(set(range(n)), 0)
    return best[0]


def _verify(n, edges):
    """Check triangle-free and no independent set of size 10."""
    if _has_triangle(n, edges):
        return False
    if _max_independent_set_size(n, edges) >= 10:
        return False
    return True


def _construct_circulant(n, connection_set):
    """Build a circulant graph on n vertices with given connection set."""
    edges = []
    conn = set(connection_set)
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in conn or (n - diff) in conn:
                edges.append([i, j])
    return edges


def _is_valid_circulant(n, connection_set):
    """Check if circulant graph is triangle-free and has alpha < 10."""
    edges = _construct_circulant(n, connection_set)
    if _has_triangle(n, edges):
        return False
    if _max_independent_set_size(n, edges) >= 10:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Known good: Andrasfai graph And(9) on 26 vertices
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try to find larger circulant graphs
    # For circulant graphs, we need to check triangle-freeness and independence number
    
    # Strategy: try various n and connection sets for circulant graphs
    # Triangle-free circulant: no three elements a,b,c in connection set with a+b, a+c, b+c all in connection set (mod n)
    # Actually for circulant C(n, S), triangle exists if there exist a,b in S with a+b in S (considering the graph structure)
    
    # Let's try systematic search over n from 27 to 40
    for n in range(27, 41):
        if time.time() > deadline:
            break
        
        # For circulant graphs, try small connection sets
        # The graph must be triangle-free
        # A circulant graph C(n, S) is triangle-free iff there are no a, b, c in S (with a != b != c) such that 
        # the vertices 0, a, b form a triangle, i.e., a in S, b in S, b-a in S (or a-b in S)
        
        # Try connection sets of size 1, 2, 3, ...
        # For each n, try all subsets of {1, 2, ..., n//2} of small size
        
        # Start with size 1
        for size in [1, 2, 3]:
            if time.time() > deadline:
                break
            
            half = n // 2
            candidates = list(range(1, half + 1))
            
            # Try all combinations of 'size' elements
            for conn in itertools.combinations(candidates, size):
                if time.time() > deadline:
                    break
                
                # Check triangle-free
                is_tri_free = True
                conn_set = set(conn)
                # Add symmetric counterparts
                full_conn = set()
                for c in conn:
                    full_conn.add(c)
                    full_conn.add(n - c)
                
                # Check: for any a, b in full_conn, is (b - a) mod n in full_conn?
                # Triangle: 0, a, b with a in S, b in S, b-a in S
                for a in full_conn:
                    for b in full_conn:
                        if a == b:
                            continue
                        diff = (b - a) % n
                        if diff in full_conn and diff != 0:
                            is_tri_free = False
                            break
                    if not is_tri_free:
                        break
                
                if not is_tri_free:
                    continue
                
                # Check independence number
                edges = _construct_circulant(n, conn)
                alpha = _max_independent_set_size(n, edges)
                if alpha < 10:
                    best_n = n
                    best_edges = edges
                    break
        
        # If we found a valid graph for this n, we could try larger n
        # But we need to continue searching for the maximum n
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
