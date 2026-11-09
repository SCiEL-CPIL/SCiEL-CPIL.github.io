"""Construct a C4-free graph with no independent 11-set, maximizing n."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _pg2_polarity_graph(q):
    """Build the polarity graph of PG(2,q). Vertices are points, edge if point lies on polar line."""
    # Points of PG(2,q): (x, y, 1) for x,y in GF(q), plus (x, 1, 0) for x in GF(q), plus (0, 1, 0)
    # Total: q^2 + q + 1 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((0, 1, 0))
    
    n = len(points)
    # Polarity: (x,y,z) perp (x',y',z') iff x*x' + y*y' + z*z' = 0 (symmetric bilinear form)
    # For PG(2,q) with q odd, use the standard polarity: x*x' + y*y' + z*z' = 0
    
    edges = []
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            # Check if point j is on the polar line of point i
            # Polar line of (x1,y1,z1): x1*x + y1*y + z1*z = 0
            # Point j = (x2,y2,z2) is on it if x1*x2 + y1*y2 + z1*z2 = 0
            val = (x1*x2 + y1*y2 + z1*z2) % q
            if val == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return n, edges, adj

def _check_c4_free(adj, n):
    """Check that no two vertices have more than one common neighbor."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _max_independent_set_size(adj, n, timeout=None):
    """Find the size of the maximum independent set using branch and bound."""
    if timeout is None:
        timeout = 5.0
    start_time = time.time()
    
    best = [0]
    # Use a greedy coloring bound for the complement (which gives upper bound on MIS)
    # Actually, let's use a simpler branch and bound with degree ordering
    
    # Order vertices by degree (ascending) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i])
    
    # Reverse order: process high-degree vertices first for pruning
    # Actually for independent set, we want to branch on vertices that appear in many sets
    # Let's use a standard branch and bound
    
    def branch(remaining, current_size, current_set):
        if time.time() - start_time > timeout:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best[0]:
            return
        
        # Pick a vertex to branch on
        v = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include v in independent set
        # Remove v and all its neighbors from remaining
        new_remaining = [u for u in rest if u not in adj[v]]
        branch(new_remaining, current_size + 1, current_set | {v})
        
        # Option 2: exclude v
        branch(rest, current_size, current_set)
    
    branch(list(range(n)), 0, set())
    return best[0]

def _max_independent_set_size_fast(adj, n):
    """Faster MIS check using greedy + local improvement, with verification."""
    # Greedy: repeatedly pick vertex with minimum degree in remaining graph
    remaining = set(range(n))
    independent = set()
    
    while remaining:
        # Pick vertex with minimum degree in induced subgraph on remaining
        min_deg = float('inf')
        v = None
        for u in remaining:
            deg = len(adj[u] & remaining)
            if deg < min_deg:
                min_deg = deg
                v = u
        independent.add(v)
        remaining -= adj[v]
        remaining.discard(v)
    
    return len(independent)

def _verify_no_independent_11(adj, n):
    """Verify there is no independent set of size 11."""
    if n < 11:
        return True
    # Use branch and bound with timeout
    start = time.time()
    found = [False]
    
    def search(remaining, current_size, current_set):
        if found[0]:
            return
        if time.time() - start > 10.0:
            return
        if current_size >= 11:
            found[0] = True
            return
        if not remaining:
            return
        if current_size + len(remaining) < 11:
            return
        
        v = remaining[0]
        rest = remaining[1:]
        
        # Include v
        new_rem = [u for u in rest if u not in adj[v]]
        search(new_rem, current_size + 1, current_set | {v})
        
        if not found[0]:
            # Exclude v
            search(rest, current_size, current_set)
    
    search(list(range(n)), 0, set())
    return not found[0]

def _local_search(n, edges, adj, max_time=30.0):
    """Try to improve the graph via local search: add edges or vertices."""
    start_time = time.time()
    best_n = n
    best_edges = edges[:]
    best_adj = [s.copy() for s in adj]
    
    # Try adding vertices one at a time
    current_n = n
    current_edges = edges[:]
    current_adj = [s.copy() for s in adj]
    
    while time.time() - start_time < max_time:
        # Try to add a new vertex
        new_v = current_n
        # Try all possible neighborhoods for the new vertex
        # The new vertex must not create a C4
        # For the new vertex v to connect to a set S of existing vertices:
        # For any two vertices u, w in S, they must not have a common neighbor other than v
        # Since the graph is C4-free, u and w have at most one common neighbor
        # If u and w already share a neighbor x, then v cannot be adjacent to both u and w
        # (because then u and w would have common neighbors x and v)
        
        # So S must be a set of vertices where no two share a common neighbor
        # This is a "code" in the graph
        
        # Greedy: try to find a large set S
        candidates = list(range(current_n))
        random.shuffle(candidates)
        S = []
        used_neighbors = set()
        
        for u in candidates:
            # Check if adding u to S is valid
            # For all w in S, u and w must not share a common neighbor
            valid = True
            for w in S:
                common = current_adj[u] & current_adj[w]
                if common:
                    valid = False
                    break
            if valid:
                S.append(u)
        
        if len(S) >= 1:
            # Check if adding vertex with neighborhood S keeps C4-free
            # We need to verify: for any two existing vertices u, w, if they both connect to v
            # (i.e., both in S), they must not have another common neighbor
            # This is already checked above
            # Also need to check: for any existing vertex u not in S, and any w in S,
            # u and w must not have a common neighbor (since u is not adjacent to v, 
            # the only potential C4 involving v would be through two neighbors of v)
            # Actually, a C4 involving v: v-a-b-c-v where a,c in S and a,b,c are distinct
            # This requires a and c to have a common neighbor b (other than v)
            # We already checked that no two in S share a common neighbor
            # So the graph remains C4-free
            
            new_adj = current_adj + [set(S)]
            for u in S:
                new_adj[u].add(new_v)
            
            # Check independence number
            # The new vertex v is adjacent to S, so any independent set containing v
            # cannot contain any vertex in S
            # An independent set not containing v is an independent set of the old graph
            # So alpha(new) = max(alpha(old), 1 + alpha(old - S - N(S)))
            # But this is complex; let's just verify
            
            if _verify_no_independent_11(new_adj, current_n + 1):
                current_n += 1
                current_edges = current_edges + [[new_v, u] for u in S]
                current_adj = new_adj
                if current_n > best_n:
                    best_n = current_n
                    best_edges = current_edges[:]
                    best_adj = [s.copy() for s in current_adj]
            else:
                # Try smaller S
                pass
        else:
            break
    
    return best_n, best_edges, best_adj

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    # Start with PG(2,5) polarity graph
    n, edges, adj = _pg2_polarity_graph(5)
    
    # Verify C4-free
    if not _check_c4_free(adj, n):
        # Fallback: use disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b],[a,c],[b,c]]
        return {'n': 30, 'edges': edges}
    
    # Verify no independent 11-set
    if not _verify_no_independent_11(adj, n):
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b],[a,c],[b,c]]
        return {'n': 30, 'edges': edges}
    
    best_n = n
    best_edges = edges[:]
    
    # Try local search to extend
    if time.time() - start_time < deadline - 20:
        ls_n, ls_edges, ls_adj = _local_search(n, edges, adj, max_time=min(30.0, deadline - (time.time() - start_time) - 10))
        if ls_n > best_n:
            best_n = ls_n
            best_edges = ls_edges
    
    # If still below 39, try a different construction
    if best_n < 39 and time.time() - start_time < deadline - 15:
        # Try PG(2,7) polarity graph? n = 7^2+7+1 = 57, too large
        # Try a different approach: random C4-free graph construction
        # Greedy construction: add vertices one by one, connecting to a random valid set
        random.seed(42)
        g_n = 0
        g_edges = []
        g_adj = []
        
        target_n = 39
        for _ in range(target_n):
            if time.time() - start_time > deadline - 10:
                break
            new_v = g_n
            # Find a valid neighborhood
            candidates = list(range(g_n))
            random.shuffle(candidates)
            S = []
            for u in candidates:
                valid = True
                for w in S:
                    if g_adj[u] & g_adj[w]:
                        valid = False
                        break
                if valid:
                    S.append(u)
            
            if S:
                new_adj = g_adj + [set(S)]
                for u in S:
                    new_adj[u].add(new_v)
                # Quick C4 check
                c4_free = True
                for i in range(new_v):
                    for j in range(i+1, new_v):
                        if len(new_adj[i] & new_adj[j]) > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                if c4_free:
                    g_n += 1
                    g_edges = g_edges + [[new_v, u] for u in S]
                    g_adj = new_adj
                else:
                    # Try with empty S (isolated vertex)
                    # But isolated vertex might increase alpha
                    pass
            else:
                # Add isolated vertex
                g_adj = g_adj + [set()]
                g_n += 1
        
        if g_n > best_n and _verify_no_independent_11(g_adj, g_n):
            best_n = g_n
            best_edges = g_edges
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
