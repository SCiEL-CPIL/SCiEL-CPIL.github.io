"""Evolved program: grow from Paley(35) toward 36+ via randomized local search."""
import time
import random

_DEADLINE = 140.0  # safety margin under 150s kill
_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _paley_35():
    # Paley graph on 35 vertices (35 = 5*7, not a prime power -> use union of two Paley?)
    # Actually 35 is not a prime power. Use the known R(4,6) graph on 35 vertices.
    # The standard construction: Paley graph exists for q = p^k where p^k ≡ 1 mod 4.
    # 35 is not such a q. Instead, use a known 35-vertex Ramsey(4,6) graph.
    # A well-known 35-vertex example comes from exoo. We'll use a heuristic:
    # Build a graph with 35 vertices, degree ~17, using a structured pattern.
    # Use the Paley graph on 31 vertices plus 4 extra? Let's use a different approach.
    # Use the Clebsch-like or a known construction.
    # Simplest: use a random-ish structured graph that is known to work.
    # The 35-vertex Ramsey graph for R(4,6) is well-studied.
    # We'll construct it via the "Paley-like" graph on 35: 
    # Actually, let's use the known fact: the Paley graph on 29 vertices is (4,6)-Ramsey-free? No.
    # Let me use a concrete known 35-vertex graph.
    # The standard 35-vertex R(4,6) graph can be constructed as follows:
    # Take the complete 5-partite graph K_{7,7,7,7,7} and remove a perfect matching in each part?
    # That gives 35 vertices, degree 28. Too high (K4 exists).
    # 
    # Better approach: Use the known exoo 35-vertex graph. Since I can't look it up,
    # I'll use a structured random graph and verify, then optimize.
    # 
    # Let me use a different strategy: build a graph on 35 vertices using a 
    # "random regular-ish" graph with degree ~17 and check.
    
    random.seed(42)
    n = 35
    # Start with a structured graph: use the Paley graph on 31 (31 is prime, 31≡3 mod 4, 
    # so Paley graph uses quadratic residues mod 31)
    # Paley graph on 31: vertices 0..30, edge i-j if (j-i) is a quadratic residue mod 31
    squares_31 = {x*x % 31 for x in range(1, 31)}
    adj = [[0]*35 for _ in range(35)]
    for i in range(31):
        for j in range(i+1, 31):
            if (j - i) % 31 in squares_31:
                adj[i][j] = adj[j][i] = 1
    # Add 4 more vertices (31,32,33,34) connected in a balanced way
    for k in range(31, 35):
        for i in range(31):
            # Connect to ~half of the 31 vertices
            if (k - i) % 31 in squares_31 or random.random() < 0.5:
                adj[k][i] = adj[i][k] = 1
    for k in range(31, 34):
        for l in range(k+1, 35):
            if random.random() < 0.5:
                adj[k][l] = adj[l][k] = 1
    
    return adj

def _has_k4(adj, n):
    """Check if graph has a K4. Returns True if found."""
    # For each pair of adjacent vertices, check if they share 2+ common neighbors that are also adjacent
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                common = [k for k in range(n) if k != i and k != j and adj[i][k] and adj[j][k]]
                for a in range(len(common)):
                    for b in range(a+1, len(common)):
                        u, v = common[a], common[b]
                        if adj[u][v]:
                            return True
    return False

def _has_indep_6(adj, n):
    """Check if complement has a K6 (i.e., graph has independent set of size 6)."""
    # Build complement adjacency
    comp = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                comp[i][j] = comp[j][i] = 1
    # Check for K6 in complement
    # Use a simple recursive search with pruning
    def find_clique(vertices, size_needed, current):
        if len(current) == size_needed:
            return True
        if len(vertices) < size_needed:
            return False
        # Pruning: not enough vertices left
        for idx in range(len(vertices)):
            v = vertices[idx]
            # Check if v is connected to all in current
            ok = True
            for u in current:
                if not comp[v][u]:
                    ok = False
                    break
            if not ok:
                continue
            if find_clique(vertices[idx+1:], size_needed, current + [v]):
                return True
        return False
    
    # Use a smarter approach: try all 6-combinations is too slow for n=36.
    # Use Bron-Kerbosch style or just check via the recursive above with early exit.
    # For n<=40, a 6-clique search in complement should be fast enough.
    return find_clique(list(range(n)), 6, [])

def _count_k4s(adj, n):
    """Count number of K4s in the graph."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                common = [k for k in range(n) if k != i and k != j and adj[i][k] and adj[j][k]]
                for a in range(len(common)):
                    for b in range(a+1, len(common)):
                        u, v = common[a], common[b]
                        if adj[u][v]:
                            count += 1
    return count

def _count_indep_6(adj, n):
    """Count number of independent 6-sets."""
    comp = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                comp[i][j] = comp[j][i] = 1
    
    count = 0
    def find_clique(vertices, size_needed, current):
        nonlocal count
        if len(current) == size_needed:
            count += 1
            return
        if len(vertices) < size_needed:
            return
        for idx in range(len(vertices)):
            v = vertices[idx]
            ok = True
            for u in current:
                if not comp[v][u]:
                    ok = False
                    break
            if not ok:
                continue
            find_clique(vertices[idx+1:], size_needed, current + [v])
    
    find_clique(list(range(n)), 6, [])
    return count

def _try_flip(adj, n, i, j):
    """Try flipping edge (i,j). Return new adj if it improves (fewer violations)."""
    old = adj[i][j]
    adj[i][j] = adj[j][i] = 1 - old
    # Quick check: does this create a K4 or independent 6?
    # Instead of full check, just compute violation count
    v1 = _count_k4s(adj, n)
    v2 = _count_indep_6(adj, n)
    total = v1 + v2
    if total == 0:
        return adj, True
    # Revert if worse
    # We need to compare with previous, so this is tricky. Let's just return and let caller decide.
    adj[i][j] = adj[j][i] = old
    return adj, False

def _violation_count(adj, n):
    return _count_k4s(adj, n) + _count_indep_6(adj, n)

def construct():
    """Construct a K4-free graph with no independent 6-set, maximizing n."""
    random.seed(12345)
    
    # Start from Paley(31) + 4 extra, then optimize
    adj = _paley_35()
    n = 35
    
    # First, check if our starting graph is valid
    while _violation_count(adj, n) > 0 and _now() < _DEADLINE - 20:
        # Random flip to reduce violations
        best_adj = None
        best_v = _violation_count(adj, n)
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            old = adj[i][j]
            adj[i][j] = adj[j][i] = 1 - old
            v = _violation_count(adj, n)
            if v < best_v:
                best_v = v
                best_adj = [row[:] for row in adj]
                if v == 0:
                    break
            else:
                adj[i][j] = adj[j][i] = old
        if best_adj is not None:
            adj = best_adj
        else:
            # Stuck, try a random restart with different seed
            adj = _paley_35()
            random.seed(random.randint(0, 100000))
    
    if _violation_count(adj, n) > 0:
        # Couldn't fix 35-vertex graph, fall back to known smaller
        # Use Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}
    
    # Now try to grow to 36, 37, etc.
    best_n = n
    best_adj = [row[:] for row in adj]
    
    for target_n in range(n + 1, 41):
        if _now() > _DEADLINE - 10:
            break
        # Start from current best, add one vertex
        adj_new = [row[:] + [0] for row in adj]
        adj_new.append([0] * (target_n - 1))
        adj_new[target_n - 1] = [0] * (target_n - 1)
        
        # Connect new vertex to ~half the existing vertices
        new_v = target_n - 1
        for i in range(target_n - 1):
            if random.random() < 0.5:
                adj_new[new_v][i] = adj_new[i][new_v] = 1
        
        # Optimize the new graph
        improved = True
        while improved and _now() < _DEADLINE - 10:
            improved = False
            current_v = _violation_count(adj_new, target_n)
            if current_v == 0:
                break
            for _ in range(500):
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j:
                    continue
                old = adj_new[i][j]
                adj_new[i][j] = adj_new[j][i] = 1 - old
                v = _violation_count(adj_new, target_n)
                if v < current_v:
                    current_v = v
                    improved = True
                    if v == 0:
                        break
                else:
                    adj_new[i][j] = adj_new[j][i] = old
        
        if _violation_count(adj_new, target_n) == 0:
            best_n = target_n
            best_adj = [row[:] for row in adj_new]
            # Continue trying to grow further
        else:
            # Couldn't grow to this size, try a few more random restarts
            success = False
            for attempt in range(20):
                if _now() > _DEADLINE - 10:
                    break
                adj_try = [row[:] for row in best_adj]
                # Add vertex
                adj_try = [row[:] + [0] for row in adj_try]
                adj_try.append([0] * target_n)
                new_v = target_n - 1
                for i in range(target_n - 1):
                    if random.random() < 0.5:
                        adj_try[new_v][i] = adj_try[i][new_v] = 1
                
                # Optimize
                current_v = _violation_count(adj_try, target_n)
                for _ in range(1000):
                    if current_v == 0:
                        break
                    i = random.randint(0, target_n - 1)
                    j = random.randint(0, target_n - 1)
                    if i == j:
                        continue
                    old = adj_try[i][j]
                    adj_try[i][j] = adj_try[j][i] = 1 - old
                    v = _violation_count(adj_try, target_n)
                    if v < current_v:
                        current_v = v
                    else:
                        adj_try[i][j] = adj_try[j][i] = old
                
                if current_v == 0:
                    best_n = target_n
                    best_adj = [row[:] for row in adj_try]
                    success = True
                    break
            if not success:
                break  # Can't grow further
    
    # Extract edges from best_adj
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
