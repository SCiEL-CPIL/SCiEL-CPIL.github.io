"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_k5(adj, n):
    """Check if graph with adjacency matrix adj (n x n bool) has a K5."""
    for combo in combinations(range(n), 5):
        ok = True
        for i in range(5):
            for j in range(i+1, 5):
                if not adj[combo[i]][combo[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _has_indep5(adj, n):
    """Check if graph has an independent set of size 5."""
    for combo in combinations(range(n), 5):
        ok = True
        for i in range(5):
            for j in range(i+1, 5):
                if adj[combo[i]][combo[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _is_valid(adj, n):
    """Check no K5 and no independent 5-set."""
    if _has_k5(adj, n):
        return False
    if _has_indep5(adj, n):
        return False
    return True

def _clebsch_complement():
    """Construct the complement of the Clebsch graph (16 vertices).
    
    The Clebsch graph is the 5-dimensional halved cube, or equivalently,
    the graph on vertices {0,1}^5 where two vertices are adjacent if they
    differ in exactly 3 or 5 positions. Its complement has edges when
    vertices differ in 0, 1, or 2 positions (excluding self).
    
    Actually, let me use a simpler characterization: The Clebsch graph is
    strongly regular (16, 5, 0, 2). Its complement is strongly regular (16, 10, 6, 6).
    """
    # The Clebsch graph can be constructed from the 5-bit strings.
    # Two 5-bit strings are adjacent in the Clebsch graph if their Hamming
    # distance is 3 or 5.
    # So in the complement, they're adjacent if distance is 1 or 2.
    n = 16
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # Compute Hamming distance
            xor = i ^ j
            dist = bin(xor).count('1')
            if dist == 1 or dist == 2:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _paley_graph(q):
    """Construct Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _try_duplicate_vertex(adj, n, deadline):
    """Try to duplicate a vertex with a random connection pattern to new vertex.
    
    Pick a random existing vertex v, and create a new vertex that has a
    similar connection pattern to v (with some random perturbation).
    """
    if n == 0:
        return None
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        # Pick a random vertex to duplicate
        src = random.randint(0, n - 1)
        
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * n + [False])
        for i in range(n):
            new_adj[i].append(False)
        
        # Create new vertex with connections similar to src, with perturbation
        for i in range(n):
            if i == src:
                # Don't connect to self
                continue
            # Start with same pattern as src, then flip with some probability
            if adj[src][i]:
                # src is connected to i; new vertex connects with prob 0.8
                if random.random() < 0.8:
                    new_adj[n][i] = True
                    new_adj[i][n] = True
            else:
                # src is not connected to i; new vertex connects with prob 0.2
                if random.random() < 0.2:
                    new_adj[n][i] = True
                    new_adj[i][n] = True
        
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def _try_add_vertex_random(adj, n, deadline, prob=0.5):
    """Try to add a new vertex with a random connection pattern."""
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * n + [False])
        for i in range(n):
            new_adj[i].append(False)
        
        for i in range(n):
            if random.random() < prob:
                new_adj[n][i] = True
                new_adj[i][n] = True
        
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def _try_flip_edge(adj, n, deadline):
    """Try flipping a random edge and check validity. Return new adj if valid, else None."""
    if n < 2:
        return None
    i, j = random.sample(range(n), 2)
    new_adj = [row[:] for row in adj]
    new_adj[i][j] = not new_adj[i][j]
    new_adj[j][i] = new_adj[i][j]
    if _is_valid(new_adj, n):
        return new_adj
    return None

def _try_add_vertex_greedy(adj, n, deadline):
    """Try to add a vertex using greedy strategies."""
    # Strategy 1: Start with no edges, add one by one
    new_adj = [row[:] for row in adj]
    new_adj.append([False] * n + [False])
    for i in range(n):
        new_adj[i].append(False)
    
    if _is_valid(new_adj, n + 1):
        return new_adj
    
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            break
        new_adj[n][i] = True
        new_adj[i][n] = True
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    # Strategy 2: Start with all edges, remove one by one
    new_adj = [row[:] for row in adj]
    new_adj.append([False] * n + [False])
    for i in range(n):
        new_adj[i].append(False)
    
    for i in range(n):
        new_adj[n][i] = True
        new_adj[i][n] = True
    
    if _is_valid(new_adj, n + 1):
        return new_adj
    
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            break
        new_adj[n][i] = False
        new_adj[i][n] = False
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s to leave safety margin
    
    random.seed(42)
    
    # Try multiple strategies and keep the best result
    best_n = 0
    best_adj = None
    
    # Strategy A: Start from Clebsch complement and expand
    adj = _clebsch_complement()
    n = 16
    
    # Verify it's valid (should be: Clebsch has no K4, complement has no independent 4-set)
    # Actually, Clebsch is (16,5,0,2) so no K4. Complement is (16,10,6,6) so no independent 4-set.
    # But we need no K5 and no independent 5-set. Let's verify.
    if _is_valid(adj, n):
        best_n = n
        best_adj = [row[:] for row in adj]
    
    # Phase 1: Try to add vertices using various methods
    while n < 46 and time.time() < deadline:
        added = False
        
        # Try vertex duplication
        new_adj = _try_duplicate_vertex(adj, n, deadline)
        if new_adj is not None:
            adj = new_adj
            n += 1
            if n > best_n:
                best_n = n
                best_adj = [row[:] for row in adj]
            added = True
        
        # Try random vertex addition
        if not added:
            new_adj = _try_add_vertex_random(adj, n, deadline, prob=0.5)
            if new_adj is not None:
                adj = new_adj
                n += 1
                if n > best_n:
                    best_n = n
                    best_adj = [row[:] for row in adj]
                added = True
        
        # Try greedy vertex addition
        if not added:
            new_adj = _try_add_vertex_greedy(adj, n, deadline)
            if new_adj is not None:
                adj = new_adj
                n += 1
                if n > best_n:
                    best_n = n
                    best_adj = [row[:] for row in adj]
                added = True
        
        if not added:
            # Try edge flips to make room
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                flipped = _try_flip_edge(adj, n, deadline)
                if flipped is not None:
                    adj = flipped
                    improved = True
                    break
            if not improved:
                break
    
    # Strategy B: Also try starting from Paley graph on 37 vertices
    if time.time() < deadline:
        adj2 = _paley_graph(37)
        n2 = 37
        if _is_valid(adj2, n2) and n2 > best_n:
            best_n = n2
            best_adj = [row[:] for row in adj2]
        
        while n2 < 46 and time.time() < deadline:
            added = False
            
            new_adj = _try_duplicate_vertex(adj2, n2, deadline)
            if new_adj is not None:
                adj2 = new_adj
                n2 += 1
                if n2 > best_n:
                    best_n = n2
                    best_adj = [row[:] for row in adj2]
                added = True
            
            if not added:
                new_adj = _try_add_vertex_random(adj2, n2, deadline, prob=0.45)
                if new_adj is not None:
                    adj2 = new_adj
                    n2 += 1
                    if n2 > best_n:
                        best_n = n2
                        best_adj = [row[:] for row in adj2]
                    added = True
            
            if not added:
                new_adj = _try_add_vertex_greedy(adj2, n2, deadline)
                if new_adj is not None:
                    adj2 = new_adj
                    n2 += 1
                    if n2 > best_n:
                        best_n = n2
                        best_adj = [row[:] for row in adj2]
                    added = True
            
            if not added:
                improved = False
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    flipped = _try_flip_edge(adj2, n2, deadline)
                    if flipped is not None:
                        adj2 = flipped
                        improved = True
                        break
                if not improved:
                    break
    
    # Use the best result found
    if best_adj is None:
        # Fallback: return empty graph
        return 0, []
    
    adj = best_adj
    n = best_n
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    
    return n, edges

# EVOLVE-BLOCK-END
