"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_k5_fast(adj, n):
    """Fast check: does the graph contain a K5? Uses neighborhood intersection."""
    # For each 4-clique, check if any vertex is connected to all 4
    # First find all 4-cliques
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            # Find common neighbors of a and b
            common_ab = [c for c in range(b+1, n) if adj[a][c] and adj[b][c]]
            for c in common_ab:
                for d in range(c+1, n):
                    if not adj[c][d]:
                        continue
                    if not adj[a][d] or not adj[b][d]:
                        continue
                    # {a,b,c,d} is a 4-clique; check if any vertex connects to all
                    # Check vertices that connect to all of a,b,c,d
                    # Only need to check vertices > d to avoid duplicates, but also need to check
                    # Actually, any vertex connected to all 4 forms a K5
                    # We can check if there's a common neighbor of all 4
                    # Use intersection of neighborhoods
                    common_abc = [e for e in range(d+1, n) if adj[a][e] and adj[b][e] and adj[c][e] and adj[d][e]]
                    if common_abc:
                        return True
    return False

def _has_indep5_fast(adj, n):
    """Fast check: does the graph contain an independent set of size 5?
    Equivalent to checking if the complement graph has a K5."""
    for a in range(n):
        for b in range(a+1, n):
            if adj[a][b]:
                continue
            # a and b are non-adjacent; find common non-neighbors
            common_non_ab = [c for c in range(b+1, n) if not adj[a][c] and not adj[b][c]]
            for c in common_non_ab:
                for d in range(c+1, n):
                    if adj[c][d]:
                        continue
                    if adj[a][d] or adj[b][d]:
                        continue
                    # {a,b,c,d} is an independent 4-set; check for a 5th
                    common_non_abc = [e for e in range(d+1, n) if not adj[a][e] and not adj[b][e] and not adj[c][e] and not adj[d][e]]
                    if common_non_abc:
                        return True
    return False

def _is_valid(adj, n):
    """Check no K5 and no independent 5-set."""
    if _has_k5_fast(adj, n):
        return False
    if _has_indep5_fast(adj, n):
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
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Pick a random vertex to duplicate
        src = random.randint(0, n - 1)
        
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * n + [False])
        for i in range(n):
            new_adj[i].append(False)
        
        # Create new vertex with connections similar to src, with perturbation
        # Use different probabilities based on attempt count for diversity
        p_keep = 0.7 + 0.1 * (attempt % 5) / 5.0
        p_flip = 0.1 + 0.1 * (attempt % 3) / 3.0
        
        for i in range(n):
            if i == src:
                continue
            if adj[src][i]:
                if random.random() < p_keep:
                    new_adj[n][i] = True
                    new_adj[i][n] = True
            else:
                if random.random() < p_flip:
                    new_adj[n][i] = True
                    new_adj[i][n] = True
        
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def _try_add_vertex_random(adj, n, deadline, prob=0.5):
    """Try to add a new vertex with a random connection pattern."""
    for attempt in range(300):
        if time.time() > deadline:
            return None
        
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * n + [False])
        for i in range(n):
            new_adj[i].append(False)
        
        # Vary the probability slightly per attempt
        p = prob + 0.1 * (random.random() - 0.5)
        p = max(0.2, min(0.8, p))
        
        for i in range(n):
            if random.random() < p:
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
    # Strategy 1: Start with no edges, add one by one in random order
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
    
    # Strategy 3: Start from a vertex's neighborhood pattern
    for src in range(min(n, 10)):
        if time.time() > deadline:
            break
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * n + [False])
        for i in range(n):
            new_adj[i].append(False)
        for i in range(n):
            if i != src and adj[src][i]:
                new_adj[n][i] = True
                new_adj[i][n] = True
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s to leave safety margin
    
    random.seed(42)
    
    # Try multiple strategies and keep the best result
    best_n = 0
    best_adj = None
    
    # Strategy A: Start from Clebsch complement and expand
    adj = _clebsch_complement()
    n = 16
    
    if _is_valid(adj, n):
        best_n = n
        best_adj = [row[:] for row in adj]
    
    # Phase 1: Try to add vertices using various methods
    stuck_count = 0
    while n < 46 and time.time() < deadline:
        added = False
        
        # Try vertex duplication
        new_adj = _try_duplicate_vertex(adj, n, deadline)
        if new_adj is not None:
            adj = new_adj
            n += 1
            stuck_count = 0
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
                stuck_count = 0
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
                stuck_count = 0
                if n > best_n:
                    best_n = n
                    best_adj = [row[:] for row in adj]
                added = True
        
        if not added:
            # Try edge flips to make room
            improved = False
            for _ in range(150):
                if time.time() > deadline:
                    break
                flipped = _try_flip_edge(adj, n, deadline)
                if flipped is not None:
                    adj = flipped
                    improved = True
                    break
            if not improved:
                stuck_count += 1
                if stuck_count > 5:
                    break
    
    # Strategy B: Also try starting from Paley graph on 37 vertices
    if time.time() < deadline:
        adj2 = _paley_graph(37)
        n2 = 37
        if _is_valid(adj2, n2) and n2 > best_n:
            best_n = n2
            best_adj = [row[:] for row in adj2]
        
        stuck_count2 = 0
        while n2 < 46 and time.time() < deadline:
            added = False
            
            new_adj = _try_duplicate_vertex(adj2, n2, deadline)
            if new_adj is not None:
                adj2 = new_adj
                n2 += 1
                stuck_count2 = 0
                if n2 > best_n:
                    best_n = n2
                    best_adj = [row[:] for row in adj2]
                added = True
            
            if not added:
                new_adj = _try_add_vertex_random(adj2, n2, deadline, prob=0.45)
                if new_adj is not None:
                    adj2 = new_adj
                    n2 += 1
                    stuck_count2 = 0
                    if n2 > best_n:
                        best_n = n2
                        best_adj = [row[:] for row in adj2]
                    added = True
            
            if not added:
                new_adj = _try_add_vertex_greedy(adj2, n2, deadline)
                if new_adj is not None:
                    adj2 = new_adj
                    n2 += 1
                    stuck_count2 = 0
                    if n2 > best_n:
                        best_n = n2
                        best_adj = [row[:] for row in adj2]
                    added = True
            
            if not added:
                improved = False
                for _ in range(80):
                    if time.time() > deadline:
                        break
                    flipped = _try_flip_edge(adj2, n2, deadline)
                    if flipped is not None:
                        adj2 = flipped
                        improved = True
                        break
                if not improved:
                    stuck_count2 += 1
                    if stuck_count2 > 5:
                        break
    
    # Strategy C: Try additional random restarts if time remains
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            random.seed(42 + restart * 1000)
            adj3 = _paley_graph(37)
            n3 = 37
            if not _is_valid(adj3, n3):
                continue
            
            for step in range(9):
                if time.time() > deadline - 5:
                    break
                added = False
                new_adj = _try_duplicate_vertex(adj3, n3, deadline)
                if new_adj is not None:
                    adj3 = new_adj
                    n3 += 1
                    added = True
                if not added:
                    new_adj = _try_add_vertex_random(adj3, n3, deadline, prob=0.5)
                    if new_adj is not None:
                        adj3 = new_adj
                        n3 += 1
                        added = True
                if not added:
                    new_adj = _try_add_vertex_greedy(adj3, n3, deadline)
                    if new_adj is not None:
                        adj3 = new_adj
                        n3 += 1
                        added = True
                if not added:
                    break
            
            if n3 > best_n:
                best_n = n3
                best_adj = [row[:] for row in adj3]
    
    # Use the best result found
    if best_adj is None:
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
