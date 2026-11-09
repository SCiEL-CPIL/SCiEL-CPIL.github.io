"""
Small 4-regular graph of girth 10 (cage problem n(4,10)).

Strategy:
  1. Start from the Lazebnik-Ustimenko-Woldar (LUW) algebraic construction, which
     gives a valid 4-regular graph with girth >= 10.
  2. Run a simulated-annealing-style local search using 2-switches (edge swaps
     that preserve k-regularity) to try to reduce the number of vertices while
     maintaining girth >= 10.
  3. If the search doesn't improve, return the original LUW graph.

  The LUW graph for k=4, g=10 has 320 vertices. The theoretical lower bound is
  243. We attempt to find a smaller graph via stochastic local search.

  Key insight: A 2-switch on a k-regular graph preserves k-regularity. We can
  use this to explore the space of k-regular graphs. The objective is to
  minimize n while keeping girth >= g. However, changing n requires adding/
  removing vertices, which is more complex. Instead, we focus on:
  - Trying to find a graph with fewer vertices by starting from smaller random
    k-regular graphs and trying to increase their girth.
  - Using the LUW graph as a fallback if no better solution is found.

  For the cage problem, we want the SMALLEST n such that a k-regular graph with
  girth >= g exists. So we try n from the Moore bound upward, and for each n,
  try to find a k-regular graph with girth >= g.

  Given the 120s time limit, we need to be efficient. We'll:
  - Compute the Moore bound for (k=4, g=10).
  - Try random 4-regular graphs with n vertices, starting from the Moore bound,
    and use local search to increase girth.
  - If we find a valid graph, return it.
  - Otherwise, return the LUW graph.

  The Moore bound for a k-regular graph of girth g:
  For even g = 2r: n >= k * (1 + (k-1) + ... + (k-1)^(r-1))
  For odd g = 2r+1: n >= 1 + k * (1 + (k-1) + ... + (k-1)^(r-1))

  For k=4, g=10 (even, r=5):
  n >= 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484? No wait.

  Actually, the Moore bound for a k-regular graph of girth g:
  For even g = 2r: n >= 2 * (1 + (k-1) + ... + (k-1)^(r-1)) * ... 

  Let me recall correctly. The Moore bound M(k, g):
  If g = 2r (even): M = k * sum_{i=0}^{r-1} (k-1)^i
  If g = 2r+1 (odd): M = 1 + k * sum_{i=0}^{r-1} (k-1)^i

  For k=4, g=10: g is even, r=5.
  M = 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484.

  Wait, that doesn't match. The known lower bound is 243. Let me recheck.

  Actually, the Moore bound formula:
  For a k-regular graph of girth g:
  - If g = 2r (even): n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
  - If g = 2r+1 (odd): n >= 1 + k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))

  For k=4, g=10, r=5:
  n >= 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484.

  But the problem statement says the best lower bound is 243. So my formula must be wrong.

  Let me look this up mentally. The Moore bound for cages:
  For a k-regular graph with girth g:
  If g is even, g = 2r: n >= k * sum_{i=0}^{r-1} (k-1)^i
  If g is odd, g = 2r+1: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i

  For k=3, g=5 (odd, r=2): n >= 1 + 3*(1+2) = 1+9 = 10. The Petersen graph has 10 vertices. Correct.
  For k=3, g=6 (even, r=3): n >= 3*(1+2+4) = 21. The smallest 3-regular graph of girth 6 has 22 vertices (the Robertson graph? No, it's 22). Actually the (3,6)-cage has 22 vertices. So 21 is the lower bound, 22 is achieved.

  For k=4, g=10 (even, r=5): n >= 4*(1+3+9+27+81) = 4*121 = 484.

  Hmm, but the problem says the lower bound is 243. So either my formula is wrong or I'm misunderstanding.

  Actually, I think the issue is that the Moore bound for even girth is:
  n >= 2 * sum_{i=0}^{r-1} (k-1)^i * ... 

  Let me think differently. From a vertex, in a tree of depth r-1, we have 1 + k + k(k-1) + ... + k(k-1)^(r-2) vertices. For even girth 2r, the graph is bipartite-like, and the bound is:
  n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))

  For k=4, r=5: 2*(1+3+9+27+81) = 2*121 = 242. Close to 243.

  So the Moore bound is 242 for k=4, g=10. The best known lower bound is 243 (slightly stronger).

  OK so the Moore bound is 242. We want to find a 4-regular graph with girth >= 10 on as few vertices as possible, starting from 242.

  Given the complexity, a full search from 242 is likely infeasible in 120s. The LUW construction gives 320. Let's try to improve from 320 downward, or try random constructions at various sizes.

  Given time constraints, I'll implement:
  1. LUW construction as baseline (320 vertices).
  2. Try to find a smaller graph via random initialization + local search for girth improvement, starting at n=280, 260, 250, 245, 242.
  3. Return the best valid graph found.

  For efficiency, I'll use a fast girth check via BFS from each vertex up to depth (g-1)/2 or g/2.
"""

import time
import random
import sys
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _compute_girth(n, edges, max_girth=12):
    """Compute the girth of a graph using BFS from each vertex."""
    if not edges:
        return float('inf')
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    girth = float('inf')
    
    for start in range(n):
        # BFS up to depth (girth-1)//2 or max_depth
        max_depth = (max_girth - 1) // 2
        if max_depth < 2:
            max_depth = 2
        
        # dist[v] = distance from start to v, parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        found_cycle = None
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
                        found_cycle = cycle_len
                    if girth <= max_girth:
                        break
            if found_cycle and girth <= max_girth:
                break
        if girth <= max_girth:
            break
    
    return girth


def _is_valid_graph(n, edges, k, g):
    """Check if the graph is k-regular, simple, and has girth >= g."""
    if len(edges) != n * k // 2:
        return False
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    girth = _compute_girth(n, edges, max_girth=g)
    return girth >= g


def _random_k_regular(n, k, seed=None):
    """Generate a random k-regular graph using configuration model + 2-switches."""
    if seed is not None:
        random.seed(seed)
    
    # Configuration model
    half_edges = []
    for i in range(n):
        half_edges.extend([i] * k)
    random.shuffle(half_edges)
    
    edges = []
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u != v:
            edges.append((min(u, v), max(u, v)))
    
    # Fix self-loops and multi-edges with 2-switches
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Remove self-loops
    new_edges = []
    for u, v in edges:
        if u != v:
            new_edges.append((u, v))
    edges = new_edges
    
    # Use 2-switches to fix multi-edges and ensure regularity
    for _ in range(1000):
        # Check for multi-edges
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        has_multiedge = False
        for i in range(n):
            if len(adj[i]) != len(set(adj[i])):
                has_multiedge = True
                break
            if len(adj[i]) != k:
                has_multiedge = True
                break
        
        if not has_multiedge:
            break
        
        # Try a 2-switch
        u, v = random.sample(range(n), 2)
        if len(adj[u]) >= 2 and len(adj[v]) >= 2:
            w = random.choice(adj[u])
            x = random.choice(adj[v])
            if w != x and w != v and x != u and (min(u, v), max(u, v)) not in [(min(w, x), max(w, x))]:
                # Check if u-w, v-x, u-x, v-w are valid
                if (min(u, w), max(u, w)) in [(min(a, b), max(a, b)) for a, b in edges]:
                    if (min(v, x), max(v, x)) in [(min(a, b), max(a, b)) for a, b in edges]:
                        if (min(u, x), max(u, x)) not in [(min(a, b), max(a, b)) for a, b in edges]:
                            if (min(v, w), max(v, w)) not in [(min(a, b), max(a, b)) for a, b in edges]:
                                # Perform switch
                                new_edges = []
                                for a, b in edges:
                                    if (a, b) == (min(u, w), max(u, w)) or (a, b) == (min(v, x), max(v, x)):
                                        continue
                                    new_edges.append((a, b))
                                new_edges.append((min(u, x), max(u, x)))
                                new_edges.append((min(v, w), max(v, w)))
                                edges = new_edges
    
    # Verify
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None
    
    return edges


def _two_switch(edges, n, k, adj, u, v, w, x):
    """Apply a 2-switch: remove edges u-w and v-x, add u-x and v-w."""
    new_edges = []
    for a, b in edges:
        if (a, b) == (min(u, w), max(u, w)) or (a, b) == (min(v, x), max(v, x)):
            continue
        new_edges.append((a, b))
    new_edges.append((min(u, x), max(u, x)))
    new_edges.append((min(v, w), max(v, w)))
    return new_edges


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Get LUW baseline
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Compute Moore bound
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # Try to find smaller graphs
    # We'll try n values from max(moore, 240) up to best_n - 1
    target_ns = []
    for n_try in range(max(moore, 240), best_n):
        target_ns.append(n_try)
    
    # Try in order of increasing n (smallest first)
    random.seed(42)
    
    for n_try in target_ns:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            edges = _random_k_regular(n_try, k, seed=random.randint(0, 1000000))
            if edges is None:
                continue
            
            # Check girth
            girth = _compute_girth(n_try, edges, max_girth=g)
            if girth >= g:
                best_n = n_try
                best_edges = edges
                found = True
                break
            
            # Local search: try 2-switches to increase girth
            adj = [[] for _ in range(n_try)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            current_girth = girth
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random 2-switch
                u, v = random.sample(range(n_try), 2)
                if len(adj[u]) < 2 or len(adj[v]) < 2:
                    continue
                w = random.choice(adj[u])
                x = random.choice(adj[v])
                
                # Check validity
                if w == x or w == v or x == u:
                    continue
                if (min(u, v), max(u, v)) == (min(w, x), max(w, x)):
                    continue
                if (min(u, w), max(u, w)) == (min(v, x), max(v, x)):
                    continue
                
                # Check if new edges would create multi-edges
                if (min(u, x), max(u, x)) in [(min(a, b), max(a, b)) for a, b in edges]:
                    continue
                if (min(v, w), max(v, w)) in [(min(a, b), max(a, b)) for a, b in edges]:
                    continue
                
                new_edges = _two_switch(edges, n_try, k, adj, u, v, w, x)
                
                # Quick girth check
                new_girth = _compute_girth(n_try, new_edges, max_girth=g)
                if new_girth >= current_girth:
                    edges = new_edges
                    current_girth = new_girth
                    if current_girth >= g:
                        best_n = n_try
                        best_edges = edges
                        found = True
                        break
                    # Update adj
                    adj = [[] for _ in range(n_try)]
                    for a, b in edges:
                        adj[a].append(b)
                        adj[b].append(a)
        
        if found:
            break
    
    return best_n, best_edges
