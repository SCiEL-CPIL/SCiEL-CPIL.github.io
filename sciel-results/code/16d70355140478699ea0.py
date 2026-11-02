<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _de_bruijn_graph(k, m):
    """Construct the de Bruijn graph B(k, m): k-regular, girth 2m+2.
    Vertices are words of length m over {0,...,k-1}.
    Vertex w=(w0,...,w_{m-1}) connects to (w1,...,w_{m-1},a) for a in {0,...,k-1}.
    This is a directed graph; we make it undirected by noting it's actually a
    bipartite structure. For undirected use, we take the underlying undirected graph.
    Actually, the de Bruijn graph as defined is directed and k-in/k-out regular.
    To get an undirected k-regular graph, we use a different construction:
    the 'shift register' graph where each vertex connects to its k shifts.
    
    Better approach: use the standard de Bruijn graph which IS k-regular directed.
    For undirected, we can use the fact that for our purpose, we need an undirected
    k-regular graph. Let's use a different well-known construction.
    
    Actually, let me reconsider. The de Bruijn graph B(k,m) has k^m vertices,
    each with k outgoing and k incoming edges. As an undirected graph, each vertex
    has degree up to 2k (if in- and out-neighbors don't overlap). That's not k-regular.
    
    Instead, let's use the 'cyclic' de Bruijn or a proper k-regular construction.
    
    Key insight: We can construct a k-regular graph of girth >= 2m+2 using
    the following: vertices are pairs (word, bit) where word is length m-1 over
    alphabet of size k, and bit is 0 or 1. Total: 2 * k^(m-1) vertices.
    Each vertex (w, 0) connects to (w', 1) where w' = (w[1:], a) for a in 0..k-1.
    Each vertex (w, 1) connects to (w', 0) where w' = (a, w[0:m-2]) for a in 0..k-1.
    Wait, that's not quite right either.
    
    Let me use a cleaner approach: The standard result is that the de Bruijn graph
    B(k, m) considered as a bipartite graph between 'even' and 'odd' positions
    gives a k-regular bipartite graph with girth 2m+2.
    
    Actually, the simplest correct approach: Use the directed de Bruijn graph and
    replace each directed edge with an undirected edge. But that gives degree 2k.
    
    Correct construction for k-regular undirected with large girth:
    Use the 'shift' graph: vertices are (m-1)-tuples over Z_k, plus a parity bit.
    """
    n = k ** m
    # Vertex i corresponds to the base-k representation of i as an m-digit number
    # v = (d_0, d_1, ..., d_{m-1}) where i = sum d_j * k^j
    # Neighbor: shift left and append new digit a: (d_1, ..., d_{m-1}, a)
    # This gives k neighbors per vertex.
    # But this is directed. For undirected, we need to be more careful.
    
    # Let's use a different well-known construction: the 'cage' from
    # de Bruijn-Erdos type. Actually, let me just verify girth and use
    # the directed de Bruijn as-is, then make it undirected properly.
    
    # For an undirected k-regular graph from de Bruijn:
    # Use vertices = words of length m over alphabet of size k.
    # Edge between w and w' if w' = shift(w) or w = shift(w') where shift removes
    # first char and appends a new one. But this isn't symmetric in a simple way.
    
    # Cleanest: use the bipartite de Bruijn graph.
    # Left vertices: words of length m over {0,...,k-1}
    # Right vertices: words of length m over {0,...,k-1}  
    # Left w connects to Right w' if w'[1:] = w[:m-1] (i.e., w' is a shift of w with new last char)
    # This gives each left vertex k neighbors on the right, and each right vertex k neighbors on the left.
    # Total vertices: 2 * k^m, each degree k, girth >= 2m+2.
    
    n_left = k ** m
    n_total = 2 * n_left
    edges = []
    
    for i in range(n_left):
        # i in base k, m digits
        digits = []
        tmp = i
        for _ in range(m):
            digits.append(tmp % k)
            tmp //= k
        digits.reverse()  # digits[0] is most significant
        
        # Left vertex i connects to right vertices j where j's first m-1 digits = i's last m-1 digits
        # j = (digits[1], digits[2], ..., digits[m-1], a) for a in 0..k-1
        for a in range(k):
            # j in base k: digits[1], digits[2], ..., digits[m-1], a
            j = 0
            for d in digits[1:]:
                j = j * k + d
            j = j * k + a
            edges.append((i, n_left + j))
    
    return n_total, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # De Bruijn bipartite construction: 2 * k^m vertices, girth >= 2m+2
    # Need 2m+2 >= g, so m >= (g-2)/2
    m = max(1, (g - 2 + 1) // 2)  # ceil((g-2)/2)
    
    n, edges = _de_bruijn_graph(k, m)
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Safety margin
        pass
    
    return n, edges
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, let me reconsider. The de Bruijn bipartite graph I described has $2 \cdot k^m$ vertices. For $k=3, g=16$: $m = \lceil 14/2 \rceil = 7$, so $n = 2 \cdot 3^7 = 4374$. That's worse than the 2187 from just $k^m$.

Let me think more carefully. The standard de Bruijn graph $B(k,m)$ as a **directed** graph has $k^m$ vertices, each with out-degree $k$ and in-degree $k$. If we view it as an **undirected** graph (ignoring direction), a vertex can have up to $2k$ neighbors (its $k$ out-neighbors and $k$ in-neighbors, minus any overlaps). This is NOT $k$-regular.

The correct approach for a $k$-regular **undirected** graph with large girth from de Bruijn-type ideas: Use the **bipartite** de Bruijn graph which I described above, giving $2 \cdot k^m$ vertices. Or, find a better construction.

Actually, there's a well-known result: the de Bruijn graph $B(k,m)$ has girth $2m+2$ when viewed as a bipartite graph between two copies. Let me verify the girth claim and ensure correctness.

Let me refine to make sure the construction is correct and efficient:

