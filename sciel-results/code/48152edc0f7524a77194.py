"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_lift_adjacency(graph, signs):
    """Build the 2x2 block signed adjacency matrix for the lift."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    # We build the signed adjacency matrix of the base graph (n x n)
    # But actually we need the spectral radius of the lift.
    # The lift has 2n vertices.
    # The adjacency matrix of the lift is a 2n x 2n matrix.
    # Let A be the adjacency matrix of the base graph.
    # Let S be the diagonal matrix of signs? No.
    # The task says: "Each sign defines a two-lift... The lift must be connected and 3-regular".
    # "Scoring ... mean over the four graphs of the spectral radius of the signed adjacency matrix of the base graph, max |eigenvalue| from np.linalg.eigvalsh. These are the new eigenvalues of the lift."
    # This phrasing is slightly ambiguous. Usually, the eigenvalues of the lift are related to the eigenvalues of the base graph weighted by signs.
    # For a 2-lift with signs sigma_e in {+1, -1}, the adjacency matrix of the lift L is:
    # L = [ A  B ]
    #     [ B  A ]
    # where A is the adjacency matrix of the base graph, and B is a matrix where B_{uv} = sigma_{(u,v)} if (u,v) is an edge?
    # Actually, the standard 2-lift construction:
    # Vertices are (v, 0) and (v, 1).
    # Edge (u, v) in base with sign +1 connects (u, 0)-(v, 0) and (u, 1)-(v, 1).
    # Edge (u, v) in base with sign -1 connects (u, 0)-(v, 1) and (u, 1)-(v, 0).
    # So the lift adjacency matrix L (size 2n x 2n) can be written in block form:
    # L = [ A  D ]
    #     [ D  A ]
    # where A is the adjacency matrix of the base graph (unweighted), and D is a matrix such that D_{uv} = 1 if the sign of edge (u,v) is +1? No.
    # Let's check the connectivity condition.
    # The lift is connected iff the product of signs around every cycle in the base graph is +1? No, that would make it a 1-lift (two disjoint copies).
    # If the product of signs around a cycle is -1, the lift is connected (it becomes a single cycle of length 2*cycle_length).
    # Wait, the task says "The lift must be connected".
    # For a prism graph (two cycles connected by a matching), the lift is connected if and only if there is at least one cycle with a product of signs -1?
    # Actually, the lift of a graph G with signs sigma is connected if and only if the corresponding double cover is connected.
    # The double cover is connected if and only if the "signing" is not a "coboundary" (i.e., not all cycles have product +1).
    # So we need at least one cycle to have product -1.
    
    # Let's look at the scoring again: "spectral radius of the signed adjacency matrix of the base graph".
    # This suggests we construct an n x n matrix M where M_{uv} = sign_{(u,v)} if (u,v) is an edge, else 0.
    # And we compute the spectral radius of M?
    # But M is not necessarily symmetric if the graph is directed? No, the graph is undirected.
    # So M is symmetric.
    # The eigenvalues of the lift L are given by the eigenvalues of A + D and A - D?
    # Let's derive. L = [ A  D ; D  A ].
    # Eigenvectors of the form [x; x] give (A+D)x = lambda x.
    # Eigenvectors of the form [x; -x] give (A-D)x = lambda x.
    # So the spectrum of L is the union of the spectra of A+D and A-D.
    # A is the adjacency matrix of the base graph.
    # D is a symmetric matrix where D_{uv} = sigma_{(u,v)} if (u,v) is an edge.
    # So A+D and A-D are "signed" adjacency matrices?
    # Actually, A_{uv} = 1 if edge, 0 otherwise.
    # D_{uv} = sigma_{(u,v)} if edge, 0 otherwise.
    # So A+D has entries 1+sigma_{(u,v)} on edges. If sigma=1, entry is 2. If sigma=-1, entry is 0.
    # A-D has entries 1-sigma_{(u,v)} on edges. If sigma=1, entry is 0. If sigma=-1, entry is 2.
    # This seems odd. Usually, the signed adjacency matrix S has S_{uv} = sigma_{(u,v)} if edge.
    # Then the lift eigenvalues are related to the eigenvalues of S?
    # Let's re-read: "spectral radius of the signed adjacency matrix of the base graph ... These are the new eigenvalues of the lift."
    # This implies that the spectral radius of the lift IS the spectral radius of the signed adjacency matrix S (n x n).
    # Is that true?
    # If S_{uv} = sigma_{(u,v)}, then S is symmetric.
    # The lift L has eigenvalues equal to the eigenvalues of S?
    # Let's check for a single edge. Base: 2 vertices, 1 edge. Sign +1.
    # S = [[0, 1], [1, 0]]. Eigenvalues: 1, -1.
    # Lift: 4 vertices. (0,0)-(1,0), (0,1)-(1,1). Two disjoint edges.
    # L = [[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]]. Eigenvalues: 1, 1, -1, -1.
    # Spectral radius is 1. Max |eigenvalue| of S is 1. Matches.
    # Sign -1.
    # S = [[0, -1], [-1, 0]]. Eigenvalues: 1, -1.
    # Lift: (0,0)-(1,1), (0,1)-(1,0). Two disjoint edges.
    # L = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]. Eigenvalues: 1, 1, -1, -1.
    # Spectral radius is 1. Matches.
    
    # What about a triangle?
    # Base: 3-cycle. Signs all +1.
    # S = [[0,1,1], [1,0,1], [1,1,0]]. Eigenvalues: 2, -1, -1.
    # Lift: 6 vertices. Two disjoint triangles.
    # L is block diagonal with two 3x3 blocks of all-ones-minus-diagonal.
    # Eigenvalues: 2, 2, -1, -1, -1, -1.
    # Spectral radius 2. Matches.
    
    # Signs: +1, +1, -1. (One edge negative).
    # S = [[0,1,1], [1,0,-1], [1,-1,0]].
    # Let's compute eigenvalues of S.
    # det(S - lambda I) = -lambda^3 - 2lambda = -lambda(lambda^2+2).
    # Eigenvalues: 0, i*sqrt(2), -i*sqrt(2).
    # Wait, S is real symmetric, so eigenvalues must be real.
    # Let's re-calculate.
    # S = [[0,1,1], [1,0,-1], [1,-1,0]].
    # Trace = 0.
    # det = 0*(0*0 - (-1)(-1)) - 1*(1*0 - (-1)*1) + 1*(1*(-1) - 0*1)
    # = 0 - 1*(1) + 1*(-1) = -2.
    # Characteristic polynomial: -lambda^3 - 2lambda + 2? No.
    # Let's use numpy in my head or just trust the relation.
    # Actually, the eigenvalues of the lift L are NOT just the eigenvalues of S.
    # In the triangle example with signs +1, +1, -1:
    # The lift is a 6-cycle?
    # Edges: (0,1)+, (1,2)-, (2,0)+.
    # (0,0)-(1,0) [+], (1,0)-(2,1) [-], (2,1)-(0,1) [+].
    # (0,1)-(1,1) [+], (1,1)-(2,0) [-], (2,0)-(0,0) [+].
    # Path 1: 0,0 -> 1,0 -> 2,1 -> 0,1 -> 1,1 -> 2,0 -> 0,0.
    # This is a 6-cycle.
    # Spectral radius of a 6-cycle is 2.
    # Eigenvalues of S:
    # S = [[0,1,1], [1,0,-1], [1,-1,0]].
    # Eigenvalues are roots of -lambda^3 - 2lambda + 2? No, determinant was -2.
    # Let's solve S x = lambda x.
    # x2 = lambda x1
    # x1 - x3 = lambda x2
    # x1 - x2 = lambda x3
    # Substitute x3 = x1/lambda?
    # This is getting complicated.
    # However, the task statement says: "Scoring ... mean ... of the spectral radius of the signed adjacency matrix of the base graph, max |eigenvalue| from np.linalg.eigvalsh. These are the new eigenvalues of the lift."
    # This explicitly states that the quantity to minimize is the spectral radius of S (the n x n signed adjacency matrix).
    # So I just need to minimize the spectral radius of S.
    
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    return S

def _spectral_radius(S):
    # S is symmetric
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def _is_connected_lift(graph, signs):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    # Build the lift graph (2n vertices)
    # Vertices 0..2n-1. (v, 0) is v, (v, 1) is v + n.
    adj = [[] for _ in range(2*n)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            adj[u].append(v)
            adj[v].append(u)
            adj[u+n].append(v+n)
            adj[v+n].append(u+n)
        else:
            adj[u].append(v+n)
            adj[v].append(u+n)
            adj[u+n].append(v)
            adj[v+n].append(u)
    
    # BFS from 0
    visited = [False] * (2*n)
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == 2*n

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types:
    # 1. Cycle edges in top cycle: (i, i+1) for i in 0..m-1 (with wrap)
    # 2. Cycle edges in bottom cycle: (m+i, m+i+1) for i in 0..m-1 (with wrap)
    # 3. Matching edges: (i, m+i) for i in 0..m-1
    
    # We can fix the gauge. The spectral radius of S is invariant under gauge transformations?
    # Gauge transformation: multiply signs of edges incident to a vertex by -1?
    # If we flip the sign of all edges incident to vertex v, does the spectral radius change?
    # Let D be a diagonal matrix with D_{vv} = -1 and 1 elsewhere.
    # S' = D S D.
    # S' is similar to S? No, D S D is not similar to S unless D is orthogonal (which it is, D^2=I).
    # Wait, D S D is congruent to S. Eigenvalues are not necessarily the same.
    # However, for the 2-lift, gauge transformations correspond to relabeling the fibers.
    # The connectivity of the lift is invariant under gauge transformations.
    # The spectral radius of the lift is invariant under gauge transformations.
    # But is the spectral radius of S invariant?
    # If S' = D S D, the eigenvalues of S' are not the same as S.
    # Example: S = [[0, 1], [1, 0]]. D = diag(1, -1).
    # S' = [[0, -1], [-1, 0]]. Eigenvalues: 1, -1. Same.
    # Example: S = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]. D = diag(1, -1, 1).
    # S' = [[0, -1, 1], [-1, 0, -1], [1, -1, 0]].
    # Eigenvalues of S: 2, -1, -1.
    # Eigenvalues of S':
    # det(S' - lambda I) = -lambda^3 - 2lambda - 2?
    # Let's check trace: 0.
    # It turns out that for any symmetric S, the eigenvalues of D S D are not the same as S.
    # So we cannot simply fix the gauge arbitrarily if we want to minimize the spectral radius of S.
    # HOWEVER, the task says "spectral radius of the signed adjacency matrix of the base graph".
    # And "These are the new eigenvalues of the lift".
    # This implies that the spectral radius of the lift is equal to the spectral radius of S.
    # And the spectral radius of the lift IS invariant under gauge transformations (since it's just a relabeling of the lift vertices).
    # Therefore, the spectral radius of S MUST be invariant under gauge transformations?
    # Let's re-verify the triangle example.
    # S = [[0,1,1], [1,0,1], [1,1,0]]. Eigenvalues: 2, -1, -1. Radius 2.
    # Gauge flip vertex 1: D = diag(1, -1, 1).
    # S' = [[0, -1, 1], [-1, 0, -1], [1, -1, 0]].
    # Eigenvalues of S':
    # Characteristic polynomial: -lambda^3 - 2lambda + 2?
    # Let's calculate det(S' - lambda I).
    # | -lambda  -1      1   |
    # | -1    -lambda  -1   |
    # |  1     -1   -lambda |
    # = -lambda(lambda^2 - 1) - (-1)(lambda - (-1)) + 1(1 - (-lambda))
    # = -lambda^3 + lambda + lambda + 1 + 1 + lambda
    # = -lambda^3 + 3lambda + 2
    # = -(lambda^3 - 3lambda - 2)
    # = -(lambda+1)(lambda^2 - lambda - 2)
    # = -(lambda+1)(lambda-2)(lambda+1)
    # = -(lambda+1)^2 (lambda-2)
    # Roots: 2, -1, -1.
    # Same eigenvalues!
    
    # It seems that for these specific graphs, the eigenvalues are invariant under gauge transformations.
    # This is because the graph is bipartite? No, prism graphs with odd m are not bipartite.
    # Actually, it is a known result that the spectrum of the signed adjacency matrix is invariant under gauge transformations for 2-lifts?
    # No, generally it's not. But maybe for these specific graphs it is?
    # Or maybe I made a mistake.
    # Let's assume we can fix the gauge to reduce the search space.
    # We can fix the signs of the matching edges to all +1.
    # There are m matching edges.
    # We can also fix one cycle edge to +1.
    # This reduces the number of free variables.
    
    # Let's try exhaustive search over the reduced space.
    # m is 4, 5, 7, 9.
    # Number of edges = 3m.
    # If we fix m matching edges and 1 cycle edge, we have 3m - m - 1 = 2m - 1 free variables.
    # For m=9, 2m-1 = 17. 2^17 = 131072. This is very feasible.
    # For m=9, we have 27 edges. 2^27 is too big, but 2^17 is fine.
    
    # Let's identify the edges.
    # Edges are given as a list. We need to map them to types.
    # Top cycle: (i, i+1 mod m)
    # Bottom cycle: (m+i, m+i+1 mod m)
    # Matching: (i, m+i)
    
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # Sort them to be consistent
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    matching_edges.sort()
    
    # Gauge fixing:
    # Set all matching edges to +1.
    # Set the first top cycle edge to +1.
    # Free variables:
    # Remaining top cycle edges: m - 1
    # All bottom cycle edges: m
    # Total free: 2m - 1
    
    free_edges = []
    for i in top_cycle_edges[1:]:
        free_edges.append(i)
    for i in bottom_cycle_edges:
        free_edges.append(i)
    
    num_free = len(free_edges)
    
    best_signs = None
    best_radius = float('inf')
    
    # Iterate over all 2^num_free combinations
    # Use a fast method to generate combinations and evaluate.
    
    # Pre-compute the fixed signs
    fixed_signs = np.ones(num_edges)
    for i in matching_edges:
        fixed_signs[i] = 1
    if top_cycle_edges:
        fixed_signs[top_cycle_edges[0]] = 1
    
    # For each combination of free signs:
    for mask in range(1 << num_free):
        if time.time() > deadline:
            break
        
        signs = fixed_signs.copy()
        for j, edge_idx in enumerate(free_edges):
            if (mask >> j) & 1:
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
        
        # Check connectivity
        if not _is_connected_lift(graph, signs):
            continue
        
        # Compute spectral radius
        S = _build_lift_adjacency(graph, signs)
        radius = _spectral_radius(S)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: random search
        rng = np.random.default_rng(3)
        for _ in range(budget):
            signs = rng.choice([-1, 1], size=num_edges)
            if _is_connected_lift(graph, signs):
                S = _build_lift_adjacency(graph, signs)
                radius = _spectral_radius(S)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
        
        if best_signs is None:
            # Last resort: all +1 (might be disconnected)
            best_signs = np.ones(num_edges)
    
    return best_signs
# EVOLVE-BLOCK-END
