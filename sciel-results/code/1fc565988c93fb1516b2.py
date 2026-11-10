"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

N = 31
M = 91

def count_c4(adj):
    """Count C4s from numpy adjacency matrix."""
    common = adj @ adj
    c4 = 0
    for u in range(N):
        for v in range(u+1, N):
            c = common[u,v]
            if c >= 2:
                c4 += c*(c-1)//2
    return c4

def make_adj(edges):
    adj = np.zeros((N, N), dtype=np.int32)
    for u, v in edges:
        adj[u,v] = 1
        adj[v,u] = 1
    return adj

def spectral_cayley_init():
    """
    Construct a Cayley graph on Z_5 x Z_5 using Kronecker product structure.
    A = I_5 ⊗ B + B ⊗ I_5 where B is a 5x5 circulant matrix.
    This gives a 25-vertex graph with structured low C4 count.
    Then add 6 more edges to reach 91 total.
    """
    # B is a 5x5 circulant: each vertex connects to vertices at distance 1 and 2
    # This gives degree 4 in each component
    # A = I_5 ⊗ B + B ⊗ I_5 gives degree 8 on Z_5 x Z_5 (25 vertices)
    # That's 25*8/2 = 100 edges, too many.
    
    # Let's use a sparser B: only distance 1 (5-cycle)
    # A = I_5 ⊗ C5 + C5 ⊗ I_5 gives degree 4, so 25*4/2 = 50 edges
    # Need 41 more edges to reach 91.
    
    # Alternative: use B as complete graph K5 minus self-loops
    # A = I_5 ⊗ (K5-I) + (K5-I) ⊗ I_5
    # Each vertex in Z_5 x Z_5 connects to 4 + 4 = 8 neighbors
    # 25*8/2 = 100 edges. Still too many.
    
    # Let me use a different approach: define the graph directly on 31 vertices
    # using the polarity graph of PG(2,5) as base, then optimize.
    
    # Actually, let's construct a proper Cayley graph on Z_5 x Z_5 with 90 edges
    # and add 1 edge.
    
    # For a Cayley graph on Z_5 x Z_5 with connection set S:
    # |E| = 25 * |S| / 2 (if S is symmetric, no identity)
    # For 90 edges: 25 * |S| / 2 = 90 => |S| = 7.2, not integer.
    
    # For 91 edges: not a regular Cayley graph.
    
    # Let's use a semi-regular approach: most vertices have degree 6, some have degree 5.
    # Total degree sum = 182. If 27 vertices have degree 6 and 4 have degree 5:
    # 27*6 + 4*5 = 162 + 20 = 182. 
    
    # Construct using algebraic structure:
    # Vertices are (i,j) in Z_5 x Z_5 plus 6 special vertices.
    # Base graph on Z_5 x Z_5 with 85 edges (degree ~6-7), then add 6 edges to special vertices.
    
    # Simpler: use the known polarity graph construction for PG(2,5)
    # The polarity graph of PG(2,q) has q^2+q+1 vertices and q(q+1)/2 * (q+1) edges?
    # For q=5: 31 vertices, and the number of edges is...
    
    # The standard polarity graph (Erdős–Rényi) for PG(2,q):
    # Vertices: points of PG(2,q), so q^2+q+1 = 31 for q=5
    # Edges: two points are adjacent if they are orthogonal under the polarity
    # This gives a graph with exactly q(q+1)/2 * something...
    
    # Actually, for the classical polarity (orthogonal polarity), each point is adjacent to q+1 points
    # including itself (absolute points). Excluding self-loops, each vertex has degree q or q+1.
    # For q=5: most vertices have degree 5, some have degree 6.
    # Total edges: (31*5 + 6*1)/2 = (155+6)/2 = 80.5? That doesn't work.
    
    # Let me just use the known result: the Erdős–Rényi polarity graph ER(q) has
    # q^2+q+1 vertices and q(q+1)/2 * (q+1) edges? No...
    
    # The correct count: for the standard polarity, the number of edges is q(q+1)/2 * (q+1)
    # Wait, let me think differently.
    
    # Each line in PG(2,q) has q+1 points. The polarity maps points to lines.
    # A point P is adjacent to Q if P lies on the polar line of Q.
    # For each point P, the polar line has q+1 points, one of which is P itself (if P is absolute).
    # So P is adjacent to q or q+1 other points.
    
    # Number of absolute points: for a non-singular polarity in PG(2,q) with q odd,
    # there are q+1 absolute points.
    
    # So: (q+1) vertices have degree q+1, and q^2 vertices have degree q.
    # Total degree = (q+1)(q+1) + q^2 * q = (q+1)^2 + q^3
    # For q=5: 36 + 125 = 161. Not even, so something's off.
    
    # Let me just use a greedy construction with the spectral insight.
    
    # Create a 31x31 matrix based on modular arithmetic
    adj = np.zeros((N, N), dtype=np.int32)
    edges = []
    
    # Use the structure: vertices 0-24 are (i,j) in Z_5 x Z_5, vertices 25-30 are extra
    # Connect (i,j) to (i+a, j+b) for (a,b) in a connection set S
    # Choose S to minimize C4s
    
    # Connection set: use elements that are "independent" in the Fourier sense
    # S = {(1,0), (0,1), (1,1), (2,0), (0,2), (1,2), (2,1)}
    # This gives degree 7 for most vertices, 25*7/2 = 87.5, not integer.
    
    # Use S with 8 elements (symmetric): degree 8, 25*8/2 = 100 edges
    # Too many. Use 7 elements but make it work.
    
    # Let's use a different approach: construct the graph greedily
    # starting from a structured initialization.
    
    # Initialize with a grid-like structure
    for i in range(5):
        for j in range(5):
            v = i * 5 + j
            # Connect to horizontal and vertical neighbors (torus)
            for a in [1, 2]:
                v2 = (i + a) % 5 * 5 + j
                if v2 > v:
                    edges.append([v, v2])
                    adj[v, v2] = 1
                    adj[v2, v] = 1
                v3 = i * 5 + (j + a) % 5
                if v3 > v:
                    edges.append([v, v3])
                    adj[v, v3] = 1
                    adj[v3, v] = 1
            # Diagonal connections
            v4 = (i + 1) % 5 * 5 + (j + 1) % 5
            if v4 > v:
                edges.append([v, v4])
                adj[v, v4] = 1
                adj[v4, v] = 1
    
    # Now add edges involving vertices 25-30
    for extra in range(25, 31):
        # Connect to 3-4 vertices in the grid
        targets = []
        for i in range(5):
            for j in range(5):
                v = i * 5 + j
                if v not in targets and len(targets) < 4:
                    targets.append(v)
        for v in targets[:3]:
            if adj[extra, v] == 0:
                edges.append([extra, v])
                adj[extra, v] = 1
                adj[v, extra] = 1
        if len(edges) >= M:
            break
    
    # If we have too many edges, remove some
    if len(edges) > M:
        edges = edges[:M]
        adj = make_adj(edges)
    
    # If we have too few, add more
    if len(edges) < M:
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        for i in range(N):
            for j in range(i+1, N):
                if len(edges) >= M:
                    break
                if (i, j) not in edge_set:
                    edges.append([i, j])
                    edge_set.add((i, j))
                    break
    
    return edges

def local_search(edges, time_budget):
    """Hill-climbing with 2-opt moves and fast delta computation."""
    adj = make_adj(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = list(edges)
    
    start_time = time.time()
    deadline = start_time + time_budget
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                return best_edges
            u1, v1 = edges[i]
            e1 = (min(u1,v1), max(u1,v1))
            
            # Compute C4s containing edge (u1,v1)
            common_neighbors = np.where(adj[u1] & adj[v1])[0]
            remove_delta = len(common_neighbors) * (len(common_neighbors) - 1) // 2
            
            adj[u1,v1] = 0
            adj[v1,u1] = 0
            edge_set.discard(e1)
            
            best_move = None
            best_delta = 0
            
            for u2, v2 in all_pairs:
                e2 = (min(u2,v2), max(u2,v2))
                if e2 in edge_set:
                    continue
                # Compute C4s created by adding edge (u2,v2)
                common_new = np.where(adj[u2] & adj[v2])[0]
                add_delta = len(common_new) * (len(common_new) - 1) // 2
                delta = add_delta - remove_delta
                if delta < best_delta:
                    best_delta = delta
                    best_move = (u2, v2)
            
            if best_move is not None:
                u2, v2 = best_move
                e2 = (min(u2,v2), max(u2,v2))
                adj[u2,v2] = 1
                adj[v2,u2] = 1
                edge_set.add(e2)
                edges[i] = [u2, v2]
                current_c4 += best_delta
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = list(edges)
                improved = True
            else:
                adj[u1,v1] = 1
                adj[v1,u1] = 1
                edge_set.add(e1)
        
        if not improved:
            break
    
    return best_edges

def construct():
    """Main entry: spectral init + local search."""
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = None
    best_c4 = float('inf')
    
    # Try spectral initialization
    edges = spectral_cayley_init()
    if len(edges) == M:
        adj = make_adj(edges)
        c4 = count_c4(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = list(edges)
    
    # Local search from spectral init
    if best_edges is not None and time.time() < deadline - 10:
        ls_edges = local_search(list(best_edges), 10)
        adj_ls = make_adj(ls_edges)
        c4_ls = count_c4(adj_ls)
        if c4_ls < best_c4:
            best_c4 = c4_ls
            best_edges = list(ls_edges)
    
    # Try random restarts
    for seed in range(50):
        if time.time() > deadline - 8:
            break
        random.seed(seed)
        target = [6]*27 + [5]*4
        random.shuffle(target)
        
        edges_rand = []
        edge_set = set()
        deg = [0]*N
        
        candidates = [(i,j) for i in range(N) for j in range(i+1,N)]
        random.shuffle(candidates)
        
        for u,v in candidates:
            if len(edges_rand) >= M:
                break
            if deg[u] < target[u] and deg[v] < target[v]:
                edges_rand.append([u,v])
                edge_set.add((u,v))
                deg[u] += 1
                deg[v] += 1
        
        if len(edges_rand) < M:
            for u,v in candidates:
                if len(edges_rand) >= M:
                    break
                if (u,v) not in edge_set:
                    edges_rand.append([u,v])
                    edge_set.add((u,v))
                    deg[u] += 1
                    deg[v] += 1
        
        if len(edges_rand) != M:
            continue
        
        adj_r = make_adj(edges_rand)
        c4_r = count_c4(adj_r)
        if c4_r < best_c4:
            best_c4 = c4_r
            best_edges = list(edges_rand)
        
        if time.time() < deadline - 5:
            ls_r = local_search(list(edges_rand), 2)
            adj_ls_r = make_adj(ls_r)
            c4_ls_r = count_c4(adj_ls_r)
            if c4_ls_r < best_c4:
                best_c4 = c4_ls_r
                best_edges = list(ls_r)
    
    if best_edges is None or len(best_edges) != M:
        pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
        best_edges = pairs[:M]
    
    return best_edges
# EVOLVE-BLOCK-END
