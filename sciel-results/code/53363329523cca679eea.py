"""Construct C4-free graph with high minimum degree for R(C4,K1,39)."""
import time
import random
import itertools
from collections import defaultdict

# Deadline guard
_START = time.time()
TIME_LIMIT = 115.0  # seconds, leave margin before 120s

def _remaining():
    return TIME_LIMIT - (time.time() - _START)

def _has_c4_fast(adj_sets, u, v):
    """Check if adding edge (u,v) creates a C4.
    A C4 containing edge (u,v) exists iff there is a vertex w adjacent to both u and v
    (that's a triangle, not C4). Actually a C4 is u-x-v-y-u.
    Adding edge (u,v) creates a C4 iff there exist distinct x,y such that
    x adjacent to both u and v, and y adjacent to both u and v, and x!=y,
    and the edges (u,x),(x,v),(v,y),(y,u) form a cycle.
    Wait: if we add edge (u,v), a C4 through (u,v) would be u-v-x-y-u or u-v-y-x-u.
    That requires: v adjacent to x, x adjacent to y, y adjacent to u, and u adjacent to v (new).
    So we need: exists x in N(v) and y in N(u) with x != y, x != u, y != v,
    and x adjacent to y, and x != v, y != u.
    Actually simpler: a C4 containing the new edge (u,v) is a 4-cycle u-a-v-b-u.
    So we need a in N(v) (a != u), b in N(u) (b != v), a != b, and a adjacent to b.
    Equivalently: |N(u) ∩ N(v)| >= 2 would mean there are two common neighbors, but that's for triangles...
    
    Let me reconsider. A C4 is a cycle of length 4: vertices a-b-c-d-a.
    If we add edge (u,v), a C4 containing this edge has form u-v-x-y-u.
    So: v-x edge exists, x-y edge exists, y-u edge exists.
    Thus x ∈ N(v), y ∈ N(u), x ≠ y, x ≠ u, y ≠ v, and x-y is an edge.
    """
    # Common approach: count paths of length 2 between u and v.
    # If there are >= 2 distinct paths of length 2 from u to v (i.e., two common neighbors),
    # that creates a C4? No, two common neighbors a,b with u-a, a-v, u-b, b-v gives a 4-cycle u-a-v-b-u.
    # Yes! So adding (u,v) creates a C4 iff u and v have at least 2 common neighbors.
    common = adj_sets[u] & adj_sets[v]
    return len(common) >= 2

def _min_degree(adj_sets, n):
    return min(len(adj_sets[i]) for i in range(n))

def _is_valid(adj_sets, n, target_min_deg):
    """Check no C4 and min degree >= target_min_deg."""
    if _min_degree(adj_sets, n) < target_min_deg:
        return False
    # Check for C4
    for u in range(n):
        for v in adj_sets[u]:
            if v > u:
                common = adj_sets[u] & adj_sets[v]
                if len(common) >= 2:
                    return False
    return True

def _build_polarity_graph(q):
    """Build a polarity graph (C4-free) on q+1 or q vertices.
    For q ≡ 3 mod 4, the polarity graph of the projective plane has q+1 vertices.
    Vertices: 0..q-1 correspond to points, and we use the standard construction.
    
    Actually, the standard C4-free polarity graph:
    - Vertices: points of PG(2,q) plus one extra, or just use a simpler construction.
    
    Let's use the known construction: For prime power q ≡ 3 (mod 4),
    the graph with vertices {0, 1, ..., q} where i~j iff (i-j)^2 ≡ -1 (mod q) for i,j in {0,...,q-1},
    and vertex q is adjacent to all vertices i where i^2 ≡ -1 (mod q).
    
    This gives a C4-free graph on q+1 vertices with minimum degree ~ sqrt(q).
    """
    if q == 3:
        # Small case
        return None
    
    # For q ≡ 3 mod 4
    n = q + 1
    adj_sets = [set() for _ in range(n)]
    
    # Find elements x where x^2 ≡ -1 mod q
    # For q ≡ 3 mod 4, -1 is not a quadratic residue, so no such elements.
    # Wait, that means vertex q has no neighbors. That's not good.
    
    # Let me use a different standard construction.
    # The Erdős–Rényi polarity graph:
    # Vertices: 0 to q (q+1 vertices)
    # For i, j in {0, ..., q-1}: i~j iff (i-j)^2 ≡ -1 mod q
    # Vertex q is adjacent to i iff i^2 ≡ -1 mod q
    # But for q ≡ 3 mod 4, -1 is not a QR, so this doesn't work well.
    
    # Alternative: use q ≡ 1 mod 4 for the standard construction.
    # Let me just use a random greedy approach instead, which is more flexible.
    return None

def _greedy_build(n, target_min_deg, seed=42, max_iter=500000):
    """Greedy edge addition to build a C4-free graph with min degree >= target_min_deg."""
    random.seed(seed)
    adj_sets = [set() for _ in range(n)]
    
    # Try to add edges in a greedy fashion
    # Prioritize edges that help low-degree vertices
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        if _min_degree(adj_sets, n) >= target_min_deg:
            # Check if valid (no C4)
            if _is_valid(adj_sets, n, target_min_deg):
                return adj_sets
            break
        
        # Find the vertex with minimum degree
        min_deg = min(len(adj_sets[i]) for i in range(n))
        low_verts = [i for i in range(n) if len(adj_sets[i]) == min_deg]
        u = random.choice(low_verts)
        
        # Try to add an edge from u to some other vertex
        candidates = [v for v in range(n) if v != u and v not in adj_sets[u]]
        random.shuffle(candidates)
        
        for v in candidates:
            if _has_c4_fast(adj_sets, u, v):
                continue
            # Add edge
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            break
    
    if _min_degree(adj_sets, n) >= target_min_deg:
        if _is_valid(adj_sets, n, target_min_deg):
            return adj_sets
    return None

def _try_n(n, seed_base=0):
    """Try to construct a valid graph on n vertices."""
    target_min_deg = n - 39
    
    # Try multiple seeds
    for seed in range(seed_base, seed_base + 20):
        if _remaining() < 10.0:
            break
        result = _greedy_build(n, target_min_deg, seed=seed)
        if result is not None:
            # Build edge list
            edges = []
            for u in range(n):
                for v in result[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    return None

def _try_structured(n, target_min_deg):
    """Try a more structured approach: start from a dense base and prune."""
    # Start with a random graph and remove edges that create C4s
    # Or use a spectral approach
    
    # For now, try a different greedy strategy: add all possible edges, then remove C4-creating edges
    random.seed(12345)
    adj_sets = [set() for _ in range(n)]
    
    # Initialize with a sparse random graph
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.3:
                if not _has_c4_fast(adj_sets, u, v):
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
    
    # Now try to boost degrees
    for it in range(200000):
        if _remaining() < 10.0:
            break
        if _min_degree(adj_sets, n) >= target_min_deg:
            if _is_valid(adj_sets, n, target_min_deg):
                edges = []
                for u in range(n):
                    for v in adj_sets[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            break
        
        min_deg = min(len(adj_sets[i]) for i in range(n))
        low_verts = [i for i in range(n) if len(adj_sets[i]) == min_deg]
        u = random.choice(low_verts)
        
        candidates = [v for v in range(n) if v != u and v not in adj_sets[u]]
        random.shuffle(candidates)
        
        for v in candidates:
            if _has_c4_fast(adj_sets, u, v):
                continue
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            break
    
    if _min_degree(adj_sets, n) >= target_min_deg:
        if _is_valid(adj_sets, n, target_min_deg):
            edges = []
            for u in range(n):
                for v in adj_sets[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    return None

def construct():
    global _START
    _START = time.time()
    
    best = {'n': 39, 'edges': []}  # fallback
    
    # Try from n=47 down to 40
    for n in range(47, 39, -1):
        if _remaining() < 15.0:
            break
        
        target_min_deg = n - 39
        print(f"Trying n={n}, target_min_deg={target_min_deg}")
        
        # Try greedy approach
        result = _try_n(n, seed_base=0)
        if result is not None:
            best = result
            print(f"  Found valid graph for n={n}")
            break
        
        # Try structured approach
        if _remaining() > 20.0:
            result = _try_structured(n, target_min_deg)
            if result is not None:
                best = result
                print(f"  Found valid graph for n={n} (structured)")
                break
    
    return best

# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
