import time
import random
import itertools

N = 16
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute all 4-subsets as bitmasks
_SUBSETS4_MASKS = None
# For each 4-subset mask, store the list of vertex indices
_SUBSETS4_VECS = None
# Precompute disjointness: for each pair of 4-masks, whether they're disjoint
_DISJOINT = None

def _init():
    global _SUBSETS4_MASKS, _SUBSETS4_VECS, _DISJOINT
    if _SUBSETS4_MASKS is None:
        _SUBSETS4_MASKS = []
        _SUBSETS4_VECS = []
        for comb in itertools.combinations(range(N), 4):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            _SUBSETS4_MASKS.append(mask)
            _SUBSETS4_VECS.append(comb)
        # Precompute disjointness table
        n4 = len(_SUBSETS4_MASKS)
        _DISJOINT = [[False] * n4 for _ in range(n4)]
        for i in range(n4):
            for j in range(i+1, n4):
                if _SUBSETS4_MASKS[i] & _SUBSETS4_MASKS[j] == 0:
                    _DISJOINT[i][j] = True
                    _DISJOINT[j][i] = True

def _has_q3_in_bipartite(adj_masks, rows_mask, cols_mask):
    """Check if the bipartite subgraph induced by rows_mask and cols_mask contains Q3.
       adj_masks[v] is a bitmask of neighbors of v.
       Q3 is a 4x4 bipartite graph with each vertex degree 3 (complement of perfect matching).
       So we need: 4 rows, 4 cols, where each row has >=3 edges to cols, each col has >=3 edges to rows,
       AND the bipartite graph contains a 4-regular... wait no.
       
       Actually Q3 as a bipartite graph: Q3 has 8 vertices (4 on each side) and 12 edges.
       Each vertex has degree 3. The bipartite adjacency is: each row connects to 3 of 4 cols
       (missing exactly one), and the missing edges form a perfect matching.
       
       So the induced bipartite graph on 4+4 vertices contains Q3 iff:
       - Each row has degree >= 3 within cols (i.e., at most 1 missing edge per row)
       - Each col has degree >= 3 within rows (i.e., at most 1 missing edge per col)
       - The set of missing edges forms a perfect matching (or is empty, but that would be K4,4 which contains Q3)
       
       Wait, if all 16 edges are present, that's K4,4 which does contain Q3 as a subgraph.
       If exactly 4 edges are missing and they form a perfect matching, that's exactly Q3.
       If fewer than 4 edges are missing (but each vertex still has degree >= 3), it still contains Q3.
       
       Actually, the condition for containing Q3 as a subgraph:
       We need to find 4 rows and 4 cols such that the bipartite graph between them has
       each vertex with degree at least 3, AND there exists a subset of edges forming Q3.
       
       Since Q3 has 12 edges and each vertex degree 3, we need the induced bipartite graph
       to have at least 12 edges total, with each vertex having degree >= 3.
       
       More precisely: the induced bipartite graph on rows x cols contains Q3 iff
       there exists a perfect matching M of missing edges such that after removing M,
       each vertex has degree >= 3. But since max degree is 4, each vertex has degree exactly 3 or 4.
       
       Simpler check: count edges. If total edges >= 12 and min degree >= 3, then it contains Q3.
       Actually no - we need the specific structure. Let me think again.
       
       Q3 bipartite: rows {0,1,2,3}, cols {0,1,2,3}. Edges: all except a perfect matching.
       So 12 edges, each vertex degree 3.
       
       A bipartite graph on 4+4 vertices contains Q3 as a subgraph iff:
       - It has at least 12 edges, AND
       - Each vertex has degree at least 3.
       
       Why? If each vertex has degree >= 3, then at most 1 edge is missing per vertex.
       Total missing edges <= 8 (4 rows * 1 + 4 cols * 1 = 8, but each missing edge is counted twice).
       Actually, missing edges: let m = number of missing edges. Each row misses at most 1, so sum of row-missing = m <= 4.
       Each col misses at most 1, so sum of col-missing = m <= 4.
       So m <= 4. And since each missing edge contributes to one row-missing and one col-missing,
       m = (sum row-missing) = (sum col-missing) <= 4.
       
       If m = 0: K4,4, contains Q3.
       If m = 1: one edge missing, the row and col of that edge have degree 3, others have degree 4.
         Does this contain Q3? We need 12 edges with specific structure. K4,4 minus one edge has 15 edges.
         We can pick any 12-edge subgraph that is Q3. Yes, it contains Q3.
       If m = 2: two edges missing. If they share a row or col, one vertex has degree 2 -> no Q3.
         If they don't share (i.e., form a matching of size 2), then two rows have degree 3, two have degree 4,
         same for cols. Total edges = 14. Does it contain Q3? We need to find 12 edges forming Q3.
         The two missing edges form a partial matching. We can extend to a full perfect matching by adding 2 more
         "virtual" missing edges... hmm, this is getting complex.
       
       Let me just use the direct check: iterate over all 24 perfect matchings and check if
       removing that matching's edges from the induced subgraph leaves all vertices with degree >= 3.
       Equivalently: for each perfect matching M, check if for all (r,c) in M, (r,c) is NOT in the graph
       OR... no wait.
       
       Q3 = K4,4 minus a perfect matching. So the induced graph G contains Q3 iff there exists
       a perfect matching M such that all edges of M are NOT in G, and all other 12 edges ARE in G.
       No wait, that's too strict. G contains Q3 as a subgraph means there's a subset of edges of G
       that forms Q3. So we need: there exists a perfect matching M such that for all (r,c) NOT in M,
       the edge (r,c) IS in G. The edges in M may or may not be in G.
       
       So the check is: does there exist a perfect matching M of the 4x4 grid such that
       all 12 non-M edges are present in G?
       
       Equivalently: the complement of G (within the 4x4) does NOT contain a perfect matching.
       Wait no. We need: there exists M (perfect matching) such that M ⊆ complement(G).
       That is, the complement of G contains a perfect matching.
       
       Yes! That's the clean check: the missing-edge graph on 4x4 contains a perfect matching.
    """
    rows = [v for v in range(N) if (rows_mask >> v) & 1]
    cols = [v for v in range(N) if (cols_mask >> v) & 1]
    
    # Build missing-edge bitmask for each row
    # missing[r_idx][c_idx] = True if edge (rows[r_idx], cols[c_idx]) is NOT in adj
    # We need: does the missing-edge matrix have a perfect matching?
    
    # For each row, compute which cols are missing
    # missing_mask[r] = bitmask of cols that are missing for row r
    missing_masks = []
    for r in rows:
        mm = 0
        for j, c in enumerate(cols):
            if not (adj_masks[r] >> c) & 1:
                mm |= (1 << j)
        missing_masks.append(mm)
    
    # Check if missing-edge bipartite graph has a perfect matching
    # Use the 24 precomputed matchings
    for perm in _PERM_MATCHINGS:
        ok = True
        for r_idx, c_idx in perm:
            if not (missing_masks[r_idx] >> c_idx) & 1:
                ok = False
                break
        if ok:
            return True
    return False

_PERM_MATCHINGS = [tuple(zip(range(4), p)) for p in itertools.permutations(range(4))]

def _creates_q3_fast(u, v, adj_masks, side_a_mask):
    """Check if adding edge (u,v) creates a Q3.
       u is in side_a (rows), v is in side_b (cols).
       adj_masks is a list of 16 bitmasks."""
    # We need to check all 4-subsets of rows containing u and 4-subsets of cols containing v
    # that are disjoint (they always are since u in side_a, v in side_b, and sides are disjoint)
    
    # Get row candidates: 4-subsets of side_a containing u
    # Get col candidates: 4-subsets of side_b containing v
    
    # Precompute: for each vertex, which 4-subsets (by index in _SUBSETS4_MASKS) contain it
    # And which are within side_a / side_b
    
    # For efficiency, iterate over 4-subset masks
    for i, rows_mask in enumerate(_SUBSETS4_MASKS):
        if not (rows_mask >> u) & 1:
            continue
        # rows must be subset of side_a
        if rows_mask & ~side_a_mask:
            continue
        for j, cols_mask in enumerate(_SUBSETS4_MASKS):
            if not (cols_mask >> v) & 1:
                continue
            # cols must be subset of side_b (i.e., disjoint from side_a)
            if cols_mask & side_a_mask:
                continue
            if not _DISJOINT[i][j]:
                continue
            # Check Q3
            if _has_q3_in_bipartite(adj_masks, rows_mask, cols_mask):
                return True
    return False

def _is_q3_free_fast(adj_masks, side_a_mask):
    """Full check: any bad pair?"""
    for i, rows_mask in enumerate(_SUBSETS4_MASKS):
        if rows_mask & ~side_a_mask:
            continue
        for j, cols_mask in enumerate(_SUBSETS4_MASKS):
            if cols_mask & side_a_mask:
                continue
            if not _DISJOINT[i][j]:
                continue
            if _has_q3_in_bipartite(adj_masks, rows_mask, cols_mask):
                return False
    return True

def _try_build(seed):
    rng = random.Random(seed)
    # random bipartition: choose 8 vertices as side A
    verts = list(range(N))
    rng.shuffle(verts)
    side_a = frozenset(verts[:8])
    side_b = frozenset(verts[8:])
    side_a_mask = 0
    for v in side_a:
        side_a_mask |= (1 << v)
    
    # all candidate edges between side_a and side_b
    edges = [(a, b) for a in side_a for b in side_b]
    rng.shuffle(edges)
    
    adj_masks = [0] * N
    for (u, v) in edges:
        if time.time() > DEADLINE:
            break
        # Tentatively add edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        if _creates_q3_fast(u, v, adj_masks, side_a_mask):
            # Remove edge
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            continue
    
    # Verify
    if not _is_q3_free_fast(adj_masks, side_a_mask):
        return None
    
    # Build edge list
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj_masks[u] >> v) & 1:
                result.append([u, v])
    return result

def construct():
    _init()
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]  # K3,13, 39 edges
    best_count = 39
    seed = 0
    while time.time() < DEADLINE:
        result = _try_build(seed)
        if result is not None:
            cnt = len(result)
            if cnt > best_count:
                best_count = cnt
                best_edges = result
        seed += 1
    return best_edges
