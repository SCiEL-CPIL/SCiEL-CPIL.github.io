import time
import random
import itertools

N = 16
DEADLINE = 145.0

_PERM_MATCHINGS = [tuple(zip(range(4), p)) for p in itertools.permutations(range(4))]

def _has_q3_containing_edge(u, v, adj_masks):
    """Check if adding edge (u,v) creates a Q3.
    Q3 is bipartite with parts of size 4. Edge (u,v) must connect the two parts.
    Case 1: u is in part A, v is in part B.
    We need 3 more vertices in A (call them a1,a2,a3) and 3 more in B (b1,b2,b3)
    such that the 8-vertex induced subgraph contains Q3.
    
    Q3 structure: each vertex has degree 3. In the bipartite representation,
    each vertex in A connects to exactly 3 of the 4 vertices in B (missing one),
    and the missing edges form a perfect matching.
    
    So for the 4x4 bipartite graph to contain Q3:
    - Each of the 4 A-vertices has degree >= 3 in the 4 B-vertices
    - Each of the 4 B-vertices has degree >= 3 in the 4 A-vertices
    - The missing edges contain a perfect matching
    
    Equivalently: the complement (missing edges) of the 4x4 adjacency matrix
    contains a perfect matching.
    
    For speed: enumerate C(7,3)=35 choices for A-side companions and C(7,3)=35 for B-side.
    That's 35*35 = 1225 checks per edge, each very fast (4x4 matrix check).
    """
    # u is in part A, v is in part B
    # Find candidates for A: vertices other than u that are NOT v
    # Find candidates for B: vertices other than v that are NOT u
    
    a_candidates = [x for x in range(N) if x != u and x != v]
    b_candidates = [x for x in range(N) if x != u and x != v]
    
    # For each choice of 3 from a_candidates (to be in A with u) and 3 from b_candidates (to be in B with v)
    # Check if the 4x4 bipartite subgraph contains Q3
    
    # Pre-filter: for a vertex a in A to have degree >= 3 in B={v,b1,b2,b3},
    # it must be connected to at least 3 of {v,b1,b2,b3}.
    # Similarly for b in B to have degree >= 3 in A={u,a1,a2,a3}.
    
    for a_combo in itertools.combinations(a_candidates, 3):
        A = (u,) + a_combo
        # Quick check: u must have degree >= 3 in B. We'll check that after picking B.
        # But we can pre-filter: for each a in A, count how many of b_candidates it connects to
        # Actually let's just iterate
        for b_combo in itertools.combinations(b_candidates, 3):
            B = (v,) + b_combo
            
            # Build 4x4 adjacency: A[i] connected to B[j]?
            # Check: each A vertex has degree >= 3 in B, each B vertex has degree >= 3 in A
            # And the missing edges contain a perfect matching
            
            # Build missing edge matrix
            # missing[i][j] = 1 if A[i] NOT connected to B[j]
            missing = [[0]*4 for _ in range(4)]
            valid = True
            for i in range(4):
                deg = 0
                for j in range(4):
                    if (adj_masks[A[i]] >> B[j]) & 1:
                        deg += 1
                    else:
                        missing[i][j] = 1
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            for j in range(4):
                deg = 0
                for i in range(4):
                    if (adj_masks[A[i]] >> B[j]) & 1:
                        deg += 1
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if missing matrix has a perfect matching
            # Use the 24 precomputed permutations
            found = False
            for perm in _PERM_MATCHINGS:
                ok = True
                for i, j in perm:
                    if missing[i][j] == 0:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                return True
    
    return False


def _is_q3_free(adj_masks):
    """Check if the graph has no Q3 subgraph."""
    # Q3 is bipartite with parts of size 4.
    # Enumerate all ways to pick 4 vertices as part A and 4 as part B (disjoint).
    # For each such pair, check if the bipartite subgraph contains Q3.
    # C(16,4) = 1820 choices for A, then C(12,4) = 495 for B. Total ~900K checks.
    # Each check is fast (4x4 matrix). This might be too slow for full verification.
    # Alternative: check only during construction.
    
    # For the final verification, we need to be thorough.
    # Let's use a smarter approach: for each 4-subset A, for each 4-subset B disjoint from A,
    # check the bipartite condition.
    
    all_4subsets = list(itertools.combinations(range(N), 4))
    n4 = len(all_4subsets)
    
    for i in range(n4):
        A = all_4subsets[i]
        A_set = set(A)
        A_mask = 0
        for a in A:
            A_mask |= (1 << a)
        for j in range(i+1, n4):
            B = all_4subsets[j]
            # Check disjoint
            if any(b in A_set for b in B):
                continue
            
            # Build 4x4 adjacency
            valid = True
            missing = [[0]*4 for _ in range(4)]
            for ii in range(4):
                deg = 0
                for jj in range(4):
                    if (adj_masks[A[ii]] >> B[jj]) & 1:
                        deg += 1
                    else:
                        missing[ii][jj] = 1
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            for jj in range(4):
                deg = 0
                for ii in range(4):
                    if (adj_masks[A[ii]] >> B[jj]) & 1:
                        deg += 1
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check perfect matching in missing
            found = False
            for perm in _PERM_MATCHINGS:
                ok = True
                for ii, jj in perm:
                    if missing[ii][jj] == 0:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                return False
    
    return True


def _try_build(seed):
    rng = random.Random(seed)
    adj_masks = [0] * N
    
    # Start with empty graph, add edges greedily
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > DEADLINE - 5:
            break
        # Try adding edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        if _has_q3_containing_edge(u, v, adj_masks):
            # Remove
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
    
    # Verify
    if not _is_q3_free(adj_masks):
        return None
    
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj_masks[u] >> v) & 1:
                result.append([u, v])
    return result


def _try_build_bipartite(seed):
    """Try building a bipartite graph with parts of size 8."""
    rng = random.Random(seed)
    verts = list(range(N))
    rng.shuffle(verts)
    side_a = verts[:8]
    side_b = verts[8:]
    side_a_set = set(side_a)
    side_b_set = set(side_b)
    
    adj_masks = [0] * N
    
    # All bipartite edges
    edges = [(a, b) for a in side_a for b in side_b]
    rng.shuffle(edges)
    
    for (u, v) in edges:
        if time.time() > DEADLINE - 5:
            break
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        if _has_q3_containing_edge(u, v, adj_masks):
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
    
    # Verify (only need to check bipartite Q3 since graph is bipartite)
    # For a bipartite graph, Q3 must be in the bipartite structure
    # Check all 4-subsets of side_a and 4-subsets of side_b
    all_4a = list(itertools.combinations(side_a, 4))
    all_4b = list(itertools.combinations(side_b, 4))
    
    for A in all_4a:
        for B in all_4b:
            valid = True
            missing = [[0]*4 for _ in range(4)]
            for ii in range(4):
                deg = 0
                for jj in range(4):
                    if (adj_masks[A[ii]] >> B[jj]) & 1:
                        deg += 1
                    else:
                        missing[ii][jj] = 1
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            for jj in range(4):
                deg = 0
                for ii in range(4):
                    if (adj_masks[A[ii]] >> B[jj]) & 1:
                        deg += 1
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            for perm in _PERM_MATCHINGS:
                ok = True
                for ii, jj in perm:
                    if missing[ii][jj] == 0:
                        ok = False
                        break
                if ok:
                    return None  # Not Q3-free
    
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj_masks[u] >> v) & 1:
                result.append([u, v])
    return result


def construct():
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]  # K3,13, 39 edges
    best_count = 39
    seed = 0
    
    while time.time() < DEADLINE - 10:
        # Alternate between bipartite and general approaches
        if seed % 2 == 0:
            result = _try_build_bipartite(seed)
        else:
            result = _try_build(seed)
        
        if result is not None:
            cnt = len(result)
            if cnt > best_count:
                best_count = cnt
                best_edges = result
        seed += 1
    
    return best_edges
