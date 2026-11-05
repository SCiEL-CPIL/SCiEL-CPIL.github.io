"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices.
    
    Strategy: Use a circulant graph C(40, S) where edges are defined by a fixed
    connection set S. The Hamiltonian cycle uses step 1 (and 39). We search for
    a minimal set of additional steps that, combined with step 1, allow us to
    form cycles of every length 3..40.
    
    Key insight: A cycle of length k in a circulant graph can be formed by a
    sequence of steps that sum to 0 mod N, visiting k distinct vertices.
    
    For a simple cycle of length k using only steps from S ∪ {1, N-1}:
    - If we use step d (where d < N/2), we can form a cycle by going d, then
      (N-d) steps of 1... no, that's not right.
    
    Actually, let me think differently. In a circulant graph C(N, S), a cycle
    of length k exists if there's a sequence of k steps s_1, ..., s_k from S
    (with signs ±) such that sum s_i ≡ 0 (mod N) and all partial sums are distinct.
    
    For our purposes, the simplest cycles are:
    - Triangle: needs steps that sum to 0, e.g., 1 + 1 + (N-2) ≡ 0 mod N
      So we need step 2 in S (since N-2 is the reverse of 2).
    - Square: 1 + 1 + 1 + (N-3) ≡ 0, need step 3.
    - In general, a k-cycle can be formed by (k-1) steps of 1 and one step of (N-(k-1)).
      This requires step (k-1) in S (or equivalently N-(k-1)).
    
    So if S = {2, 3, 4, ..., 20}, then:
    - Length k (3 ≤ k ≤ 21): use (k-1) steps of 1 and one step of (k-1).
      Cycle: 0, 1, 2, ..., k-2, k-1, 0. Edges: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0).
      This needs edge (k-1, 0), which is step k-1. ✓
    - Length k (22 ≤ k ≤ 39): use (N-k+1) steps of 1 and one step of (N-k+1).
      Cycle: 0, N-1, N-2, ..., k-1, 0. Wait, let me reconsider.
      
      For length k > 20: we need a cycle of length k.
      Use one step of (N-k+1) and (k-1) steps of 1? No, that gives length k but
      the step (N-k+1) is large.
      
      Alternative: Use step (N-k+1) going "backwards". 
      Cycle: 0, N-(N-k+1), N-(N-k+1)-1, ..., 0.
      That is: 0, k-1, k-2, ..., 0. Length is k.
      Edges: (0, k-1), (k-1, k-2), ..., (1, 0).
      This needs edge (0, k-1), which is step k-1. But k-1 > 20 for k > 21.
      
      Hmm, so we need steps up to 20 to cover lengths 3..21, and for lengths 22..39
      we need steps 21..38? That's too many.
      
    Wait, I made an error. Let me reconsider.
    
    In a circulant graph C(40, S), the edges are (i, i+s mod N) for s in S.
    If S contains d, it also implicitly contains N-d (since the graph is undirected).
    
    For a cycle of length k:
    - We can use a "star" pattern: from vertex 0, go to d, then d-1, d-2, ..., 0.
      That's k = d+1 vertices: 0, d, d-1, ..., 0. Wait, that's d+1 vertices but
      the edges are (0,d), (d, d-1), (d-1, d-2), ..., (1, 0).
      This requires edge (0,d) [step d] and edges (j, j-1) for j=1..d [step 1].
      So a cycle of length d+1 exists if d ∈ S.
      
    - Similarly, we can go 0, N-d, N-d+1, ..., N-1, 0. That's d+1 vertices.
      Edges: (0, N-d) [step d], (N-d, N-d+1), ..., (N-1, 0) [step 1].
      Same requirement: d ∈ S.
    
    So each step d ∈ S gives us cycles of length d+1.
    To cover lengths 3..39, we need d ∈ {2, 3, ..., 38}. That's 37 steps.
    Total edges: 40 (from step 1) + 37*40/2... no, each step d adds N/2 edges if d ≠ N/2,
    or N/2 edges if d = N/2. Actually, each step d adds N edges (i, i+d) for i=0..N-1,
    but since the graph is undirected, each edge is counted once, so N edges per step.
    Wait no: the edge set for step d is {(i, i+d mod N) : i = 0..N-1}. Since (i, i+d) = (i+d, i)
    when we consider undirected edges, and i+d mod N ≠ i for d ≠ 0, we have N distinct edges
    for each step d (as long as d ≠ N/2; if d = N/2, we have N/2 distinct edges).
    
    So total edges = N (for step 1) + sum over d in S\{1} of (N if d ≠ N/2 else N/2).
    For 37 steps, that's 40 + 37*40 = 1520 edges. Way too many.
    
    The circulant approach with many steps is too dense. Let me go back to the
    original approach but optimize it.
    
    Original approach: Hamiltonian cycle (40 edges) + 19 chords (from vertex 0 to vertices 2..20).
    Total: 59 edges. Score: 59-40 = 19.
    
    Can we do better? The key constraint is that we need cycles of every length 3..40.
    Each chord (0, d) gives us cycles of length d+1 and N-d+1.
    With chords for d=2..20, we cover:
    - d=2: lengths 3 and 39
    - d=3: lengths 4 and 38
    - ...
    - d=20: lengths 21 and 21
    
    So 19 chords cover all 37 required lengths (3..39). Total 59 edges.
    
    Can we use fewer chords by having chords that interact? For example, two chords
    might together enable a cycle that neither alone can.
    
    Consider chord (0, a) and chord (b, c). Can they together form a cycle of some
    length that isn't directly given by either chord alone?
    
    This is complex. Let me try a different approach: use a set of chords not all
    from vertex 0, and see if we can cover all cycle lengths with fewer than 19 chords.
    
    Actually, the theoretical minimum: each chord can cover at most 2 cycle lengths
    (the two cycles formed with the Hamiltonian cycle). We need to cover 37 lengths
    (3..39). So we need at least ceil(37/2) = 19 chords. Thus 59 edges is optimal
    for this construction family.
    
    But wait - can a single chord cover more than 2 cycle lengths when combined
    with other chords? For example, if we have chords (0, a) and (a, b), we might
    form a cycle 0-a-b-...-0 using multiple chords.
    
    This could potentially reduce the number of chords needed. Let me try to find
    a construction with fewer than 19 chords.
    
    Alternative: Use a different base graph. Instead of a Hamiltonian cycle, use
    a different 40-edge graph that might allow more efficient cycle coverage.
    
    For now, let's stick with the 59-edge solution but verify it's correct.
    """
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Chords from vertex 0 to vertices 2..20
    chords = []
    for d in range(2, 21):
        chords.append((0, d))
    
    all_edges = list(base_edges) + chords
    
    # Construct explicit cycles for each length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        elif k <= 21:
            # Short cycle from chord (0, k-1)
            d = k - 1
            cycles.append(list(range(0, d + 1)))  # [0, 1, 2, ..., d]
        else:
            # Long cycle from chord (0, N-k+1)
            d = N - k + 1
            # Vertices: 0, N-1, N-2, ..., d
            cyc = [0] + list(range(N - 1, d - 1, -1))
            cycles.append(cyc)
    
    # Normalize edges and verify
    edge_set = set()
    for e in all_edges:
        edge_set.add((min(e), max(e)))
    
    valid = True
    for cyc in cycles:
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        return {'edges': [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)],
                'cycles': [list(range(k)) for k in range(3, N + 1)]}
    
    return {'edges': list(edge_set), 'cycles': cycles}
# EVOLVE-BLOCK-END
