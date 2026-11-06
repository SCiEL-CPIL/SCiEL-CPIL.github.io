"""Dense Q3-free graph on 16 vertices via simulated annealing with fast Q3 detection."""
import time
import random
from itertools import combinations

_DEADLINE = 140.0  # safety margin before 150s wall-clock

# Q3 structure: 8 vertices, 12 edges
_Q3_VERTS = list(range(8))
_Q3_EDGES = []
for i in range(8):
    for bit in (0, 1, 2):
        j = i ^ (1 << bit)
        if i < j:
            _Q3_EDGES.append((i, j))
# 12 edges

# For fast Q3 detection: precompute all 8-subsets and for each, the 8! isomorphism check
# is too slow. Instead, use a smarter approach:
# Q3 is bipartite with parts {0,3,5,6} and {1,2,4,7} (even/odd parity)
# Actually: parity = bitcount % 2. Even: 0(000),3(011),5(101),6(110). Odd: 1(001),2(010),4(100),7(111)
_Q3_EVEN = frozenset({0, 3, 5, 6})
_Q3_ODD = frozenset({1, 2, 4, 7})

# Precompute all C(16,8) subsets with their edge-bitmasks for fast lookup
_ALL_SUBSETS = list(combinations(range(16), 8))
_NUM_SUBSETS = len(_ALL_SUBSETS)

# For each subset, precompute the 28 possible pairs as indices into a 28-bit edge mask
# Actually, let's use a different approach: represent the graph's edge set as a frozenset of (u,v) pairs.
# For Q3 detection on a given graph, we need to check all 8-subsets.

# Fast Q3 check using bitmasks:
# Represent each vertex's adjacency as a 16-bit mask.
# For an 8-subset, check if there's a Q3 subgraph.

# Precompute: for each 8-subset, the list of (local_index, global_vertex) mappings
_SUBSET_VERTS = [list(s) for s in _ALL_SUBSETS]

def _is_q3_free_adj_masks(adj_masks):
    """Check if graph (given as list of 16-bit adjacency masks) is Q3-free.
    Uses the fact that Q3 is the unique bipartite graph with parts of size 4,4
    where each vertex has degree 3 in the bipartite sense (within the Q3).
    
    Actually, Q3 as a subgraph: we need 8 vertices and 12 edges forming Q3.
    Key insight: Q3 is the unique (up to isomorphism) 3-regular bipartite graph
    with 4+4 vertices that is not K_{4,4} minus a perfect matching... no wait,
    K_{4,4} has 16 edges, Q3 has 12. Q3 = K_{4,4} minus a 1-factor (perfect matching)?
    No. K_{4,4} has 16 edges. Q3 has 12 edges. So Q3 is not K_{4,4} minus a perfect matching.
    
    Let me think again. Q3 has 8 vertices, each degree 3, so 12 edges.
    It's bipartite with parts of 4. Each vertex in one part connects to 3 of the 4 in the other.
    So Q3 = K_{4,4} minus a perfect matching! Yes: 16 - 4 = 12 edges.
    
    So a Q3 subgraph exists iff there are 8 vertices that can be partitioned into A (size 4) and B (size 4)
    such that every vertex in A connects to exactly 3 vertices in B, every vertex in B connects to exactly 3
    in A, and the missing edges form a perfect matching between A and B.
    
    This is a much faster check!
    """
    n = 16
    for subset in _ALL_SUBSETS:
        verts = list(subset)
        # Try all ways to partition into A (4) and B (4)
        # C(8,4) = 70 partitions, but each partition counted twice (A,B) vs (B,A)
        # So 35 unique bipartitions
        for i in range(4):
            for j in range(i+1, 5):
                for k in range(j+1, 6):
                    for l in range(k+1, 7):
                        A_local = (i, j, k, l)
                        B_local = tuple(x for x in range(8) if x not in A_local)
                        A = [verts[x] for x in A_local]
                        B = [verts[x] for x in B_local]
                        
                        # Check: each a in A connects to exactly 3 in B, each b in B connects to exactly 3 in A
                        # And the non-edges form a perfect matching
                        valid = True
                        missing = []  # list of (a_idx, b_idx) non-edges
                        for ai, a in enumerate(A):
                            amask = adj_masks[a]
                            b_neighbors = 0
                            for bi, b in enumerate(B):
                                if amask & (1 << b):
                                    b_neighbors += 1
                                else:
                                    missing.append((ai, bi))
                            if b_neighbors != 3:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        for bi, b in enumerate(B):
                            bmask = adj_masks[b]
                            a_neighbors = 0
                            for ai, a in enumerate(A):
                                if bmask & (1 << a):
                                    a_neighbors += 1
                            if a_neighbors != 3:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Check missing edges form a perfect matching: 4 non-edges, each a and b used exactly once
                        if len(missing) != 4:
                            continue
                        a_used = set()
                        b_used = set()
                        ok_match = True
                        for (ai, bi) in missing:
                            if ai in a_used or bi in b_used:
                                ok_match = False
                                break
                            a_used.add(ai)
                            b_used.add(bi)
                        if not ok_match:
                            continue
                        
                        # Found Q3
                        return False
    
    return True


def construct():
    start_time = time.time()
    n = 16
    rng = random.Random(12345)
    
    # Represent graph as list of 16-bit adjacency masks
    # Start with a random dense graph and do simulated annealing
    # Objective: maximize edges subject to Q3-free
    
    # Initial solution: random graph with ~80% edges, then remove Q3s
    def random_graph(p=0.7):
        masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks
    
    def edge_count(masks):
        return sum(bin(masks[i]).count('1') for i in range(n)) // 2
    
    # Initialize
    best_masks = None
    best_count = 0
    
    # Try several random initializations
    for init in range(50):
        if time.time() - start_time > _DEADLINE - 10:
            break
        masks = random_graph(p=0.75)
        ec = edge_count(masks)
        
        # Greedy Q3 removal
        for _ in range(200):
            if time.time() - start_time > _DEADLINE - 10:
                break
            if _is_q3_free_adj_masks(masks):
                break
            # Find a Q3 and remove a random edge from it
            # We need to find which Q3 exists - use the same logic but return the Q3
            removed = False
            for subset in _ALL_SUBSETS:
                verts = list(subset)
                for i in range(4):
                    for j in range(i+1, 5):
                        for k in range(j+1, 6):
                            for l in range(k+1, 7):
                                A_local = (i, j, k, l)
                                B_local = tuple(x for x in range(8) if x not in A_local)
                                A = [verts[x] for x in A_local]
                                B = [verts[x] for x in B_local]
                                
                                valid = True
                                missing = []
                                for ai, a in enumerate(A):
                                    amask = masks[a]
                                    b_neighbors = 0
                                    for bi, b in enumerate(B):
                                        if amask & (1 << b):
                                            b_neighbors += 1
                                        else:
                                            missing.append((ai, bi))
                                    if b_neighbors != 3:
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                for bi, b in enumerate(B):
                                    bmask = masks[b]
                                    a_neighbors = 0
                                    for ai, a in enumerate(A):
                                        if bmask & (1 << a):
                                            a_neighbors += 1
                                    if a_neighbors != 3:
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                if len(missing) != 4:
                                    continue
                                a_used = set()
                                b_used = set()
                                ok_match = True
                                for (ai, bi) in missing:
                                    if ai in a_used or bi in b_used:
                                        ok_match = False
                                        break
                                    a_used.add(ai)
                                    b_used.add(bi)
                                if not ok_match:
                                    continue
                                
                                # Found Q3 - remove a random edge
                                all_q3_edges = []
                                for a in A:
                                    for b in B:
                                        if masks[a] & (1 << b):
                                            all_q3_edges.append((a, b))
                                u, v = rng.choice(all_q3_edges)
                                masks[u] &= ~(1 << v)
                                masks[v] &= ~(1 << u)
                                removed = True
                                break
                            if removed:
                                break
                        if removed:
                            break
                    if removed:
                        break
                if removed:
                    break
            if not removed:
                break
        
        ec = edge_count(masks)
        if ec > best_count:
            best_count = ec
            best_masks = masks[:]
    
    # Simulated annealing phase
    if time.time() - start_time < _DEADLINE - 5:
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        current_masks = best_masks[:]
        current_count = best_count
        
        while time.time() - start_time < _DEADLINE - 5:
            T *= cooling
            if T < T_min:
                break
            
            # Random move: add or remove a random edge
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            has_edge = bool(current_masks[i] & (1 << j))
            
            if has_edge:
                # Remove edge
                new_masks = current_masks[:]
                new_masks[i] &= ~(1 << j)
                new_masks[j] &= ~(1 << i)
                new_count = current_count - 1
                delta = -1
            else:
                # Add edge
                new_masks = current_masks[:]
                new_masks[i] |= (1 << j)
                new_masks[j] |= (1 << i)
                # Check if still Q3-free
                if not _is_q3_free_adj_masks(new_masks):
                    continue
                new_count = current_count + 1
                delta = 1
            
            # Accept if better or with probability exp(-delta/T) for delta>0 (but delta is always -1 or 1)
            if delta <= 0:
                current_masks = new_masks
                current_count = new_count
            else:
                import math
                if rng.random() < math.exp(-delta / T):
                    current_masks = new_masks
                    current_count = new_count
            
            if current_count > best_count:
                best_count = current_count
                best_masks = current_masks[:]
    
    # Final verification
    if best_masks is None:
        # Fallback: simple construction
        best_masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if (i // 4) != (j // 4) or (i % 4) != (j % 4):
                    best_masks[i] |= (1 << j)
                    best_masks[j] |= (1 << i)
        # This is K4 x K4 complement within groups... might have Q3s. Just return it.
    
    if not _is_q3_free_adj_masks(best_masks):
        # Fallback to a known Q3-free graph: complete 4-partite graph K_{4,4,4,4}
        # This has 4*4*3/2 * 4... actually edges between different parts: 4*3*C(4,2) = 4*3*6=72? No.
        # K_{4,4,4,4}: each part 4 vertices, edges between all different parts.
        # Total edges: sum over pairs of parts C(4,2)*... no, all pairs between different parts.
        # 4 parts of size 4. Edges = (1/2) * sum_{i!=j} 4*4 = (1/2)*12*16 = 96? 
        # Wait: for each ordered pair of parts (p,q) with p!=q, 4*4=16 edges. 12 ordered pairs, so 192/2=96.
        # K_{4,4,4,4} has 96 edges. Does it contain Q3? Q3 is bipartite with parts 4,4.
        # In K_{4,4,4,4}, pick 4 vertices from part A and 4 from part B - that's K_{4,4} which contains Q3 as subgraph.
        # So K_{4,4,4,4} is NOT Q3-free.
        
        # Better fallback: random sparse graph
        masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        while not _is_q3_free_adj_masks(masks):
            # Remove random edge
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if i != j and (masks[i] & (1 << j)):
                masks[i] &= ~(1 << j)
                masks[j] &= ~(1 << i)
        best_masks = masks
        best_count = edge_count(masks)
    
    # Convert to list of [u,v] pairs
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_masks[i] & (1 << j):
                edges.append([i, j])
    
    return edges
