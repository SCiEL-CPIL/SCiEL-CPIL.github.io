"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _q3_free_check(adj_mask):
    """adj_mask: list of 16 ints, adj_mask[i] is bitmask of neighbors of i.
    Returns True if graph is Q3-free."""
    n = 16
    # Q3: 8 vertices, 12 edges, 3-regular
    # Check all 8-subsets
    for S in itertools.combinations(range(n), 8):
        S_set = set(S)
        # Quick check: count edges
        edge_count = 0
        for i in range(8):
            m = adj_mask[S[i]] & 0xFFFF
            # Count bits in m that are in S
            for j in range(i+1, 8):
                if m & (1 << S[j]):
                    edge_count += 1
        if edge_count != 12:
            continue
        # Check 3-regular
        ok = True
        for i in range(8):
            deg = 0
            m = adj_mask[S[i]]
            for j in range(8):
                if i != j and (m & (1 << S[j])):
                    deg += 1
            if deg != 3:
                ok = False
                break
        if not ok:
            continue
        # Check isomorphism to Q3
        # Q3 bipartition: even parity {0,3,5,6}, odd parity {1,2,4,7}
        # Try to find a 4-subset of S that forms one part
        # A vertex in one part has all 3 neighbors in the other part
        # Find a vertex and check if its 3 neighbors in S are all in the complement
        # Let's just try all 4-subsets as potential part A
        for subset_a in itertools.combinations(S, 4):
            sa = set(subset_a)
            sb = S_set - sa
            # Check: every vertex in sa has all 3 neighbors in sb, and vice versa
            valid = True
            for v in subset_a:
                neighbors_in_S = [u for u in S if u != v and (adj_mask[v] & (1 << u))]
                if len(neighbors_in_S) != 3:
                    valid = False
                    break
                if any(u in sa for u in neighbors_in_S):
                    valid = False
                    break
            if not valid:
                continue
            for v in sb:
                neighbors_in_S = [u for u in S if u != v and (adj_mask[v] & (1 << u))]
                if len(neighbors_in_S) != 3:
                    valid = False
                    break
                if any(u in sb for u in neighbors_in_S):
                    valid = False
                    break
            if not valid:
                continue
            # Now check the specific Q3 structure
            # In Q3, part A = {0,3,5,6}. 
            # 0 is adjacent to 1,2,4 (all in part B)
            # 3 is adjacent to 1,2,7
            # 5 is adjacent to 1,4,7
            # 6 is adjacent to 2,4,7
            # So within part A, the adjacency pattern to part B is specific.
            # Let's check: for each pair in part A, how many common neighbors in part B?
            # In Q3: 
            # 0,3: common neighbors {1,2} -> 2
            # 0,5: common neighbors {1,4} -> 2
            # 0,6: common neighbors {2,4} -> 2
            # 3,5: common neighbors {1,7} -> 2
            # 3,6: common neighbors {2,7} -> 2
            # 5,6: common neighbors {4,7} -> 2
            # So every pair in part A has exactly 2 common neighbors in part B.
            # Similarly for part B.
            # This is actually a property of the Clebsch graph complement...
            # Let's verify this is sufficient. If every pair in A has 2 common neighbors in B
            # and every pair in B has 2 common neighbors in A, and it's 3-regular bipartite,
            # is it Q3?
            # The bipartite graph with 4+4 vertices, 3-regular, with every pair sharing 2 neighbors
            # on the other side. This is the incidence structure of Q3.
            # Actually this might characterize Q3 uniquely for 4x4.
            a_pairs_ok = True
            for i in range(4):
                for j in range(i+1, 4):
                    vi = subset_a[i]
                    vj = subset_a[j]
                    common = 0
                    for w in sb:
                        if (adj_mask[vi] & (1 << w)) and (adj_mask[vj] & (1 << w)):
                            common += 1
                    if common != 2:
                        a_pairs_ok = False
                        break
                if not a_pairs_ok:
                    break
            if not a_pairs_ok:
                continue
            b_pairs_ok = True
            for i in range(4):
                for j in range(i+1, 4):
                    vi = list(sb)[i]
                    vj = list(sb)[j]
                    common = 0
                    for w in subset_a:
                        if (adj_mask[vi] & (1 << w)) and (adj_mask[vj] & (1 << w)):
                            common += 1
                    if common != 2:
                        b_pairs_ok = False
                        break
                if not b_pairs_ok:
                    break
            if b_pairs_ok:
                return False
    return True

def construct():
    deadline = time.time() + 140
    
    # Seed: K3,13
    seed_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_edges = list(seed_edges)
    
    def to_mask(edges):
        mask = [0] * 16
        for u, v in edges:
            mask[u] |= (1 << v)
            mask[v] |= (1 << u)
        return mask
    
    best_mask = to_mask(best_edges)
    
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    seed_set = set((min(u,v), max(u,v)) for u, v in seed_edges)
    missing = [(u, v) for (u, v) in all_pairs if (u, v) not in seed_set]
    random.shuffle(missing)
    
    current_edges = list(seed_edges)
    current_mask = to_mask(current_edges)
    
    # Greedy add
    for u, v in missing:
        if time.time() > deadline:
            break
        current_mask[u] |= (1 << v)
        current_mask[v] |= (1 << u)
        if _q3_free_check(current_mask):
            current_edges.append([u, v])
        else:
            current_mask[u] &= ~(1 << v)
            current_mask[v] &= ~(1 << u)
    
    if len(current_edges) > len(best_edges):
        best_edges = current_edges
        best_mask = current_mask
    
    # Multiple restarts with random order
    for restart in range(100):
        if time.time() > deadline:
            break
        missing = [(u, v) for (u, v) in all_pairs 
                   if (min(u,v), max(u,v)) not in set((min(e[0],e[1]), max(e[0],e[1])) for e in best_edges)]
        random.shuffle(missing)
        
        current_edges = list(best_edges)
        current_mask = to_mask(current_edges)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            current_mask[u] |= (1 << v)
            current_mask[v] |= (1 << u)
            if _q3_free_check(current_mask):
                current_edges.append([u, v])
            else:
                current_mask[u] &= ~(1 << v)
                current_mask[v] &= ~(1 << u)
        
        if len(current_edges) > len(best_edges):
            best_edges = current_edges
            best_mask = current_mask
    
    # Final check
    final_mask = to_mask(best_edges)
    if not _q3_free_check(final_mask):
        return seed_edges
    
    return best_edges
# EVOLVE-BLOCK-END
