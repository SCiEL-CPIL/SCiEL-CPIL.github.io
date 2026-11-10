"""Dense Q3-free graph on 16 vertices via complement removal (hit all Q3s) + swap repair."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 112
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    M = len(all_edges)  # 120

    # Precompute all Q3 subgraphs (as frozensets of 12 edge indices)
    def find_all_q3s():
        q3s = []
        seen = set()
        # Q3 vertex set: 8 vertices forming a 3-cube
        # A 3-cube on vertices {a0,a1,a2,a3,a4,a5,a6,a7} has edges:
        # (a0,a1),(a0,a2),(a0,a4),
        # (a1,a3),(a1,a5),
        # (a2,a3),(a2,a6),
        # (a3,a7),
        # (a4,a5),(a4,a6),
        # (a5,a7),
        # (a6,a7)
        # That's 12 edges.
        # We need to check if these 12 edges form exactly a Q3 (no extra edges among the 8 vertices beyond these 12,
        # and the graph on these 8 vertices is isomorphic to Q3).
        # Actually for "contains Q3 as subgraph" we just need those 12 edges to exist.
        # But we need to enumerate distinct Q3s. Two different 8-sets with the same 12 edges won't happen.
        # Let's enumerate all 8-subsets of 16 vertices, and for each, check if it induces a Q3.
        
        for verts in combinations(range(n), 8):
            vset = set(verts)
            # Check if the induced subgraph on these 8 vertices is exactly Q3
            # Q3 has 12 edges. Count edges in K16 induced on these 8 vertices.
            # Actually we're checking the FULL graph K16, so all 28 edges exist.
            # We need to find a labeling of the 8 vertices as a cube.
            # A 3-cube: vertices can be labeled by 3-bit strings. Edges connect strings differing in one bit.
            # For 8 vertices, we need to check if there exists a bijection to {0,1}^3 such that
            # two vertices are adjacent iff their labels differ in exactly one bit.
            # In K16, all pairs are adjacent, so the induced subgraph is K8, not Q3.
            # Wait - we're looking for Q3 as a SUBGRAPH, not induced subgraph.
            # So we need 8 vertices and 12 specific edges forming a Q3 pattern.
            # The 12 edges must exist in our graph. Since we start with K16, they all exist.
            # The issue is: which 12-edge subsets of the 28 edges on 8 vertices form a Q3?
            # For a given 8-set, there are multiple ways to select 12 edges to form a Q3.
            # Actually, for 8 vertices, the Q3 structure is: partition into 4 pairs of "opposite" vertices
            # (distance 3 in the cube), and the 12 edges are all non-opposite pairs... no.
            # Let me think differently. Q3 on 8 vertices: each vertex has degree 3.
            # So we need to select 12 edges from the 28 such that each vertex has degree exactly 3
            # and the resulting graph is 3-regular and bipartite with parts of size 4.
            # Actually, simpler: enumerate all 3-regular bipartite graphs on 8 vertices with parts of 4.
            # There are not that many. Let's just brute-force: for each 8-set, try all ways to assign
            # each vertex a 3-bit label (up to permutation of bits and complementation), check if the
            # 12 edges of the cube are a subset of the complete graph (always true in K16).
            # The number of distinct Q3 edge-sets on a fixed 8-vertex set: 
            # The automorphism group of Q3 has size 48. The number of ways to label 8 vertices with
            # {0,1}^3 is 8! = 40320. Distinct edge sets = 40320 / 48 = 840.
            # That's too many to enumerate for each 8-set (C(16,8)=12870 * 840 is too much).
            
            # Better approach: just enumerate the 12-edge subsets directly.
            # Actually, for the complement removal strategy, I need to know which edges belong to which Q3s.
            # Let me use a different enumeration: 
            # A Q3 is determined by choosing 8 vertices and a "cube structure" on them.
            # The cube structure: pick vertex a0, its 3 neighbors a1,a2,a4, then a3=a1a2's other neighbor,
            # a5=a1a4's other, a6=a2a4's other, a7=a3a5=a3a6=a5a6.
            # In K16, for any ordered choice of a0, then 3 neighbors from remaining 7, we get a candidate.
            # But many will give the same edge set.
            
            # Most practical: for each 8-set, check if it CAN host a Q3 (always yes in K16),
            # and record the specific 12-edge sets. But 840 per 8-set is too many.
            
            # Alternative: don't enumerate all Q3s upfront. Instead, during the removal process,
            # check if a given edge set (current graph) contains a Q3 by random sampling.
            pass
        return q3s
    
    # Revised approach: don't pre-enumerate all Q3s (too many). Instead:
    # 1. Start with all 120 edges present.
    # 2. Repeatedly: find a Q3 in the current graph, remove one of its edges.
    # 3. To find a Q3: use the creates_q3 check in reverse - pick random 8-sets and check.
    # 4. After no Q3 found, do swap-based local search to add back edges.
    
    def has_q3_in_adj(adj, max_trials=200):
        """Check if adj contains a Q3. Returns the 12 edges of a Q3 if found, else None."""
        for _ in range(max_trials):
            verts = random.sample(range(n), 8)
            vset = set(verts)
            # Check all possible cube labelings (8! / 48 = 840 is too many)
            # Instead: check if there exist 3 disjoint perfect matchings on these 8 vertices
            # that form a Q3. A Q3 = 3 perfect matchings M1, M2, M3 on 8 vertices such that
            # M1 ∪ M2 ∪ M3 forms a 3-regular bipartite graph.
            # Actually simpler: pick a random vertex as a0, randomly pick 3 of its neighbors in the
            # 8-set as a1,a2,a4, then determine a3,a5,a6,a7 and check all 12 edges exist.
            a0 = random.choice(verts)
            neigh_in_set = [v for v in verts if v != a0 and (adj[a0] & (1 << v))]
            if len(neigh_in_set) < 3:
                continue
            a1, a2, a4 = random.sample(neigh_in_set, 3)
            # a3 must be adjacent to both a1 and a2, in the set, not a0
            a3_cands = [v for v in verts if v not in (a0,a1,a2,a4) and (adj[a1] & (1<<v)) and (adj[a2] & (1<<v))]
            if not a3_cands:
                continue
            a3 = random.choice(a3_cands)
            a5_cands = [v for v in verts if v not in (a0,a1,a2,a3,a4) and (adj[a1] & (1<<v)) and (adj[a4] & (1<<v))]
            if not a5_cands:
                continue
            a5 = random.choice(a5_cands)
            a6_cands = [v for v in verts if v not in (a0,a1,a2,a3,a4,a5) and (adj[a2] & (1<<v)) and (adj[a4] & (1<<v))]
            if not a6_cands:
                continue
            a6 = random.choice(a6_cands)
            a7_cands = [v for v in verts if v not in (a0,a1,a2,a3,a4,a5,a6) and (adj[a3] & (1<<v)) and (adj[a5] & (1<<v)) and (adj[a6] & (1<<v))]
            if not a7_cands:
                continue
            a7 = random.choice(a7_cands)
            # Now check all 12 edges of the Q3 exist
            q3_edges = [(a0,a1),(a0,a2),(a0,a4),(a1,a3),(a1,a5),(a2,a3),(a2,a6),
                         (a3,a7),(a4,a5),(a4,a6),(a5,a7),(a6,a7)]
            all_exist = True
            for u,v in q3_edges:
                if not (adj[u] & (1<<v)):
                    all_exist = False
                    break
            if all_exist:
                # Verify it's really a Q3 (no extra constraints needed for subgraph)
                return q3_edges
        return None

    def count_edges(adj):
        return sum(adj[u].bit_count() for u in range(n)) // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def creates_q3_fast(u, v, adj):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    # Strategy: Start from K16, remove edges to eliminate all Q3s, then swap-repair.
    random.seed(42)
    
    best_adj = [0] * n
    best_count = -1

    for restart in range(8):
        if time.time() > deadline - 20:
            break
        
        # Start with complete graph
        adj = [(1 << n) - 2] * n  # all edges present (self-loops masked out)
        for i in range(n):
            adj[i] &= ~(1 << i)
        
        # Phase 1: Remove edges to destroy all Q3s
        removals = 0
        max_removals = 40  # We expect to remove around 20-30 edges
        for _ in range(max_removals):
            if time.time() > deadline - 15:
                break
            
            q3 = has_q3_in_adj(adj, max_trials=500)
            if q3 is None:
                # Try more trials to be sure
                q3 = has_q3_in_adj(adj, max_trials=1000)
                if q3 is None:
                    break
            
            # Remove the edge in this Q3 that appears in the most other Q3s
            # Approximate: remove a random edge from the Q3, or the one with highest degree
            # Better: remove the edge whose removal breaks the most potential Q3s
            # Simple heuristic: remove the edge with the most endpoints having high degree
            best_edge_to_remove = None
            best_score = -1
            for u, v in q3:
                # Score: prefer removing edges between high-degree vertices
                score = adj[u].bit_count() + adj[v].bit_count()
                if score > best_score:
                    best_score = score
                    best_edge_to_remove = (u, v)
            
            if best_edge_to_remove:
                u, v = best_edge_to_remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                removals += 1
        
        # Phase 2: Swap-based local search to add back edges
        # Try to add missing edges, possibly removing other edges to make room
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            
            # Try adding a random missing edge
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing:
                break
            
            u_add, v_add = random.choice(missing)
            
            # Can we add it directly?
            if not creates_q3_fast(u_add, v_add, adj):
                adj[u_add] |= (1 << v_add)
                adj[v_add] |= (1 << u_add)
                continue
            
            # Need to remove some edge to make room. Try removing edges incident to u_add or v_add
            candidates = []
            for w in range(n):
                if w != u_add and w != v_add:
                    if adj[u_add] & (1 << w):
                        candidates.append((u_add, w))
                    if adj[v_add] & (1 << w):
                        candidates.append((v_add, w))
            
            if not candidates:
                continue
            
            u_rem, v_rem = random.choice(candidates)
            
            trial = adj[:]
            trial[u_rem] &= ~(1 << v_rem)
            trial[v_rem] &= ~(1 << u_rem)
            
            if not creates_q3_fast(u_add, v_add, trial):
                trial[u_add] |= (1 << u_add)  # placeholder
                trial[u_add] &= ~(1 << u_add)  # clear
                trial[u_add] |= (1 << v_add)
                trial[v_add] |= (1 << u_add)
                
                # Try to fill more missing edges greedily
                for _fill in range(3):
                    if time.time() > deadline - 5:
                        break
                    missing_trial = [(a, b) for a in range(n) for b in range(a+1, n) if not (trial[a] & (1 << b))]
                    random.shuffle(missing_trial)
                    for a, b in missing_trial:
                        if time.time() > deadline - 5:
                            break
                        if not creates_q3_fast(a, b, trial):
                            trial[a] |= (1 << b)
                            trial[b] |= (1 << a)
                
                new_count = count_edges(trial)
                if new_count > count_edges(adj):
                    adj = trial
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Final greedy fill
    if time.time() < deadline - 2:
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (best_adj[u] & (1 << v))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 1:
                break
            if not creates_q3_fast(u, v, best_adj):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
        
        best_count = count_edges(best_adj)
    
    return edges_from_adj(best_adj)
