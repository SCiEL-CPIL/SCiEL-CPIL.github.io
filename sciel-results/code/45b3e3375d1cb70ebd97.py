"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_pairs)}

    # ---- Precompute Q3 4x4 patterns ----
    # A Q3 is a 3-regular bipartite graph on 4+4 vertices.
    # There is only one such graph up to isomorphism.
    # We generate all 576 permutations of row/col indices for the 12 edges.
    # The base Q3 edges (0-3 vs 4-7) are:
    # 0-4, 0-5, 0-6
    # 1-5, 1-6, 1-7  <-- Wait, let's use the standard one.
    # Let's use the one derived from bit flips.
    # V0: 000(0), 011(3), 101(5), 110(6)
    # V1: 001(1), 010(2), 100(4), 111(7)
    # Edges:
    # 0(000) - 1(001), 2(010), 4(100)
    # 3(011) - 1(001), 2(010), 7(111)
    # 5(101) - 1(001), 4(100), 7(111)
    # 6(110) - 2(010), 4(100), 7(111)
    
    base_edges = [
        (0, 0), (0, 1), (0, 2), # 0-1, 0-2, 0-4 (mapped to 0,1,2 in B?)
        # Let's map V0 to rows 0-3 and V1 to cols 0-3.
        # V0 = [0, 3, 5, 6] -> indices 0, 1, 2, 3
        # V1 = [1, 2, 4, 7] -> indices 0, 1, 2, 3
        # Edges:
        # 0(0) - 1(0), 2(1), 4(2) -> (0,0), (0,1), (0,2)
        # 3(1) - 1(0), 2(1), 7(3) -> (1,0), (1,1), (1,3)
        # 5(2) - 1(0), 4(2), 7(3) -> (2,0), (2,2), (2,3)
        # 6(3) - 2(1), 4(2), 7(3) -> (3,1), (3,2), (3,3)
    ]
    base_q3_edges = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,3), (2,0), (2,2), (2,3), (3,1), (3,2), (3,3)]
    
    q3_patterns = set()
    for perm_r in itertools.permutations(range(4)):
        for perm_c in itertools.permutations(range(4)):
            mask = 0
            for r, c in base_q3_edges:
                pr = perm_r[r]
                pc = perm_c[c]
                mask |= (1 << (pr * 4 + pc))
            q3_patterns.add(mask)
            
    # Precompute: for each 16-bit mask, does it contain a Q3 pattern?
    mask_is_q3_superset = [False] * (1 << 16)
    for m in range(1 << 16):
        for p in q3_patterns:
            if (m & p) == p:
                mask_is_q3_superset[m] = True
                break

    # ---- Precompute relevant (A,B) pairs for each edge ----
    # For each edge (u,v), find all (A,B) such that u in A, v in B or u in B, v in A.
    # To save memory and time, we store the list of (A_tuple, B_tuple) for each edge.
    # A and B are tuples of 4 vertices.
    edge_to_ab_pairs = {e: [] for e in all_pairs}
    
    all_verts = list(range(n))
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            # Add this (A,B) to all edges between A and B
            for a in A:
                for b in B:
                    e = (a, b) if a < b else (b, a)
                    edge_to_ab_pairs[e].append((A, B))

    # ---- Build graph as bitsets ----
    adj = [0] * n
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    def edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    # ---- Seed: K7 + K7 + K2 (57 edges, Q3-free) ----
    for i in range(7):
        for j in range(i+1, 7):
            add_edge(i, j)
    for i in range(7, 14):
        for j in range(i+1, 14):
            add_edge(i, j)
    add_edge(14, 15)

    best_edges = edge_count()
    best_adj = adj[:]

    def time_left():
        return deadline - time.time()

    # ---- Fast Q3 check for adding (u,v) ----
    def creates_q3(u, v):
        e = (u, v) if u < v else (v, u)
        for A, B in edge_to_ab_pairs[e]:
            # Build 16-bit mask for this A,B pair
            # Map A to 0-3 and B to 0-3
            # We need to know the index of each vertex in A and B
            # To avoid creating dicts, we can precompute index maps?
            # Or just use a small loop.
            mask = 0
            # We assume A and B are tuples.
            # We need to check edges (a, b) for a in A, b in B.
            # We need to map a to row index and b to col index.
            # Since A and B are fixed tuples, we can use a simple lookup.
            # But doing this for 60k pairs is slow.
            # Optimization: Precompute the 16-bit mask for the *current* graph for all (A,B) pairs?
            # No, it changes.
            
            # Let's just do it.
            # To speed up, we can precompute the index of each vertex in A and B.
            # But A and B are from a precomputed list.
            # Let's store (A, B, A_indices, B_indices) in edge_to_ab_pairs?
            # That would be huge.
            
            # Alternative: Just iterate.
            # 60k pairs * 16 checks = 1M checks.
            # 1M checks in Python is ~0.05s.
            # If we do this for 100 edges, it's 5s. Acceptable.
            
            # However, we can prune: if the current graph already has a Q3, we don't care.
            # But we are checking if *adding* (u,v) creates one.
            
            # Let's try to optimize the mask building.
            # We can precompute for each (A,B) pair, the list of 16 edge indices in the global 120-edge space?
            # No, we need the 16-bit local mask.
            
            # Let's just do it naively first.
            # If too slow, we'll optimize.
            
            # To make it faster, we can precompute the local edge positions for each (A,B) pair.
            # Let's restructure edge_to_ab_pairs to store (A, B, local_edges)
            # where local_edges is a list of 16 (global_edge_idx, local_bit_pos)
            
            # Actually, let's just use the adjacency matrix.
            # For a in A, for b in B:
            #   if has_edge(a,b) or (a==u and b==v) or (a==v and b==u):
            #       mask |= 1 << (i*4+j)
            
            # We need i, j indices.
            # We can use a dictionary for A and B?
            # Creating dicts is slow.
            # Since size is 4, we can use a linear search or a precomputed map.
            
            # Let's precompute a global map: vertex -> index in A? No, A varies.
            
            # Let's just use a small helper.
            # This might be the bottleneck.
            
            # Let's try a different approach:
            # Precompute for each (A,B) pair, the 16 global edge indices.
            # Then, to build the mask, we check if those global edges are in adj.
            
            pass # Placeholder for actual implementation in the edit below

    # Re-implement edge_to_ab_pairs with precomputed local edge indices
    edge_to_ab_pairs_opt = {e: [] for e in all_pairs}
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            # Precompute local edge indices
            local_edges = []
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    e = (a, b) if a < b else (b, a)
                    local_edges.append((edge_index[e], i*4+j))
            
            for a in A:
                for b in B:
                    e = (a, b) if a < b else (b, a)
                    edge_to_ab_pairs_opt[e].append(local_edges)

    # Update creates_q3 to use opt version
    def creates_q3_fast(u, v):
        e = (u, v) if u < v else (v, u)
        for local_edges in edge_to_ab_pairs_opt[e]:
            mask = 0
            for ge_idx, bit_pos in local_edges:
                # Check if edge is in graph
                # ge_idx is index in all_pairs
                # all_pairs[ge_idx] is (u, v)
                # We can just check adj
                a, b = all_pairs[ge_idx]
                if adj[a] & (1 << b):
                    mask |= (1 << bit_pos)
                # Note: The new edge (u,v) is NOT in adj yet.
                # So we must manually add it if it's in this local set.
                if (a == u and b == v) or (a == v and b == u):
                    mask |= (1 << bit_pos)
            
            if mask_is_q3_superset[mask]:
                return True
        return False

    # ---- Greedy: try to add every missing edge ----
    missing = [e for e in all_pairs if not has_edge(*e)]
    random.seed(42)
    random.shuffle(missing)

    for u, v in missing:
        if time_left() < 15:
            break
        if not creates_q3_fast(u, v):
            add_edge(u, v)
            if edge_count() > best_edges:
                best_edges = edge_count()
                best_adj = adj[:]

    # ---- Local Search ----
    for _ in range(500):
        if time_left() < 10:
            break
        # Try to add a random missing edge
        miss = [e for e in all_pairs if not has_edge(*e)]
        if not miss:
            break
        random.shuffle(miss)
        added_any = False
        for u, v in miss:
            if not creates_q3_fast(u, v):
                add_edge(u, v)
                added_any = True
                break
        
        if added_any:
            if edge_count() > best_edges:
                best_edges = edge_count()
                best_adj = adj[:]
        else:
            # Try a swap: remove one, add two
            rem_edges = [e for e in all_pairs if has_edge(*e)]
            if rem_edges:
                ru, rv = random.choice(rem_edges)
                remove_edge(ru, rv)
                miss = [e for e in all_pairs if not has_edge(*e)]
                random.shuffle(miss)
                added = 0
                for u, v in miss:
                    if added >= 2:
                        break
                    if not creates_q3_fast(u, v):
                        add_edge(u, v)
                        added += 1
                
                if edge_count() > best_edges:
                    best_edges = edge_count()
                    best_adj = adj[:]
                else:
                    # Revert
                    for u, v in miss[:added]:
                        remove_edge(u, v)
                    add_edge(ru, rv)

    # Restore best state
    adj[:] = best_adj[:]
    
    # Final validity check
    # We trust creates_q3_fast, but let's do a full scan just in case
    # Full scan is slow, so we only do it if we have time
    if time_left() > 5:
        found = False
        for A in itertools.combinations(all_verts, 4):
            Aset = set(A)
            for B in itertools.combinations(all_verts, 4):
                if Aset & set(B):
                    continue
                mask = 0
                for i, a in enumerate(A):
                    for j, b in enumerate(B):
                        if adj[a] & (1 << b):
                            mask |= (1 << (i*4+j))
                if mask_is_q3_superset[mask]:
                    found = True
                    break
            if found:
                break
        if found:
            # Fallback to seed
            adj = [0]*n
            for i in range(7):
                for j in range(i+1, 7):
                    add_edge(i, j)
            for i in range(7, 14):
                for j in range(i+1, 14):
                    add_edge(i, j)
            add_edge(14, 15)

    return [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
# EVOLVE-BLOCK-END
