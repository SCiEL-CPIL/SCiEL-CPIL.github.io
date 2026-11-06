"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 145
    n = 16

    # Q3: 8 vertices, 12 edges
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            diff = u ^ v
            if diff & (diff - 1) == 0:
                q3_edges.append((u, v))
    assert len(q3_edges) == 12

    q3_adj = {v: set() for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)  # 120

    # Precompute all 24 permutations of 4 elements for Q3 check
    perms4 = list(permutations(range(4)))

    def has_q3_fast(adj_bitmasks, u, v):
        """Check if adding edge (u,v) creates a Q3.
        adj_bitmasks[i] is bitmask of neighbors of i (NOT including the new edge (u,v) yet)."""
        # Map u->0, v->1 in Q3 (0^1=1, valid edge)
        # Also try u->1, v->0
        for mapping in [(u, v), (v, u)]:
            qv_to_kv = {0: mapping[0], 1: mapping[1]}
            used = {mapping[0], mapping[1]}
            
            # Q3 neighbors:
            # 0: {1, 2, 4}
            # 1: {0, 3, 5}
            # 2: {0, 3, 6}
            # 3: {1, 2, 7}
            # 4: {0, 5, 6}
            # 5: {1, 4, 7}
            # 6: {2, 4}
            # 7: {3, 5}
            
            # Vertex 2 must be a neighbor of 0 (i.e., mapping[0]) in K16
            # Vertex 4 must be a neighbor of 0 in K16
            # Vertex 3 must be a neighbor of 1 (i.e., mapping[1]) in K16
            # Vertex 5 must be a neighbor of 1 in K16
            
            # Candidates for Q3 vertex 2: neighbors of mapping[0] not in used
            # Candidates for Q3 vertex 4: neighbors of mapping[0] not in used
            # Candidates for Q3 vertex 3: neighbors of mapping[1] not in used
            # Candidates for Q3 vertex 5: neighbors of mapping[1] not in used
            
            mask0 = adj_bitmasks[mapping[0]]
            mask1 = adj_bitmasks[mapping[1]]
            
            # Remove used vertices
            mask0 &= ~(1 << mapping[0])
            mask0 &= ~(1 << mapping[1])
            mask1 &= ~(1 << mapping[0])
            mask1 &= ~(1 << mapping[1])
            
            # Get candidate lists using bit operations for speed
            cands2 = []
            cands4 = []
            cands3 = []
            cands5 = []
            
            for i in range(n):
                if i in used:
                    continue
                if mask0 & (1 << i):
                    cands2.append(i)
                    cands4.append(i)
                if mask1 & (1 << i):
                    cands3.append(i)
                    cands5.append(i)
            
            # Try all combinations
            for c2 in cands2:
                for c4 in cands4:
                    if c4 == c2:
                        continue
                    for c3 in cands3:
                        if c3 == c2 or c3 == c4:
                            continue
                        for c5 in cands5:
                            if c5 == c2 or c5 == c3 or c5 == c4:
                                continue
                            
                            # Now we have 6 vertices assigned: 0,1,2,3,4,5
                            # Need to assign 6 and 7
                            # Q3 vertex 6 connects to 2 and 4
                            # Q3 vertex 7 connects to 3 and 5
                            
                            # c6 must be neighbor of both c2 and c4
                            mask_c2 = adj_bitmasks[c2]
                            mask_c4 = adj_bitmasks[c4]
                            mask_c6 = mask_c2 & mask_c4
                            mask_c6 &= ~(1 << mapping[0])
                            mask_c6 &= ~(1 << mapping[1])
                            mask_c6 &= ~(1 << c2)
                            mask_c6 &= ~(1 << c3)
                            mask_c6 &= ~(1 << c4)
                            mask_c6 &= ~(1 << c5)
                            
                            cands6 = []
                            for i in range(n):
                                if i not in (mapping[0], mapping[1], c2, c3, c4, c5) and (mask_c6 & (1 << i)):
                                    cands6.append(i)
                            
                            if not cands6:
                                continue
                            
                            # c7 must be neighbor of both c3 and c5
                            mask_c3 = adj_bitmasks[c3]
                            mask_c5 = adj_bitmasks[c5]
                            mask_c7 = mask_c3 & mask_c5
                            mask_c7 &= ~(1 << mapping[0])
                            mask_c7 &= ~(1 << mapping[1])
                            mask_c7 &= ~(1 << c2)
                            mask_c7 &= ~(1 << c3)
                            mask_c7 &= ~(1 << c4)
                            mask_c7 &= ~(1 << c5)
                            
                            cands7 = []
                            for i in range(n):
                                if i not in (mapping[0], mapping[1], c2, c3, c4, c5) and (mask_c7 & (1 << i)):
                                    cands7.append(i)
                            
                            if not cands7:
                                continue
                            
                            for c6 in cands6:
                                for c7 in cands7:
                                    if c7 == c6:
                                        continue
                                    # Check all Q3 edges
                                    # Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                                    # Already checked: (0,1) is the new edge, (0,2),(0,4),(1,3),(1,5) by construction
                                    # Need to check: (2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                                    
                                    # (2,3): c2-c3
                                    if not (adj_bitmasks[c2] & (1 << c3)):
                                        continue
                                    # (2,6): c2-c6
                                    if not (adj_bitmasks[c2] & (1 << c6)):
                                        continue
                                    # (3,7): c3-c7
                                    if not (adj_bitmasks[c3] & (1 << c7)):
                                        continue
                                    # (4,5): c4-c5
                                    if not (adj_bitmasks[c4] & (1 << c5)):
                                        continue
                                    # (4,6): c4-c6
                                    if not (adj_bitmasks[c4] & (1 << c6)):
                                        continue
                                    # (5,7): c5-c7
                                    if not (adj_bitmasks[c5] & (1 << c7)):
                                        continue
                                    # (6,7): c6-c7
                                    if not (adj_bitmasks[c6] & (1 << c7)):
                                        continue
                                    
                                    return True
        return False
    
    # Build initial graph: K_{2,14} (guaranteed Q3-free since Q3 needs 4+4 bipartition)
    adj = [0] * n
    for a in range(2):
        for b in range(2, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    edges = set()
    for a in range(2):
        for b in range(2, 16):
            edges.add((a, b))
    
    best_adj = adj[:]
    best_edges = set(edges)
    best_count = len(edges)
    
    random.seed(42)
    
    # Greedy: try to add edges in random order
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        
        adj = [0] * n
        for a in range(2):
            for b in range(2, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        edges = set()
        for a in range(2):
            for b in range(2, 16):
                edges.add((a, b))
        
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 20:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]
    
    # Local search: try removing one edge and adding as many as possible
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            # Try each edge removal
            for e in list(edges):
                if time.time() > deadline - 15:
                    break
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add as many missing edges as possible
                missing = [ne for ne in all_edges if ne not in edges]
                random.shuffle(missing)
                
                added_any = False
                for ne in missing:
                    if time.time() > deadline - 15:
                        break
                    nu, nv = ne
                    if has_q3_fast(adj, nu, nv):
                        continue
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    added_any = True
                
                if added_any and len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
                    best_adj = adj[:]
                    improved = True
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
