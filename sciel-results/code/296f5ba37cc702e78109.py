import time
import random
import itertools
from functools import lru_cache

# Q3 edges in canonical form (0..7)
_Q3_CANONICAL_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    n_vertices = n
    
    # Precompute all Q3 embeddings as (vertex_mask, edge_mask) pairs
    # vertex_mask: 16-bit mask of which vertices are used
    # edge_mask: 120-bit mask of which edges are present (bit i corresponds to edge i in all_edges)
    
    all_edges = list(itertools.combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute all valid Q3 embeddings
    q3_embeddings = []  # List of (vertex_mask, edge_mask)
    
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Generate all injective mappings of Q3 vertices (0..7) to graph vertices (0..15)
    # For each mapping, check if it forms a valid Q3 embedding
    from itertools import permutations
    
    # To speed up, we can use the fact that Q3 has automorphisms.
    # But for n=16, let's just enumerate all P(16,8) = 16!/(16-8)! = 518,918,400 mappings.
    # That's too many. Let's be smarter.
    
    # Alternative: For each set of 8 vertices, check all 8! = 40320 permutations.
    # 16C8 * 8! = 12870 * 40320 = 518,918,400. Same thing.
    
    # Let's use a different approach: enumerate all 8-vertex subsets, then for each,
    # check if the induced subgraph contains Q3.
    
    # Actually, let's precompute by iterating over all subsets of 8 vertices and all permutations.
    # But that's 518M operations which might be too slow at startup.
    
    # Better: Use the structure of Q3. Q3 is bipartite with partitions {0,3,5,6} and {1,2,4,7}.
    # Wait, let me check: 0-1, 0-2, 0-4. 1 is odd (1 bit), 2 is even (1 bit), 4 is even (1 bit).
    # Actually in binary: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111.
    # Q3 connects vertices differing in exactly one bit.
    # Bipartition by parity (number of 1 bits mod 2):
    # Even parity (0 or 2 ones): 0(000), 3(011), 5(101), 6(110)
    # Odd parity (1 or 3 ones): 1(001), 2(010), 4(100), 7(111)
    
    # So Q3 is bipartite with parts A={0,3,5,6} and B={1,2,4,7}.
    
    # For a Q3 embedding, we need to map A to 4 graph vertices and B to 4 graph vertices.
    # Edges only go between A and B in Q3.
    
    # Let's precompute for each pair of 4-sets (S_A, S_B) from the 16 vertices,
    # whether there exists a bijection f: A->S_A, g: B->S_B such that all Q3 edges map to graph edges.
    
    # Number of 4-sets from 16: C(16,4) = 1820.
    # Pairs of 4-sets: 1820^2 = 3,312,400.
    # For each pair, check 4! * 4! = 576 bijections.
    # Total: 3,312,400 * 576 = 1,907,942,400. Still too many.
    
    # Let's just do the direct enumeration but with early termination.
    # Or, let's use a different strategy for the main loop.
    
    # Actually, let's reconsider. The current program scores 1.41, meaning ~55 edges.
    # The check in the current program is slow but works. Let me optimize it instead.
    
    # Key insight: When adding edge (u,v), we only need to check if there's a Q3 using (u,v).
    # We can precompute for each graph edge (u,v), which Q3 embeddings could use it.
    
    # Let me try a completely different approach: 
    # Use integer bitmasks for adjacency and precompute Q3 patterns.
    
    # Represent the graph as a list of 16 integers, where adj[i] is a 16-bit mask of neighbors.
    
    # Precompute all Q3 embeddings more efficiently:
    # For each ordered pair of adjacent Q3 vertices (i,j) [24 pairs], and for each ordered pair
    # of graph vertices (u,v) [16*15=240 pairs], we can try to extend to a full Q3 embedding.
    # But that's what the current code does, just slowly.
    
    # Let me optimize the backtracker using bitmasks.
    
    def creates_q3_fast(u, v, adj_masks):
        """
        Check if adding edge (u,v) to the graph (represented by adj_masks) creates a Q3.
        adj_masks[i] is a 16-bit integer with bit j set iff edge (i,j) exists.
        We assume (u,v) is NOT yet in adj_masks.
        """
        # For each of the 24 orientations (12 edges * 2 directions) in Q3:
        # Try to complete the embedding using backtracking with bitmasks.
        
        for q3_u, q3_v in q3_edges:
            # Try u->q3_u, v->q3_v
            if _try_complete(q3_u, q3_v, u, v, adj_masks, q3_adj):
                return True
            # Try u->q3_v, v->q3_u  
            if _try_complete(q3_v, q3_u, u, v, adj_masks, q3_adj):
                return True
        return False
    
    def _try_complete(q3_a, q3_b, g_a, g_b, adj_masks, q3_adj):
        """
        Try to complete a Q3 embedding where q3_a->g_a and q3_b->g_b.
        Returns True if a valid embedding exists.
        """
        # Vertices already mapped in Q3: {q3_a, q3_b}
        # Vertices already mapped in G: {g_a, g_b}
        
        # Unmapped Q3 vertices
        unmapped_q3 = [x for x in range(8) if x != q3_a and x != q3_b]
        
        # For efficiency, order unmapped_q3 by how many constraints they have to mapped vertices
        # A Q3 vertex with more neighbors already mapped will have tighter constraints
        
        def constraint_count(q3_v):
            count = 0
            for nb in q3_adj[q3_v]:
                if nb == q3_a or nb == q3_b:
                    count += 1
            return count
        
        unmapped_q3.sort(key=constraint_count, reverse=True)
        
        # Backtracking with bitmasks
        # state: mapping from Q3 vertex to G vertex (for unmapped ones)
        # We'll use a dict for the partial mapping
        
        partial_map = {q3_a: g_a, q3_b: g_b}
        used_g_mask = (1 << g_a) | (1 << g_b)
        
        def backtrack(idx):
            if idx == len(unmapped_q3):
                # All Q3 vertices mapped. Verify all Q3 edges exist in G.
                for a, b in q3_edges:
                    ga = partial_map[a]
                    gb = partial_map[b]
                    if not (adj_masks[ga] >> gb) & 1:
                        return False
                return True
            
            q3_v = unmapped_q3[idx]
            
            # Find candidate G vertices
            # Must be adjacent in G to all already-mapped Q3 neighbors of q3_v
            candidates_mask = 0
            valid = True
            for q3_nb in q3_adj[q3_v]:
                if q3_nb in partial_map:
                    g_nb = partial_map[q3_nb]
                    # Candidate must be adjacent to g_nb in G
                    # adj_masks[g_nb] has bit set for each neighbor
                    candidates_mask |= adj_masks[g_nb]
                else:
                    # This neighbor is not yet mapped, so no constraint from it yet
                    pass
            
            # Also, candidate must not be already used
            candidates_mask &= ~used_g_mask
            
            # Iterate through set bits in candidates_mask
            cm = candidates_mask
            while cm:
                # Get lowest set bit
                lb = cm & (-cm)
                g_cand = lb.bit_length() - 1
                cm ^= lb
                
                # Assign
                partial_map[q3_v] = g_cand
                new_used = used_g_mask | (1 << g_cand)
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                del partial_map[q3_v]
            
            return False
        
        return backtrack(0)
    
    # Run random greedy with bitmask adjacency
    best_count = 0
    best_adj = None
    
    num_trials = 20
    
    for t in range(num_trials):
        if time.time() > deadline - 5:
            break
            
        random.seed(t * 12345 + 67890)
        
        # Start with K3,13
        adj_masks = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        
        # Get remaining edges
        remaining_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[u] >> v) & 1:
                    remaining_edges.append((u, v))
        
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline - 5:
                break
            
            # Check if adding (u,v) creates a Q3
            # Temporarily add (u,v) for the check? No, check if it WOULD create a Q3.
            # The Q3 would use edge (u,v), so we need to check if there's a Q3 embedding
            # where (u,v) is one of the edges, and all other edges already exist.
            
            if not creates_q3_fast(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        
        # Count edges
        count = sum(bin(adj_masks[i]).count('1') for i in range(n)) // 2
        
        if count > best_count:
            best_count = count
            best_adj = adj_masks[:]
            
            if best_count >= 60:  # Good enough, stop early
                break
    
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result
