"""Dense Q3-free graph on 16 vertices via precomputed forbidden-edge sets + iterative local search."""
import time
import random
from itertools import combinations, product

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n = 16
    
    # Precompute all forbidden edge sets (sets of 12 edges that form a Q3)
    # A Q3 on 8 vertices has 12 edges. We need to find all such edge sets in K16.
    # Vertices of Q3 are labeled by 3-bit strings. Edges connect vertices differing in one bit.
    
    # The 12 edges of Q3 in the canonical labeling (vertices 0..7 = 000..111):
    # Edge set: pairs differing in exactly one bit
    q3_edges_canonical = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            # Check if diff is a power of 2 (differ in exactly one bit)
            if diff & (diff - 1) == 0:
                q3_edges_canonical.append((i, j))
    
    # q3_edges_canonical has 12 edges
    # Now we need all injective mappings from {0..7} to {0..15}
    # For each such mapping, the 12 edges become a forbidden set.
    
    # Total: P(16,8) = 16!/(16-8)! = 518,918,400 mappings
    # But we can reduce by symmetry: fix vertex 0 of Q3 to map to vertex 0 of graph
    # Then P(15,7) = 15!/8! = 32,432,400 still too many
    
    # Better: use the fact that Q3 has automorphism group of size 48
    # So for each 8-subset of vertices, there are C(16,8)*8!/48 distinct Q3s
    # = 12870 * 40320 / 48 = 12870 * 840 = 10,810,800
    # Still too many to enumerate explicitly in reasonable time.
    
    # Alternative: Use a smarter approach.
    # For a given edge set E, it contains a Q3 iff there exist 8 vertices inducing exactly the Q3 pattern.
    # Let's precompute for each pair of vertices (u,v) the set of "completions" that would form a Q3.
    
    # Actually, let's use a different strategy:
    # 1. Precompute all 8-vertex subsets that could potentially host a Q3
    # 2. For each such subset, determine all Q3 edge patterns
    # 3. Build a list of forbidden edge sets
    
    # But 10.8 million forbidden sets is too many to store.
    
    # Better approach: Use a SAT-like or backtracking method, or use local search with
    # a fast Q3-detection oracle.
    
    # Let's use the following strategy:
    # - Represent the graph as an adjacency bitmask
    # - For local search, we need a fast check: does adding edge (u,v) create a Q3?
    # - We can precompute, for each edge (u,v), the set of other edges that would complete a Q3 with (u,v)
    
    # For a fixed edge (u,v) in the Q3, the remaining 6 vertices and 11 edges must form
    # a specific pattern. Let's precompute all such patterns.
    
    # In Q3, given an edge (u,v), the structure is determined up to isomorphism.
    # Let's enumerate all Q3s where u maps to 000 and v maps to 001 (one specific edge orientation).
    
    # The 12 edges of Q3 with 000=0, 001=1:
    # Vertices: 000,001,010,011,100,101,110,111
    # Edges from 000: (000,001), (000,010), (000,100)
    # Edges from 001: (001,011), (001,101)
    # Edges from 010: (010,011), (010,110)
    # Edges from 100: (100,101), (100,110)
    # Edges from 011: (011,111)
    # Edges from 101: (101,111)
    # Edges from 110: (110,111)
    # That's 12 edges.
    
    # So for a fixed edge (u,v) corresponding to (000,001), the other 6 vertices are:
    # 010, 011, 100, 101, 110, 111
    # And the other 11 edges are determined.
    
    # For each permutation of 6 distinct vertices from the remaining 14, assign them to
    # positions 010, 011, 100, 101, 110, 111. There are P(14,6) = 14!/8! = 2,162,160.
    # For each such assignment, we get a forbidden edge set of 12 edges.
    
    # Total over all edges: C(16,2) * 2,162,160 = 120 * 2,162,160 = 259,459,200
    # Still too many.
    
    # Let's use a different approach: direct local search with a fast Q3 check.
    
    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) to graph with adjacency masks 'adj' creates a Q3."""
        # In Q3, u and v are adjacent. WLOG, think of u as 000 and v as 001.
        # Neighbors of u (excluding v): 010, 100 -> two vertices, call them a, b
        # Neighbors of v (excluding u): 011, 101 -> two vertices, call them c, d
        # a=010 is adjacent to c=011 and to 110
        # b=100 is adjacent to d=101 and to 110
        # c=011 is adjacent to 111
        # d=101 is adjacent to 111
        # 110 is adjacent to 111
        # So the structure is:
        # u -- a -- 110
        # |       |
        # v -- b -- 111
        #     |     |
        #     c ----+
        #     d ----+
        
        # More precisely:
        # u's neighbors (not v): a, b
        # v's neighbors (not u): c, d
        # a is adjacent to c and to e (where e is the vertex at 110)
        # b is adjacent to d and to e
        # c is adjacent to f (vertex at 111)
        # d is adjacent to f
        # e is adjacent to f
        
        # So we need:
        # - a, b in adj[u] \\ {v}
        # - c, d in adj[v] \\ {u}
        # - c in adj[a], d in adj[b]
        # - There exists e adjacent to a, b, and f
        # - There exists f adjacent to c, d, e
        
        neigh_u = adj[u] & ~(1 << v) & ~(1 << u)
        neigh_v = adj[v] & ~(1 << u) & ~(1 << v)
        
        # Try all pairs (a, b) from neighbors of u
        a_bits = neigh_u
        while a_bits:
            a_bit = a_bits & (-a_bits)
            a = a_bit.bit_length() - 1
            a_bits -= a_bit
            
            b_bits = neigh_u & ~(1 << a)
            while b_bits:
                b_bit = b_bits & (-b_bits)
                b = b_bit.bit_length() - 1
                b_bits -= b_bit
                
                # c must be in adj[a] ∩ adj[v] \ {u, v, a, b}
                # d must be in adj[b] ∩ adj[v] \ {u, v, a, b, c}
                # Actually c is in adj[a] ∩ adj[v], d is in adj[b] ∩ adj[v]
                # and c != d, c != a, c != b, d != a, d != b
                
                c_candidates = adj[a] & adj[v] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                d_candidates = adj[b] & adj[v] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                
                c_bits = c_candidates
                while c_bits:
                    c_bit = c_bits & (-c_bits)
                    c = c_bit.bit_length() - 1
                    c_bits -= c_bit
                    
                    d_cand = d_candidates & ~(1 << c)
                    if not d_cand:
                        continue
                    
                    d_bits = d_cand
                    while d_bits:
                        d_bit = d_bits & (-d_bits)
                        d = d_bit.bit_length() - 1
                        d_bits -= d_bit
                        
                        # Now find e: adjacent to a and b, not in {u,v,a,b,c,d}
                        e_cand = (adj[a] & adj[b]) & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if not e_cand:
                            continue
                        
                        # Find f: adjacent to c, d, e, not in {u,v,a,b,c,d,e}
                        used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        
                        e_bits = e_cand
                        while e_bits:
                            e_bit = e_bits & (-e_bits)
                            e = e_bit.bit_length() - 1
                            e_bits -= e_bit
                            
                            f_cand = (adj[c] & adj[d] & adj[e]) & ~used & ~(1 << e)
                            if f_cand:
                                # Found a Q3
                                return True
        
        return False
    
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
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
    
    def greedy_fill(adj, edge_list, deadline):
        """Try to add each edge in edge_list if it doesn't create a Q3."""
        added = 0
        for u, v in edge_list:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added
    
    def local_search(adj, deadline):
        """Iterative local search with random restarts and swaps."""
        best_adj = adj[:]
        best_count = edge_count(adj)
        
        # First, greedily add all possible missing edges
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        random.shuffle(missing)
        greedy_fill(adj, missing, deadline)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Now try swaps: remove one edge, try to add multiple others
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            trial = best_adj[:]
            
            # Build list of existing edges
            existing = []
            for u in range(n):
                mask = trial[u]
                while mask:
                    bit = mask & (-mask)
                    v = bit.bit_length() - 1
                    if v > u:
                        existing.append((u, v))
                    mask -= bit
            
            if not existing:
                break
            
            # Try several random swaps
            for swap_try in range(30):
                if time.time() > deadline - 5:
                    break
                
                trial = best_adj[:]
                
                # Remove a random edge
                u_rem, v_rem = random.choice(existing)
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                # Now greedily add missing edges
                missing = [(u, v) for u in range(n) for v in range(u+1, n) 
                           if not (trial[u] & (1 << v))]
                random.shuffle(missing)
                greedy_fill(trial, missing, deadline)
                
                new_count = edge_count(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
        
        return best_adj, best_count
    
    # Main search loop
    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        
        # Random initialization: add edges randomly
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        greedy_fill(adj, all_edges, deadline)
        
        adj, cnt = local_search(adj, deadline)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Print progress every 10 restarts
        if (restart + 1) % 10 == 0:
            elapsed = time.time() - (deadline - 110)
            print(f"Restart {restart+1}/{num_restarts}, best_count={best_count}, elapsed={elapsed:.1f}s")
    
    # Final polish
    if time.time() < deadline - 3:
        missing = [(u, v) for u in range(n) for v in range(u+1, n) 
                   if not (best_adj[u] & (1 << v))]
        random.shuffle(missing)
        greedy_fill(best_adj, missing, deadline)
        best_count = edge_count(best_adj)
    
    return edges_from_adj(best_adj)
