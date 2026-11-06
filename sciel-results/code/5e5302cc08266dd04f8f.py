"""Construct a large (5,5)-free graph via extension from Paley graph."""

import time
import random

def _paley_graph(n):
    """Construct the Paley graph on n vertices (n must be prime congruent to 1 mod 4)."""
    # For n=37, this is the seed.
    # We use bitmasks for adjacency. adj[i] is an int where bit j is 1 if (i,j) is an edge.
    adj = [0] * n
    if n == 37:
        q = 37
        # Quadratic residues mod 37
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    else:
        # Fallback for other n, though we primarily use 37
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj

def _has_k5(adj, n):
    """Check if there is a K5 in the graph represented by bitmasks."""
    # A K5 exists if there is a vertex c such that its neighborhood contains a K4.
    # To speed up, we can just check all subsets of size 5? No, that's O(n^5).
    # Better: For each vertex c, let N(c) be its neighbors.
    # Check if N(c) contains a K4.
    # To check if N(c) contains a K4, we can iterate over triples in N(c) and check if they form a triangle, 
    # or iterate over pairs and check common neighbors.
    
    for c in range(n):
        nbrs = adj[c]
        # Get list of neighbors
        nbr_list = [i for i in range(n) if (nbrs >> i) & 1]
        m = len(nbr_list)
        if m < 4:
            continue
        
        # Check for K4 in nbr_list
        # Map neighbor to index in nbr_list for faster lookup? Or just use bitmasks.
        # Let's iterate over pairs (a,b) in nbr_list.
        # For each pair, common = adj[a] & adj[b] & nbrs
        # If common contains an edge (c,d), then a,b,c,d is K4.
        for i in range(m):
            a = nbr_list[i]
            for j in range(i + 1, m):
                b = nbr_list[j]
                common = adj[a] & adj[b] & nbrs
                # Check if common contains any edge
                # Iterate over set bits in common
                temp = common
                while temp:
                    # Get lowest set bit
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    temp ^= lsb
                    # Check if there is another bit in common that is connected to idx
                    # If adj[idx] & common & ~(1 << idx) != 0, then there is an edge
                    if adj[idx] & common & ~(1 << idx):
                        return True
    return False

def _has_ind5(adj, n):
    """Check if there is an independent set of size 5."""
    # Independent set of size 5 in G is a clique of size 5 in complement G'.
    # Complement adjacency: comp_adj[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
    # We can compute comp_adj on the fly or precompute.
    # Let's precompute for speed in the loop if called multiple times, 
    # but here it's a one-off check.
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        # Neighbors in complement are non-neighbors in original (excluding self)
        comp_adj[i] = (full_mask ^ (1 << i)) & ~adj[i]
    
    return _has_k5(comp_adj, n)

def _is_valid(adj, n):
    return not _has_k5(adj, n) and not _has_ind5(adj, n)

def _try_extend(adj, n):
    """
    Try to add a new vertex n to the graph on n vertices.
    Returns new adj list of size n+1 if successful, else None.
    """
    new_n = n + 1
    # We need to choose a mask for the new vertex's connections to old vertices.
    # Let mask be an integer of n bits.
    # Constraints:
    # 1. No K5 involving new vertex:
    #    For any K4 in the neighbors of new vertex, it must not form K5.
    #    Actually, if new vertex connects to a set S, then S must not contain a K4.
    #    Because if S contains K4, then new vertex + K4 = K5.
    # 2. No Ind5 involving new vertex:
    #    For any Ind4 in the non-neighbors of new vertex, it must not form Ind5.
    #    Non-neighbors are complement of S (within old vertices).
    #    So complement of S must not contain an Ind4.
    #    Which means S must not have an Ind4 in its complement? 
    #    Let T = non-neighbors. T must not contain Ind4.
    
    # Strategy: Randomly sample masks and check validity.
    # To be smarter, we can start with a mask derived from a random vertex's neighborhood
    # or just random.
    
    # Heuristic: 
    # The set of neighbors S must be K4-free.
    # The set of non-neighbors T must be Ind4-free (which means T is a clique-free set in complement? No, T contains no independent 4-set in G, so T is a clique of size < 5 in G? No.
    # Ind4 in G is K4 in G'. So T must not contain K4 in G'.
    
    # Let's just try random masks.
    # For n=37, the space is 2^37 which is large. But we only need one valid mask.
    # We can use local search on the mask bits.
    
    # Initialize mask randomly
    mask = 0
    for _ in range(n):
        if random.random() < 0.5:
            mask |= (1 << _)
            
    # Local search on mask
    # Objective: Minimize violations.
    # Violations: 
    #   V1: Number of K4s in neighbors of new vertex (in old graph)
    #   V2: Number of Ind4s in non-neighbors of new vertex (in old graph)
    
    # Precompute K4s and Ind4s in the old graph?
    # That's expensive. 
    # Instead, for a given mask, count violations.
    # Counting K4s in a subset S:
    #   Iterate over triples in S, check if they form a triangle.
    #   Or iterate over pairs, check common neighbors in S.
    
    def count_violations(mask):
        # Neighbors S
        # Non-neighbors T
        
        # Count K4s in S
        k4_count = 0
        # List of bits set in mask
        s_bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_bits.append(idx)
            m ^= lsb
        
        # If len(s_bits) < 4, k4_count = 0
        if len(s_bits) >= 4:
            # Check for K4 in s_bits
            # For each pair (a,b) in s_bits, common = adj[a] & adj[b] & mask
            # If common has an edge, it's a K4.
            # To avoid double counting, we can just check if ANY K4 exists.
            # For the objective, we want to minimize the number of K4s.
            # Approximation: check if any K4 exists. If yes, penalty 100.
            # This is a step function, hard for local search.
            # Better: count the number of K4s.
            
            # Efficient counting:
            # For each a in s_bits:
            #   For each b in s_bits with b > a:
            #     common = adj[a] & adj[b] & mask
            #     count edges in common.
            #     Each K4 is counted how many times?
            #     A K4 {a,b,c,d} is counted for pairs (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
            #     For pair (a,b), common contains c,d. Edge (c,d) is counted.
            #     So each K4 contributes 6 to the sum of edges in common over all pairs?
            #     Let's verify.
            #     K4 {1,2,3,4}.
            #     Pair (1,2): common has 3,4. Edge (3,4) exists. Count 1.
            #     Pair (1,3): common has 2,4. Edge (2,4) exists. Count 1.
            #     ...
            #     Yes, sum of edges in common over all pairs = 6 * (#K4s).
            
            total_edges_in_common = 0
            L = len(s_bits)
            for i in range(L):
                a = s_bits[i]
                for j in range(i + 1, L):
                    b = s_bits[j]
                    common = adj[a] & adj[b] & mask
                    # Count bits in common? No, count edges in common.
                    # Iterate bits in common
                    c_bits = common
                    while c_bits:
                        lsb = c_bits & -c_bits
                        idx = lsb.bit_length() - 1
                        c_bits ^= lsb
                        # Check if any other bit in common is connected to idx
                        if adj[idx] & common & ~(1 << idx):
                            total_edges_in_common += 1
                            # Optimization: if we just need to know if >0, we can break early?
                            # But we want to minimize count.
            k4_count = total_edges_in_common // 6

        # Count Ind4s in T (non-neighbors)
        # T is complement of mask in old vertices.
        t_mask = ((1 << n) - 1) ^ mask
        ind4_count = 0
        t_bits = []
        m = t_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            t_bits.append(idx)
            m ^= lsb
            
        if len(t_bits) >= 4:
            # Ind4 in G is K4 in G'.
            # So count K4s in t_bits using complement adjacencies?
            # Or just check if t_bits contains an independent set of size 4.
            # Same logic: Sum of non-edges in common non-neighborhood?
            # For each pair (a,b) in t_bits:
            #   common_non = ~adj[a] & ~adj[b] & t_mask & ~(1<<a) & ~(1<<b) ?
            #   Actually, we want c,d in t_bits such that c,d not connected to a,b and c,d not connected to each other.
            #   This is equivalent to K4 in complement.
            
            # Let's use the same "edges in common" trick on the complement graph.
            # Comp adj for i: (1<<n)-1 ^ (1<<i) ^ adj[i]
            # But we only care about vertices in t_bits.
            
            # Alternative: 
            # For each pair (a,b) in t_bits:
            #   candidates = t_bits \ {a,b}
            #   valid_c = candidates \ neighbors(a) \ neighbors(b)
            #   valid_d = valid_c \ neighbors(a) \ neighbors(b) \ neighbors(c)
            #   If valid_d exists, we have an Ind4.
            
            # This is O(n^4) worst case.
            # Let's stick to the K4 counting in complement.
            # We need comp_adj restricted to t_bits.
            
            # To save time, let's just check if ANY Ind4 exists.
            # If yes, penalty.
            
            # Iterate pairs in t_bits
            L = len(t_bits)
            found_ind4 = False
            for i in range(L):
                a = t_bits[i]
                for j in range(i + 1, L):
                    b = t_bits[j]
                    # Candidates for c, d must be in t_bits, not a, not b
                    # And not connected to a, not connected to b
                    common_non = t_mask & ~(1 << a) & ~(1 << b) & ~adj[a] & ~adj[b]
                    # Now we need an edge (c,d) in the complement within common_non?
                    # i.e., c,d in common_non such that c,d are NOT connected in G.
                    # This is an edge in G'.
                    
                    # Iterate bits in common_non
                    c_bits = common_non
                    while c_bits:
                        lsb = c_bits & -c_bits
                        idx = lsb.bit_length() - 1
                        c_bits ^= lsb
                        # Check if there is another bit in common_non not connected to idx
                        # non_neigh_idx_in_common = common_non & ~adj[idx] & ~(1<<idx)
                        if common_non & ~adj[idx] & ~(1 << idx):
                            found_ind4 = True
                            break
                    if found_ind4:
                        break
                if found_ind4:
                    break
            
            if found_ind4:
                ind4_count = 100 # Large penalty
            else:
                ind4_count = 0
                
        return k4_count + ind4_count

    best_mask = mask
    best_score = count_violations(mask)
    
    if best_score == 0:
        # Valid mask found
        new_adj = adj + [0]
        new_adj[n] = mask
        for i in range(n):
            if (mask >> i) & 1:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        return new_adj

    # Local search
    for step in range(500):
        if random.random() < 0.1:
            # Restart
            mask = 0
            for _ in range(n):
                if random.random() < 0.5:
                    mask |= (1 << _)
            best_score = count_violations(mask)
            if best_score == 0:
                new_adj = adj + [0]
                new_adj[n] = mask
                for i in range(n):
                    if (mask >> i) & 1:
                        new_adj[i] |= (1 << n)
                        new_adj[n] |= (1 << i)
                return new_adj
        else:
            # Flip a random bit
            bit = random.randint(0, n - 1)
            new_mask = mask ^ (1 << bit)
            new_score = count_violations(new_mask)
            if new_score <= best_score:
                mask = new_mask
                best_score = new_score
                if best_score == 0:
                    new_adj = adj + [0]
                    new_adj[n] = mask
                    for i in range(n):
                        if (mask >> i) & 1:
                            new_adj[i] |= (1 << n)
                            new_adj[n] |= (1 << i)
                    return new_adj

    return None

def construct():
    deadline = time.time() + 280
    best_n = 37
    best_adj = _paley_graph(37)
    
    # Try to extend from 37 to 46
    current_n = 37
    current_adj = best_adj
    
    while current_n < 46:
        if time.time() > deadline:
            break
        
        extended = False
        # Try multiple times to extend
        for attempt in range(20):
            if time.time() > deadline:
                break
            new_adj = _try_extend(current_adj, current_n)
            if new_adj is not None:
                current_adj = new_adj
                current_n += 1
                extended = True
                best_n = current_n
                best_adj = current_adj
                break
        
        if not extended:
            # Failed to extend, break? Or try different random seeds?
            # We can try a few more restarts of the extension process
            # But if we fail 20 times, it might be hard.
            # Let's just break and return best.
            break
            
    # Convert bitmask adj to edge list
    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append((i, j))
            m ^= lsb
            
    return best_n, edges
