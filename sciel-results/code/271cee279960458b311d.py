import time
import random
from itertools import permutations, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill

    n = 16

    # Precompute Q3 structure
    # Even vertices: 0, 1, 2, 3 (indices in the 4-set)
    # Odd vertices: 0, 1, 2, 3
    # Edges from Even i to Odd j:
    # E0: O0, O1, O2
    # E1: O0, O1, O3
    # E2: O0, O2, O3
    # E3: O1, O2, O3
    
    # For a permutation pi of A (where pi[i] is the index in A assigned to E_i),
    # we need to find b_j in S_j.
    # S_j is the intersection of neighbors of A[k] for all k such that E_{pi[k]} is connected to O_j.
    
    # Let's precompute the sets of Even indices connected to each Odd index.
    # O0 connected to E0, E1, E2
    # O1 connected to E0, E1, E3
    # O2 connected to E0, E2, E3
    # O3 connected to E1, E2, E3
    
    odd_to_even = [
        [0, 1, 2], # O0
        [0, 1, 3], # O1
        [0, 2, 3], # O2
        [1, 2, 3]  # O3
    ]
    
    # Precompute all 4-subsets of 16
    all_4_subsets = list(combinations(range(n), 4))
    
    # Precompute all 24 permutations of 0,1,2,3
    all_perms = list(permutations(range(4)))
    
    def has_q3(adj):
        """
        adj: list of 16 ints, bit j set in adj[i] means edge (i,j) exists.
        Returns True if Q3 subgraph exists.
        """
        for A in all_4_subsets:
            # A is a tuple of 4 vertices
            # We need to check all 24 ways to assign A to E0, E1, E2, E3
            # Let pi be a permutation of (0,1,2,3).
            # pi[i] = index in A that maps to E_i.
            # So A[pi[i]] is the vertex mapped to E_i.
            
            for pi in all_perms:
                # pi is a tuple like (0, 1, 2, 3)
                # A[pi[0]] -> E0
                # A[pi[1]] -> E1
                # A[pi[2]] -> E2
                # A[pi[3]] -> E3
                
                # Calculate candidate sets for O0, O1, O2, O3
                # S_j = intersection of N(A[pi[k]]) for all k in odd_to_even[j]
                # Note: N(v) is adj[v]
                
                # S0 (for O0): connected to E0, E1, E2
                # So intersection of N(A[pi[0]]), N(A[pi[1]]), N(A[pi[2]])
                s0 = adj[A[pi[0]]] & adj[A[pi[1]]] & adj[A[pi[2]]]
                
                # S1 (for O1): connected to E0, E1, E3
                s1 = adj[A[pi[0]]] & adj[A[pi[1]]] & adj[A[pi[3]]]
                
                # S2 (for O2): connected to E0, E2, E3
                s2 = adj[A[pi[0]]] & adj[A[pi[2]]] & adj[A[pi[3]]]
                
                # S3 (for O3): connected to E1, E2, E3
                s3 = adj[A[pi[1]]] & adj[A[pi[2]]] & adj[A[pi[3]]]
                
                # Remove vertices in A from candidates
                mask_A = 0
                for v in A:
                    mask_A |= (1 << v)
                
                s0 &= ~mask_A
                s1 &= ~mask_A
                s2 &= ~mask_A
                s3 &= ~mask_A
                
                # Check if we can pick distinct b0 in S0, b1 in S1, b2 in S2, b3 in S3
                # Since S0..S3 are bitmasks, we can iterate.
                # There are at most 16 bits, so this is fast.
                
                # Try to find b0
                if not s0: continue
                # Iterate through bits in s0
                s0_copy = s0
                while s0_copy:
                    b0 = (s0_copy & -s0_copy).bit_length() - 1
                    s0_copy &= s0_copy - 1
                    
                    # Try to find b1 in S1, distinct from b0
                    if not s1: continue
                    s1_copy = s1 & ~(1 << b0)
                    while s1_copy:
                        b1 = (s1_copy & -s1_copy).bit_length() - 1
                        s1_copy &= s1_copy - 1
                        
                        # Try to find b2 in S2, distinct from b0, b1
                        if not s2: continue
                        s2_copy = s2 & ~(1 << b0) & ~(1 << b1)
                        while s2_copy:
                            b2 = (s2_copy & -s2_copy).bit_length() - 1
                            s2_copy &= s2_copy - 1
                            
                            # Try to find b3 in S3, distinct from b0, b1, b2
                            if not s3: continue
                            s3_copy = s3 & ~(1 << b0) & ~(1 << b1) & ~(1 << b2)
                            if s3_copy:
                                return True
        
        return False

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible = len(all_edges)  # 120

    best_edges_count = 0
    best_adj = None

    def greedy_add(adj, edge_order, deadline):
        """Add edges in given order, skipping those that create Q3."""
        adj = adj[:]
        count = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if has_q3(adj):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                count += 1
        return adj, count

    def local_search(adj, count, deadline):
        """Try to add more edges by trying all missing edges."""
        adj = adj[:]
        count = count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if has_q3(adj):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    count += 1
                    improved = True
        return adj, count

    trials = 0
    while time.time() < deadline - 20:
        trials += 1
        random.seed(trials * 12345 + 67890)

        edge_order = all_edges[:]
        random.shuffle(edge_order)

        adj = [0] * n
        adj, count = greedy_add(adj, edge_order, deadline - 10)

        if count > best_edges_count:
            best_edges_count = count
            best_adj = adj[:]

        if best_adj is not None and time.time() < deadline - 5:
            adj_ls, count_ls = local_search(best_adj, best_edges_count, deadline - 2)
            if count_ls > best_edges_count:
                best_edges_count = count_ls
                best_adj = adj_ls

        if trials % 3 == 0 and time.time() < deadline - 15:
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                if not has_q3(adj):
                    break
                for u, v in all_edges:
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        count -= 1
                        if not has_q3(adj):
                            break
                else:
                    break

            if count > best_edges_count:
                best_edges_count = count
                best_adj = adj[:]

        if trials > 100:
            break

    if best_adj is not None and time.time() < deadline - 1:
        adj_ls, count_ls = local_search(best_adj, best_edges_count, deadline - 0.5)
        if count_ls > best_edges_count:
            best_edges_count = count_ls
            best_adj = adj_ls

    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges

construct()
