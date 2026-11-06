import time
import random
from itertools import combinations

def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Maximizes n.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper: Check if graph is C4-free
    # adj[i] is a set of neighbors
    def is_c4_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) > 1:
                    return False
        return True

    # Helper: Max Independent Set (exact)
    def max_independent_set(adj, n):
        # Bitset representation
        # n <= 43, so we can use Python integers as bitsets
        # adj_bits[i] is an integer with bits set for neighbors of i
        
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_bits[i] = mask
        
        # Branch and bound
        # State: available vertices (bitmask), current size
        # We want max size
        
        # Pre-compute: for each vertex, its neighbors in the current set
        # This is tricky with dynamic sets.
        # Standard B&B:
        # 1. Pick a vertex v.
        # 2. Include v: remove v and its neighbors.
        # 3. Exclude v: remove v.
        
        # Optimization: Upper bound = popcount(available)
        # Better bound: Greedy coloring bound?
        # For n=43, simple B&B with good ordering might work.
        
        best = 0
        
        def popcount(x):
            return bin(x).count('1')
            
        def solve(avail, size):
            nonlocal best
            if size + popcount(avail) <= best:
                return
            if avail == 0:
                if size > best:
                    best = size
                return
                
            # Pick a vertex with high degree in avail to prune?
            # Or just pick the lowest bit.
            # Let's pick the vertex with max degree in avail for better pruning.
            # This is expensive. Let's try lowest bit first.
            
            # Find a vertex
            # v = (avail & -avail).bit_length() - 1
            # This picks the lowest index.
            
            # Better: Pick v that maximizes neighbors in avail?
            # Let's stick to simple for now, n is small.
            
            # To improve, we can sort vertices by degree in avail.
            # But that's O(n^2) per node.
            
            # Let's use a simple heuristic: pick the vertex with the most neighbors in avail.
            # If tie, lowest index.
            
            max_deg = -1
            v = -1
            temp = avail
            while temp:
                lsb = temp & -temp
                i = lsb.bit_length() - 1
                # Count neighbors of i in avail
                deg = popcount(adj_bits[i] & avail)
                if deg > max_deg:
                    max_deg = deg
                    v = i
                temp ^= lsb
                
            if v == -1:
                return
                
            # Branch 1: Include v
            # New avail: remove v and its neighbors
            new_avail_1 = avail & ~(adj_bits[v] | (1 << v))
            solve(new_avail_1, size + 1)
            
            # Branch 2: Exclude v
            new_avail_2 = avail & ~(1 << v)
            solve(new_avail_2, size)
            
        solve((1 << n) - 1, 0)
        return best

    # Initialize with 10 disjoint triangles
    n = 30
    adj = [set() for _ in range(n)]
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a].add(b); adj[b].add(a)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(c); adj[c].add(b)
        
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Check initial
    if max_independent_set(adj, n) < 11:
        best_n = n
        best_adj = [set(s) for s in adj]

    # Heuristic Search
    while time.time() < deadline:
        # Compute H: graph where (u,v) in H if N(u) & N(v) != empty
        # We want an independent set S in H.
        # This S will be the neighbors of the new vertex.
        
        # Build H adjacency
        h_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & adj[j]:
                    h_adj[i].add(j)
                    h_adj[j].add(i)
                    
        # Find Max Independent Set in H
        # H is small (n <= 43).
        # We can use the same MIS solver, but on H.
        # Note: H is a simple graph.
        
        # Convert H to adj list for MIS solver
        # The MIS solver takes adj sets.
        # We need to be careful: the MIS solver uses adj_bits which are built from adj.
        # So we can just pass h_adj to max_independent_set.
        
        # But max_independent_set returns the SIZE. We need the SET.
        # Let's write a helper to get the set.
        
        def get_max_independent_set(adj_h, n_h):
            adj_bits = [0] * n_h
            for i in range(n_h):
                mask = 0
                for j in adj_h[i]:
                    mask |= (1 << j)
                adj_bits[i] = mask
                
            best_set = 0
            best_size = 0
            
            def popcount(x):
                return bin(x).count('1')
                
            def solve(avail, curr_set, size):
                nonlocal best_set, best_size
                if size + popcount(avail) <= best_size:
                    return
                if avail == 0:
                    if size > best_size:
                        best_size = size
                        best_set = curr_set
                    return
                    
                # Pick vertex with max degree in avail
                max_deg = -1
                v = -1
                temp = avail
                while temp:
                    lsb = temp & -temp
                    i = lsb.bit_length() - 1
                    deg = popcount(adj_bits[i] & avail)
                    if deg > max_deg:
                        max_deg = deg
                        v = i
                    temp ^= lsb
                    
                if v == -1:
                    return
                
                # Include v
                new_avail_1 = avail & ~(adj_bits[v] | (1 << v))
                solve(new_avail_1, curr_set | (1 << v), size + 1)
                
                # Exclude v
                new_avail_2 = avail & ~(1 << v)
                solve(new_avail_2, curr_set, size)
                
            solve((1 << n_h) - 1, 0, 0)
            
            # Convert bitmask to set of vertices
            S = set()
            mask = best_set
            while mask:
                lsb = mask & -mask
                i = lsb.bit_length() - 1
                S.add(i)
                mask ^= lsb
            return S

        S = get_max_independent_set(h_adj, n)
        
        if not S:
            break
            
        # Add new vertex n connected to S
        new_n = n + 1
        if new_n > 43:
            break
            
        new_adj = [set(s) for s in adj]
        new_adj.append(set(S))
        for u in S:
            new_adj[u].add(new_n - 1)
            
        # Check C4
        if not is_c4_free(new_adj, new_n):
            # This should not happen if S is independent in H
            # But just in case
            break
            
        # Check Alpha
        alpha = max_independent_set(new_adj, new_n)
        
        if alpha < 11:
            # Valid improvement
            n = new_n
            adj = new_adj
            best_n = n
            best_adj = [set(s) for s in adj]
            
            # Try to add more edges to reduce alpha?
            # The strategy: Add clique on U = V \ (S U {v})
            # This is safe for C4 if U vertices don't share neighbors.
            # Let's check if we can add a clique on U.
            U = set(range(n)) - S - {n - 1}
            
            # Check if U is "C4-safe" for clique addition
            # i.e., for all u1, u2 in U, N(u1) & N(u2) == empty
            safe = True
            U_list = list(U)
            for i in range(len(U_list)):
                for j in range(i + 1, len(U_list)):
                    u1 = U_list[i]
                    u2 = U_list[j]
                    if adj[u1] & adj[u2]:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe and len(U) > 1:
                # Add clique on U
                for i in range(len(U_list)):
                    for j in range(i + 1, len(U_list)):
                        u1 = U_list[i]
                        u2 = U_list[j]
                        adj[u1].add(u2)
                        adj[u2].add(u1)
                
                # Re-check alpha
                alpha = max_independent_set(adj, n)
                if alpha < 11:
                    best_n = n
                    best_adj = [set(s) for s in adj]
                else:
                    # Revert clique if it didn't help or broke something?
                    # It shouldn't break C4.
                    # If alpha >= 11, we might have broken the condition?
                    # No, adding edges reduces alpha.
                    # So if alpha was < 11, it stays < 11.
                    pass
        else:
            # Alpha >= 11
            # Try to fix by adding edges?
            # Or just skip this vertex addition?
            # We can try to add the vertex but with a smaller S?
            # For now, let's just break or try next iteration.
            # Actually, if we can't fix it, we should stop adding vertices
            # to avoid getting stuck in a bad state.
            # But we can try to modify the graph.
            # Let's just break the loop to be safe, as we have a valid best_adj.
            break

    # Return the best graph found
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
