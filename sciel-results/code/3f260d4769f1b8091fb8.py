import random
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Leave buffer for evaluation overhead

    def has_k4(adj, n):
        """Check if graph has K4 using bitmasks for speed."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # Check if any pair in common forms a K4
                    # We need two vertices k, l in common such that k-l edge exists
                    # Iterate over vertices in common
                    mask = common
                    while mask:
                        lsb = mask & -mask
                        k = lsb.bit_length() - 1
                        mask ^= lsb
                        # Check if k has any neighbor in common with index > k to avoid duplicates?
                        # Actually, if k and l are both in common, and edge k-l exists, then i,j,k,l is K4.
                        # We just need to check if adj[k] & common has any bit set for l > k?
                        # Or simply: if (adj[k] & common) != 0, and that bit is not k itself (which it isn't in common).
                        # Wait, adj[k] & common includes k? No, no self-loops.
                        # So if adj[k] & common is non-zero, there is an edge from k to some l in common.
                        # Since i,j are connected to k and l, and i-j connected, and k-l connected -> K4.
                        if adj[k] & common:
                            return True
        return False

    def has_ind_6(adj, n):
        """Check if graph has independent set of size 6 using bitmasks."""
        # Independent set in G is clique in complement.
        # Build complement adj
        comp_adj = []
        for i in range(n):
            # All bits except self and those in adj[i]
            # Full mask for n vertices
            full_mask = (1 << n) - 1
            # Remove self and neighbors
            comp_mask = full_mask & ~(1 << i) & ~adj[i]
            comp_adj.append(comp_mask)
        
        # Find clique of size 6 in comp_adj
        # Backtracking with bitmasks
        def backtrack(candidates, size):
            if size == 0:
                return True
            if not candidates:
                return False
            if candidates.bit_count() < size:
                return False
            
            # Heuristic: pick vertex with max degree in candidate subgraph?
            # For simplicity and speed, just pick first available.
            # To improve, order by degree.
            # Let's try picking a vertex.
            # To speed up, we can try to include the first vertex.
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Option 1: Include v
            new_candidates = candidates & comp_adj[v]
            new_candidates &= ~v_bit # Remove v
            if backtrack(new_candidates, size - 1):
                return True
            
            # Option 2: Exclude v
            if backtrack(candidates & ~v_bit, size):
                return True
            
            return False

        return backtrack((1 << n) - 1, 6)

    def build_paley_graph(n):
        """
        Construct a graph on n vertices.
        For n=35, we can use a known construction or a specific Ramsey graph.
        The 35-vertex Ramsey graph R(4,6) > 35 is often constructed using the icosahedron complement or similar.
        However, a simpler approach for n=35 is to use the Paley graph if 35 is a prime power? 35 is not.
        Let's use a known 35-vertex graph: the complement of the icosahedral graph? No, Icosahedron has 12 vertices.
        
        Actually, the standard 35-vertex graph for R(4,6) is derived from the 35 vertices of the icosahedron? No.
        It is often the graph where vertices are the 35 edges of the icosahedron?
        
        Let's use a random construction with local search if the specific one is not known.
        But we can try n=20, 25, etc. with random + local search.
        
        Wait, the task says "Exoo's 35-vertex graph is the published record".
        I will implement a local search that tries to find a valid graph for n=35.
        If it fails, I'll fall back to smaller n.
        """
        pass

    def local_search(n, max_time):
        """
        Local search to find a graph on n vertices with no K4 and no Ind 6.
        """
        best_score = float('inf')
        best_adj = None
        
        # Start with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        def score(adj):
            # Score = number of K4s + number of Ind 6s (approx or exact)
            # Counting Ind 6s is hard.
            # Let's use a penalty function.
            # If has K4, return high penalty.
            # If has Ind 6, return high penalty.
            # We need to find a graph that avoids both.
            
            k4_count = 0
            # Count K4s
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        common = adj[i] & adj[j]
                        # Count edges in common
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            k = lsb.bit_length() - 1
                            mask ^= lsb
                            # Count neighbors of k in common
                            k_common = adj[k] & common
                            # Each edge counted twice?
                            # If we just sum, we get 2 * (number of K4s containing edge i-j)?
                            # Actually, for each K4, it has 6 edges.
                            # This counting is messy.
                            # Let's just check existence for validity, and use a heuristic score for optimization.
                            pass
            
            # Heuristic Score:
            # We want to minimize the maximum of (K4 count, Ind6 count).
            # Let's just return 0 if valid, 1 if invalid.
            if has_k4(adj, n):
                return 1
            if has_ind_6(adj, n):
                return 1
            return 0

        # Simulated Annealing / Hill Climbing
        current_adj = [a for a in adj]
        current_score = score(current_adj)
        
        for _ in range(5000):
            if time.time() - start_time > max_time:
                break
            
            # Random move: flip a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Flip edge i-j
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            new_score = score(current_adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score == 0:
                    break
            else:
                # Revert
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
        
        if current_score == 0:
            return current_adj
        return None

    # Try n from 35 down to 1
    for n in range(35, 0, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try local search for this n
        adj = local_search(n, time_limit - 5)
        
        if adj is not None:
            # Verify
            if not has_k4(adj, n) and not has_ind_6(adj, n):
                edges = []
                for i in range(n):
                    mask = adj[i]
                    for j in range(i + 1, n):
                        if mask & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # If local search failed, try a known construction for small n?
        # For n <= 20, we can brute force or use known Ramsey graphs.
        # But local search should work for small n.
        
    # Fallback: return a trivial valid graph (e.g., n=1)
    return {'n': 1, 'edges': []}
