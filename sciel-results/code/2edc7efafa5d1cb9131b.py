import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_valid(adj, n):
        """
        Check if the graph defined by adj masks on n vertices
        has no regular induced subgraph of size >= 6.
        """
        for k in range(6, n + 1):
            mask = (1 << k) - 1
            limit = 1 << n
            
            while mask < limit:
                m = mask
                first_vertex = True
                d0 = 0
                is_regular = True
                
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    d = (adj[v] & mask).bit_count()
                    
                    if first_vertex:
                        d0 = d
                        first_vertex = False
                    else:
                        if d != d0:
                            is_regular = False
                            break
                    m ^= lsb
                
                if is_regular:
                    return False
                
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r

        return True

    def generate_random_graph(n):
        edges = []
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    edges.append([i, j])
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return edges, adj

    def count_bad_subsets(adj, n):
        """
        Count the number of subsets of size >= 6 that induce a regular subgraph.
        Returns the count.
        """
        bad_count = 0
        # We only need to check if ANY subset is regular.
        # For local search, we want the total count to guide us.
        # However, checking all subsets of size >= 6 is expensive.
        # But we can optimize: if a subset of size 6 is regular, any superset is not necessarily regular.
        # Actually, we just need to know if the graph is valid (count == 0).
        # For local search, we need a metric.
        # Let's sum the "regularity score" of all subsets of size 6.
        # A subset is regular if all degrees are equal.
        # Score = 1 if regular, 0 otherwise.
        
        # To be fast, we only check subsets of size 6.
        # If there are no regular induced subgraphs of size 6, there might still be some of size 7+.
        # But usually, size 6 is the bottleneck.
        # If we eliminate all size 6 regular subgraphs, we are very likely valid.
        # We will verify with full check at the end.
        
        for mask in itertools.combinations(range(n), 6):
            m = 0
            for v in mask:
                m |= (1 << v)
            
            d0 = -1
            is_regular = True
            for v in mask:
                d = (adj[v] & m).bit_count()
                if d0 == -1:
                    d0 = d
                elif d != d0:
                    is_regular = False
                    break
            
            if is_regular:
                bad_count += 1
                
        return bad_count

    def local_search(n, max_iterations=1000):
        """
        Perform local search to minimize the number of regular induced subgraphs of size 6.
        """
        edges, adj = generate_random_graph(n)
        
        # Precompute masks for all subsets of size 6
        # And for each edge, which subsets it affects
        # This is complex to precompute dynamically for each n.
        # Instead, we will just recalculate the delta for a flip.
        
        current_bad = count_bad_subsets(adj, n)
        
        if current_bad == 0:
            if check_valid(adj, n):
                return edges, adj
            
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            best_delta = 0
            best_edge = None
            best_adj = None
            best_edges = None
            
            # Try flipping each edge
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if edge (i, j) is present
                    # Flip it
                    new_adj = adj[:]
                    if adj[i] & (1 << j):
                        new_adj[i] &= ~(1 << j)
                        new_adj[j] &= ~(1 << i)
                    else:
                        new_adj[i] |= (1 << j)
                        new_adj[j] |= (1 << i)
                    
                    # Calculate new bad count
                    # Optimization: Only subsets containing i and j can change.
                    # Subsets of size 6 containing i and j: choose 4 from remaining n-2.
                    # This is still many.
                    # Let's just recalculate count_bad_subsets for simplicity in this iteration,
                    # but that's O(N^6) which is too slow for 1000 iterations.
                    
                    # We need a faster delta calculation.
                    # Let's skip this complex optimization and just do random restarts with a faster check?
                    # No, local search is the structural change.
                    # Let's optimize count_bad_subsets to be faster or limit iterations.
                    
                    # Actually, for n=22, C(22,6) is ~75k.
                    # 75k * 231 edges * 1000 iters = 17 billion ops. Too slow.
                    
                    # Alternative: Just do random restarts but with a better initial guess?
                    # Or use a smaller n?
                    # Let's try n=20 first to ensure we beat 2.6 (which is likely 20).
                    # If we find 20 fast, we try 21.
                    
                    break
                break
            break
            
        return None

    # The local search above is too slow to implement correctly in this short time without profiling.
    # Let's stick to a smarter random search: 
    # 1. Try n=22 with a few hundred attempts.
    # 2. Try n=21 with a few hundred attempts.
    # 3. Try n=20 with many attempts.
    
    # But the current program ALREADY does this and scores 2.6.
    # This implies it finds n=20.
    # To improve, we must find n=21 or 22.
    
    # Let's try a specific deterministic construction for n=20 first to guarantee 2.6, 
    # then spend ALL remaining time trying n=21 and 22.
    
    # Known 20-vertex graph?
    # We can just use the random search for n=20 but with a seed that we know works? 
    # We don't know a specific seed.
    
    # Let's try to improve the random search efficiency.
    # 1. Use a fixed seed for reproducibility? No, random is better for exploration.
    # 2. Use a better probability?
    
    # Let's try a hybrid: 
    # Start with n=22. 
    # If not found in 10s, try n=21.
    # If not found in 10s, try n=20.
    
    # The current code does this but with 50 attempts per p.
    # Let's increase the attempts for n=22 and 21.
    
    best_result = None
    
    for n in [22, 21, 20]:
        if time.time() > deadline - 10.0:
            break
            
        # For n=22, 21, use more attempts
        attempts = 500 if n > 20 else 2000
        
        found = False
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            edges, adj = generate_random_graph(n)
            
            if check_valid(adj, n):
                best_result = {'n': n, 'edges': edges}
                found = True
                break
        
        if found:
            break
            
    if best_result:
        return best_result
        
    # Fallback
    return {'n': 5, 'edges': []}
