import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_edges_from_adj(adj, n):
        """Convert bitmask adjacency list to list of edges."""
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    def is_regular_subset(vertices, adj):
        """Check if the induced subgraph on 'vertices' is regular."""
        if not vertices:
            return False
        mask = 0
        for v in vertices:
            mask |= (1 << v)
        degs = []
        for v in vertices:
            # Count neighbors of v within the subset
            degs.append((adj[v] & mask).bit_count())
        return len(set(degs)) == 1

    def find_bad_subset(adj, n, max_k=None):
        """
        Find a subset of size k >= 6 that induces a regular graph.
        Returns the subset as a tuple of vertices, or None if none found.
        """
        if max_k is None:
            max_k = n
            
        # Prioritize smaller k as they are more likely to exist and cheaper to check
        for k in range(6, max_k + 1):
            # If k is large, checking all combinations might be slow, 
            # but for n=21, C(21, 11) is 352716, which is fine.
            # C(21, 15) is 116280.
            # We can iterate all combinations for n=21 up to k=10 or 11.
            # For larger k, we might sample. But let's try all first for correctness.
            
            # Optimization: If k > n/2, checking complement might be faster?
            # No, regularity is not complement-invariant in a simple way for degrees.
            
            for subset in itertools.combinations(range(n), k):
                if is_regular_subset(subset, adj):
                    return subset
        return None

    def local_search(n, time_budget):
        """
        Perform local search to find a valid graph on n vertices.
        Returns the adjacency list (bitmasks) if found, else None.
        """
        t0 = time.time()
        
        # Initialize with a random graph
        # Use a fixed seed for reproducibility in the first few attempts, 
        # but vary it for subsequent restarts if needed.
        
        for restart in range(50):
            if time.time() - t0 > time_budget:
                break
            
            random.seed(1234 + restart)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search loop
            stuck_count = 0
            for step in range(20000):
                if time.time() - t0 > time_budget:
                    return None
                
                # Find a bad subset
                # To speed up, we can check random subsets first if the space is huge,
                # but for n=21, full scan of k=6,7,8 is fast.
                # Let's do a full scan for k=6..10 and random sampling for k>10.
                
                bad_subset = None
                
                # Check k=6 to 10 exhaustively (fast)
                for k in range(6, min(11, n + 1)):
                    for subset in itertools.combinations(range(n), k):
                        if is_regular_subset(subset, adj):
                            bad_subset = subset
                            break
                    if bad_subset:
                        break
                
                # If not found in small k, check larger k with sampling or exhaustive if small enough
                if not bad_subset and n >= 11:
                    # Check k=11..n
                    # For n=21, C(21, 11) is ~350k. This is still fast.
                    # Let's just check all k from 11 to n.
                    for k in range(11, n + 1):
                        # If the number of combinations is too large, sample
                        # C(21, 15) is ~116k. C(21, 21) is 1.
                        # It's safe to check all for n=21.
                        for subset in itertools.combinations(range(n), k):
                            if is_regular_subset(subset, adj):
                                bad_subset = subset
                                break
                        if bad_subset:
                            break
                
                if bad_subset is None:
                    return adj  # Valid graph found
                
                # Flip a random edge within the bad subset
                s = list(bad_subset)
                i, j = random.sample(s, 2)
                # Flip edge (i, j)
                if adj[i] & (1 << j):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                # Occasionally, if we get stuck (same bad subset keeps appearing?), 
                # we might want to make a random move elsewhere.
                # But flipping within the bad subset is the standard repair.
        
        return None

    # Try n=21 first, then 20
    for n in [21, 20]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time based on n
        # n=21 is harder, give it more time initially
        budget = remaining if n == 21 else remaining * 0.5
        
        result_adj = local_search(n, budget)
        
        if result_adj is not None:
            return {'n': n, 'edges': get_edges_from_adj(result_adj, n)}
    
    # Fallback: Return a known valid graph for n=20 if we didn't find one?
    # Or just return n=19 or smaller to be safe?
    # The task says "Maximize n". If we fail to find 21 or 20, we should return the best we can.
    # Let's try one more time for n=20 with a different strategy or just return a valid small graph.
    # Actually, the local search for n=20 should succeed quickly if it succeeds at all.
    # If it fails, it's likely due to time limit or local minima.
    
    # As a last resort, return a valid graph for n=19 (Paley graph)
    # Paley graph of order 19 is strongly regular (19, 9, 4, 4).
    # Does it have regular induced subgraphs of size >= 6?
    # We should verify this.
    
    def get_paley_adj(p):
        adj = [0] * p
        qr = set((i * i) % p for i in range(p))
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in qr:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    for p in [19, 17, 13]:
        adj = get_paley_adj(p)
        # Quick check
        if find_bad_subset(adj, p) is None:
            return {'n': p, 'edges': get_edges_from_adj(adj, p)}
            
    # If all else fails, return a small valid graph
    # n=5, no edges is not regular? 
    # Induced subgraph on 5 vertices (empty graph) is 0-regular. 
    # But the constraint is "6 or more". So n=5 is always valid.
    return {'n': 5, 'edges': []}

if __name__ == "__main__":
    # Test locally
    res = construct()
    print(f"Found graph with n={res['n']}")
