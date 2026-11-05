import random
import itertools
from collections import Counter

def construct():
    # We want to maximize n. Try 22, then 21, then 20.
    # The problem states n <= 22.
    
    def check_graph(n, adj):
        """
        Returns the number of regular induced subgraphs of size >= 6.
        adj is a list of integers (bitmasks) where adj[i] has bit j set if (i,j) is an edge.
        """
        violations = 0
        # Iterate over all subsets of size k >= 6
        # To speed up, we can break early if we find too many, but we need the count for optimization.
        # However, for the final check, we just need 0.
        # For optimization, counting is better for gradient.
        
        # Optimization: Only check subsets of size 6, 7, ..., n
        # Precompute combinations? No, generate on the fly.
        
        # Helper to check if a mask is regular
        def is_regular(mask):
            # Get vertices
            verts = [i for i in range(n) if (mask >> i) & 1]
            if len(verts) < 6:
                return False
            
            # Calculate degrees
            degs = []
            for v in verts:
                # Degree of v in induced subgraph mask
                # adj[v] contains neighbors. We need intersection with mask.
                # Note: adj[v] does not contain v itself.
                deg = bin(adj[v] & mask).count('1')
                degs.append(deg)
            
            # Check if all degrees are equal
            if len(set(degs)) == 1:
                return True
            return False

        # Iterate combinations
        # This is the bottleneck.
        # For n=22, sum of C(22, k) for k=6..22 is large.
        # C(22,6)=74613, C(22,7)=170544, C(22,8)=319770 ...
        # Total subsets ~ 1 million?
        # 1 million * 22 ops = 22 million ops.
        # In Python, 22M ops might take 5-10 seconds.
        # This is too slow for many iterations.
        
        # We need a faster check.
        # Idea: Randomized check.
        # Pick a random subset of size >= 6. Check it.
        # If regular, it's a violation.
        # This doesn't give a full count, but we can use it for acceptance criteria in Simulated Annealing?
        # Or we can just try to find ONE violation. If we find one, the graph is invalid.
        # If we run for T seconds and find no violations, we assume it's valid (with some probability).
        # But we need to be sure for the final answer.
        
        # Let's try to optimize the full check.
        # We can precompute the degree of each vertex in the whole graph? No, induced degree depends on subset.
        
        # Alternative: Use NumPy?
        # adj_matrix is n x n.
        # For a subset mask, we can extract the submatrix.
        # But generating masks is the issue.
        
        # Let's limit the search to n=21 first, as n=22 might be impossible or very hard.
        # And use a faster heuristic.
        
        return violations

    # Since the full check is slow, we will use a "fast fail" strategy.
    # We will randomly sample subsets to detect violations quickly.
    # If no violations found in many samples, we do a full check.
    
    def fast_check(n, adj, samples=1000):
        """
        Returns True if a violation is found, False otherwise.
        """
        for _ in range(samples):
            # Pick random size k between 6 and n
            k = random.randint(6, n)
            # Pick random subset of size k
            subset = set(random.sample(range(n), k))
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check if regular
            verts = list(subset)
            degs = []
            for v in verts:
                deg = bin(adj[v] & mask).count('1')
                degs.append(deg)
            
            if len(set(degs)) == 1:
                return True
        return False

    def full_check(n, adj):
        """
        Exhaustive check. Returns True if valid (no regular induced subgraph >= 6).
        """
        # We can prune: if a subset of size k is regular, we don't need to check supersets?
        # No, a superset of a regular graph is not necessarily regular.
        # But we can check sizes 6, 7, ...
        
        # To make it faster, we can use a bitset trick if we were in C, but in Python...
        # Let's just iterate.
        # We will iterate k from 6 to n.
        
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check regularity
                # Optimization: Compute degrees for all v in comb
                # If any degree differs, break early.
                first_deg = -1
                is_reg = True
                for v in comb:
                    deg = bin(adj[v] & mask).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False # Violation found
        
        return True # No violations

    def optimize(n, max_iters=50, time_limit=100):
        """
        Local search to find a valid graph on n vertices.
        """
        import time
        start_time = time.time()
        
        # Initialize with random graph
        # Try different densities
        best_graph = None
        best_cost = float('inf')
        
        for restart in range(5):
            p = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            current_adj = adj[:]
            
            # Initial check
            if not fast_check(n, current_adj, 500):
                 # Likely valid, do full check
                 if full_check(n, current_adj):
                     return current_adj
            
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge (u, v)
                mask_uv = (1 << u) | (1 << v)
                # Check if edge exists
                has_edge = (current_adj[u] & (1 << v)) != 0
                
                # Flip
                if has_edge:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                
                # Check for violations
                # We only need to check subsets containing u or v?
                # Any new regular subgraph must involve the changed edge, so it must contain u or v.
                # Actually, flipping (u,v) only changes degrees of u and v in any induced subgraph.
                # So a subgraph S is regular in the new graph only if:
                # 1. It was regular in the old graph and didn't contain u or v (impossible, it's still regular) -> Wait.
                #    If S doesn't contain u or v, its structure is unchanged. If it was regular before, it is regular now.
                #    So we only need to check if NEW regular subgraphs appear, or if OLD ones disappear.
                #    Since we want to MINIMIZE violations, we care if we introduce new ones.
                #    A subgraph S that becomes regular must contain u or v.
                #    Also, a subgraph S that WAS regular might become non-regular if it contains u or v.
                
                # So we only need to check subsets S that contain u OR v.
                # This is much smaller!
                # Number of subsets containing u is 2^(n-1). Still large.
                # But we can restrict to sizes >= 6.
                
                # Let's just use the fast check for now, it's simpler.
                if not fast_check(n, current_adj, 200):
                    if full_check(n, current_adj):
                        return current_adj
                
                # Accept or reject?
                # Simple hill climbing: keep the change if it doesn't increase "badness".
                # Since we don't have a good cost function, let's just keep random changes if they pass fast_check.
                # If fast_check fails (finds violation), revert?
                # Or just try many random flips and check at the end.
                
                # Better: Keep a history of "good" graphs (passed fast_check).
                # If current passes fast_check, save it.
                
        return None

    # Try n=22
    adj_22 = optimize(22, max_iters=20, time_limit=30)
    if adj_22 is not None:
        edges = []
        for i in range(22):
            for j in range(i + 1, 22):
                if (adj_22[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 22, 'edges': edges}

    # Try n=21
    adj_21 = optimize(21, max_iters=50, time_limit=60)
    if adj_21 is not None:
        edges = []
        for i in range(21):
            for j in range(i + 1, 21):
                if (adj_21[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 21, 'edges': edges}

    # Try n=20 (Known to exist, but let's try to find one quickly)
    adj_20 = optimize(20, max_iters=100, time_limit=20)
    if adj_20 is not None:
        edges = []
        for i in range(20):
            for j in range(i + 1, 20):
                if (adj_20[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 20, 'edges': edges}

    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
