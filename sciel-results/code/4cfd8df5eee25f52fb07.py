import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    def is_c4_free(adj):
        """Check C4-free: no two distinct vertices share 2+ common neighbors."""
        n = adj.shape[0]
        if n <= 1:
            return True
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return not np.any(A2 >= 2)

    def get_min_degree(adj):
        if adj.size == 0:
            return 0
        return int(np.min(adj.sum(axis=1)))

    def build_hermitian_graph(q):
        """
        Construct the graph of the Hermitian variety H(3, q) over GF(q^2).
        The vertices are the points of the projective plane PG(2, q^2) that are 
        absolutely degenerate with respect to the Hermitian form.
        
        Actually, a simpler standard C4-free graph is the one defined on GF(q) 
        using the trace form for q=p^2, but that has low degree.
        
        A better known construction for high minimum degree C4-free graphs:
        Use the graph associated with the Hermitian polarity on PG(2, q^2).
        The vertices are the q^3+q^2+1 points of the projective plane.
        Two points are adjacent if they are orthogonal w.r.t the Hermitian form.
        This graph is C4-free.
        
        For q=7, q^2=49. Number of vertices = 49^2 + 49 + 1 = 2401 + 50 = 2451.
        This is way too many vertices for n<=47.
        
        Let's look at smaller constructions.
        The Erdos-Renyi construction for C4-free graphs:
        Let q be a prime power. Let V = GF(q) x GF(q).
        Edge (a,b) ~ (c,d) iff (a-c)^2 - (b-d) is a square in GF(q) (or similar).
        This is not always C4-free.
        
        The standard "Polarity Graph" of the hyperbolic quadric in PG(3, q):
        Vertices: q^2 + 1.
        Degree: q.
        For q=7, n=50. Too big.
        For q=5, n=26. Too small.
        
        We need n=46 or 47.
        
        Let's try a different algebraic construction:
        The graph defined by the equation x*y = 1 in GF(q) is a perfect matching (C4-free, degree 1).
        
        How about the "Klein's graph" or similar?
        
        Actually, for n=46, target degree is 7.
        A 7-regular C4-free graph on 46 vertices is the target.
        
        Let's use a random regular graph generation with C4-removal.
        But the previous program failed to find it efficiently.
        
        New Strategy:
        1. Start with a random 7-regular graph on 46 vertices.
        2. Use a local search to eliminate C4s.
           - Find a C4.
           - Swap edges to break the C4 while maintaining 7-regularity.
           - This is a "2-switch" or edge swap.
        
        Why this is different:
        The previous program used a greedy addition which gets stuck.
        This uses a "regular graph + repair" approach which is often more robust 
        for fixed degree constraints.
        
        Implementation:
        1. Generate a random 7-regular graph on n=46 vertices.
        2. While there is a C4:
           a. Find a C4 (u, v, w, x, u).
           b. The edges are (u,v), (v,w), (w,x), (x,u).
           c. We need to swap edges to break this C4.
           d. A valid swap: Remove (u,v) and (w,x), add (u,w) and (v,x).
           e. Check if (u,w) and (v,x) create new C4s or violate regularity (they don't violate regularity).
           f. If the swap reduces the total number of C4s (or just breaks this one without creating another immediately), accept.
        3. If stuck, restart.
        
        To make it faster:
        - Use numpy for C4 detection.
        - Use a heuristic for choosing which C4 to break.
        """
        pass

    def generate_random_regular(n, k):
        """
        Generate a random k-regular graph on n vertices.
        Uses the configuration model with rejection or repair.
        """
        if n * k % 2 != 0:
            return None
        
        # Stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Random permutation of stubs
        random.shuffle(stubs)
        
        # Pair up stubs
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, swap with another stub to fix
                # Find a j such that stubs[j] != u
                # Simple repair: swap stubs[i+1] with a random other stub
                # This might create multi-edges, which we will handle later or reject.
                # For simplicity, let's just reject and restart if self-loop or multi-edge.
                return None
            if adj[u, v] == 1:
                # Multi-edge
                return None
            adj[u, v] = 1
            adj[v, u] = 1
        
        return adj

    def find_c4s(adj):
        """
        Find all C4s in the graph.
        Returns a list of tuples (u, v, w, x) representing the C4.
        """
        n = adj.shape[0]
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        
        c4s = []
        # A C4 exists if A2[u,v] >= 2.
        # The common neighbors are the witnesses.
        # For each pair (u,v) with common neighbors c1, c2, the C4 is u-c1-v-c2-u.
        # We want to return unique C4s.
        
        for u in range(n):
            for v in range(u + 1, n):
                if A2[u, v] >= 2:
                    # Find common neighbors
                    common = np.where(adj[u] & adj[v])[0]
                    # Pick two common neighbors
                    # This specific pair (u,v) and two common neighbors form a C4.
                    # To avoid duplicates, we can just return the pair (u,v) and the common neighbors.
                    # For the swap, we need the 4 vertices.
                    # Let's just return the list of (u, v, common_list)
                    c4s.append((u, v, common))
        return c4s

    def count_c4s(adj):
        """Count the number of C4s."""
        n = adj.shape[0]
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        # Each C4 is counted 4 times in the sum of A2 (each edge of the C4 is a pair with 2 common neighbors? No.)
        # Actually, sum(A2) counts the number of pairs (u,v) and a common neighbor w.
        # A C4 u-v-w-x-u contributes 4 to this sum (pairs (u,v) with w, (v,w) with u, etc? No.)
        # Let's just count the number of entries >= 2.
        # Each C4 corresponds to a pair (u,v) having at least 2 common neighbors.
        # If a pair has k common neighbors, it contributes C(k,2) C4s.
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = A2[i, j]
                if k >= 2:
                    count += k * (k - 1) // 2
        return count

    def local_search_regular_c4_free(n, k, max_iters=5000, restarts=100):
        """
        Start with a random k-regular graph and use local search to remove C4s.
        """
        for r in range(restarts):
            if time.time() - start_time > deadline - 5:
                break
            
            # 1. Generate random k-regular graph
            adj = None
            for _ in range(20):
                adj = generate_random_regular(n, k)
                if adj is not None:
                    break
            
            if adj is None:
                continue
            
            current_c4s = count_c4s(adj)
            
            if current_c4s == 0:
                # Verify min degree
                if np.min(adj.sum(axis=1)) == k:
                    return adj
            
            # 2. Local search
            for it in range(max_iters):
                if time.time() - start_time > deadline - 5:
                    break
                
                if current_c4s == 0:
                    return adj
                
                # Find a C4 to break
                # Get a random pair with common neighbors
                A = adj.astype(np.int32)
                A2 = A @ A
                np.fill_diagonal(A2, 0)
                
                # Find indices where A2 >= 2
                mask = A2 >= 2
                if not np.any(mask):
                    break # Should not happen if current_c4s > 0
                
                pairs = np.argwhere(mask)
                if len(pairs) == 0:
                    break
                    
                # Pick a random problematic pair
                idx = random.randint(0, len(pairs) - 1)
                u, v = pairs[idx]
                
                # Get common neighbors
                common = np.where(adj[u] & adj[v])[0]
                
                if len(common) < 2:
                    continue
                
                # Pick two common neighbors c1, c2
                c1, c2 = random.sample(common.tolist(), 2)
                
                # The C4 is u-c1-v-c2-u (or u-c2-v-c1-u)
                # Edges: (u,c1), (c1,v), (v,c2), (c2,u)
                
                # Attempt a swap to break this C4.
                # We want to remove two edges and add two edges.
                # Option 1: Remove (u,c1) and (v,c2), add (u,v) and (c1,c2).
                #   This requires u and v to not be adjacent (they are, if A2[u,v]>=2? No, A2[u,v] counts common neighbors. u and v might be adjacent or not).
                #   If u and v are adjacent, adding (u,v) creates a multi-edge.
                #   If c1 and c2 are adjacent, adding (c1,c2) creates a multi-edge.
                
                # Let's try all 3 possible swaps for the C4 u-c1-v-c2-u:
                # The cycle is u-c1-v-c2-u.
                # Swap 1: Remove (u,c1) and (v,c2). Add (u,v) and (c1,c2).
                # Swap 2: Remove (c1,v) and (c2,u). Add (c1,c2) and (u,v). (Same as above, just different order)
                # Actually, the edges of the C4 are e1=(u,c1), e2=(c1,v), e3=(v,c2), e4=(c2,u).
                # We can swap non-adjacent edges.
                # Pair 1: e1 and e3. Remove (u,c1) and (v,c2). Add (u,v) and (c1,c2).
                # Pair 2: e2 and e4. Remove (c1,v) and (c2,u). Add (c1,c2) and (u,v). (Same edges added)
                # Wait, e1=(u,c1) and e3=(v,c2).
                # New edges: (u,v) and (c1,c2).
                # Check validity:
                # 1. (u,v) not in adj (unless u==v, which is false).
                # 2. (c1,c2) not in adj.
                # 3. No self loops (u!=v, c1!=c2).
                
                if adj[u, v] == 0 and adj[c1, c2] == 0:
                    # Perform swap
                    adj[u, c1] = 0
                    adj[c1, u] = 0
                    adj[v, c2] = 0
                    adj[c2, v] = 0
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[c1, c2] = 1
                    adj[c2, c1] = 1
                    
                    new_c4s = count_c4s(adj)
                    # Accept if it reduces C4 count, or with some probability if equal (to escape local minima)
                    if new_c4s < current_c4s or (new_c4s == current_c4s and random.random() < 0.1):
                        current_c4s = new_c4s
                    else:
                        # Revert
                        adj[u, c1] = 1
                        adj[c1, u] = 1
                        adj[v, c2] = 1
                        adj[c2, v] = 1
                        adj[u, v] = 0
                        adj[v, u] = 0
                        adj[c1, c2] = 0
                        adj[c2, c1] = 0
                else:
                    # Try another swap? 
                    # There are only 2 ways to pair non-adjacent edges in a 4-cycle.
                    # We tried the one that adds (u,v) and (c1,c2).
                    # The other would be removing (u,c2) and (c1,v)? No, those are adjacent in the cycle?
                    # Cycle: u-c1-v-c2-u.
                    # Edges: (u,c1), (c1,v), (v,c2), (c2,u).
                    # Non-adjacent pairs: {(u,c1), (v,c2)} and {(c1,v), (c2,u)}.
                    # We tried the first. Let's try the second.
                    # Remove (c1,v) and (c2,u). Add (c1,c2) and (v,u).
                    # This is the same set of added edges: (u,v) and (c1,c2).
                    # So if the first was invalid, the second is also invalid (same added edges).
                    # So we are stuck with this C4 instance.
                    # We can try a different C4.
                    pass

        return None

    def try_n(n):
        """Try to construct a valid graph on n vertices."""
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
            
        # If target degree is 0, empty graph works
        if target_deg == 0:
            return {'n': n, 'edges': []}
            
        # Use the local search regular graph method
        adj = local_search_regular_c4_free(n, target_deg)
        
        if adj is not None:
            # Final verification
            if is_c4_free(adj) and np.min(adj.sum(axis=1)) >= target_deg:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # Fallback: Try a greedy approach if local search fails
        # (Reusing logic from previous program but with time check)
        adj_greedy = np.zeros((n, n), dtype=np.int8)
        degrees = np.zeros(n, dtype=int)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > deadline - 5:
                break
            if np.min(degrees) >= target_deg:
                break
            if degrees[u] >= target_deg and degrees[v] >= target_deg:
                continue
            
            # Check C4
            common = np.sum(adj[u] & adj[v])
            if common <= 1:
                adj[u, v] = 1
                adj[v, u] = 1
                degrees[u] += 1
                degrees[v] += 1
        
        if np.min(degrees) >= target_deg and is_c4_free(adj_greedy):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_greedy[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        return None

    # Try n=47 first, then 46, 45...
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_n(n)
        if result is not None:
            return result
    
    # Fallback: n=39, empty graph
    return {'n': 39, 'edges': []}

