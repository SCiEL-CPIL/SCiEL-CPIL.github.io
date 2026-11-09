import time
import random
import heapq

# EVOLVE-BLOCK-START
def construct():
    n = 512
    # 1. Build Conflict Graph
    # Two numbers x, y conflict if they are in a 4-AP.
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # We can just check all pairs? No, that's O(N^2).
    # Better: iterate over all 4-APs and add edges.
    
    adj = [set() for _ in range(n + 1)]
    
    # Generate all 4-APs
    # a from 1 to n
    # d from 1 to (n - a) // 3
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The set is {a, b, c, e}
            # All pairs in this set are conflicting
            # (a,b), (a,c), (a,e), (b,c), (b,e), (c,e)
            # We can just add all pairs
            verts = [a, b, c, e]
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = verts[i], verts[j]
                    adj[u].add(v)
                    adj[v].add(u)
    
    # 2. Greedy Vertex Cover with Restarts
    # We want to minimize the size of the Vertex Cover C.
    # The Independent Set is V \ C.
    
    def greedy_vc(seed_offset=0):
        # Use a specific random seed for reproducibility in a run, 
        # but vary it across restarts.
        # Actually, just using random module is fine.
        
        remaining = set(range(1, n + 1))
        deg = {v: len(adj[v]) for v in remaining}
        
        # Max-heap: (-deg, v)
        # We need to handle stale entries in the heap.
        heap = [(-deg[v], v) for v in remaining]
        heapq.heapify(heap)
        
        cover = set()
        
        # To allow randomization, we can shuffle the initial heap or 
        # add random noise to degrees. 
        # Simplest: shuffle the vertices before building the heap?
        # No, heap order depends on degree.
        # Let's just pick the max degree, but if there are ties, pick randomly?
        # Or just run the standard greedy.
        
        # A better randomization: 
        # 1. Randomly permute vertices.
        # 2. In case of tie in degree, pick the one earlier in the permutation?
        # This is hard with a heap.
        
        # Alternative: 
        # Just run the standard greedy. It's deterministic.
        # To get diversity, we can remove a random subset of vertices first? No.
        
        # Let's try a slightly randomized greedy:
        # When selecting the max degree vertex, if there are multiple with similar degree,
        # pick one at random.
        
        # Implementation:
        # Pop from heap. If stale (deg changed or removed), discard.
        # If valid, select v.
        
        # To introduce randomness, we can add a small random value to the degree?
        # Or just shuffle the initial list and use a stable sort?
        
        # Let's stick to a robust deterministic first, then if time allows, restarts with random noise.
        
        # Actually, for MIS, "Pick Min Degree" is the standard greedy.
        # "Pick Max Degree" for Vertex Cover is the standard greedy.
        # They are dual. 
        # Let's use the Vertex Cover approach as it tends to be very effective for this structure.
        
        while heap:
            neg_d, v = heapq.heappop(heap)
            if v not in remaining:
                continue
            # Check if degree is still current
            # We don't store current degree in heap, we store the one at push time.
            # If deg[v] != -neg_d, it's stale.
            if deg[v] != -neg_d:
                continue
                
            if deg[v] == 0:
                # No edges left incident to v, we can leave it in IS (not in Cover)
                # But if we are building Cover, we only add vertices that cover edges.
                # If deg[v]==0, it doesn't need to be in Cover to cover edges.
                # However, the loop condition is "while edges exist".
                # If max degree is 0, no edges exist.
                break
            
            # Select v for Cover
            cover.add(v)
            remaining.remove(v)
            
            # Update neighbors
            for u in adj[v]:
                if u in remaining:
                    deg[u] -= 1
                    # Push new degree
                    heapq.heappush(heap, (-deg[u], u))
                    # Remove edge from u's perspective? 
                    # We don't explicitly remove edges from adj, we just use 'remaining' and 'deg'.
                    # But we must ensure we don't process removed vertices.
                    # The 'v not in remaining' check handles that.
            
            # Note: We don't remove u from adj[v] because v is gone.
            # We don't remove v from adj[u] because we check 'u in remaining' and 'v not in remaining' logic implicitly via deg updates?
            # Actually, deg[u] is decremented. 
            # We don't need to physically remove v from adj[u] as long as we trust deg[u].
            # But when we pop u later, we check if u is in remaining.
            # And we check if deg[u] matches heap.
            # This is correct.
        
        # The Independent Set is all vertices NOT in cover
        # Wait, vertices with deg 0 that were never added to cover are in IS.
        # Vertices removed from remaining (added to cover) are NOT in IS.
        # So IS = {v in 1..n | v not in cover}
        
        is_set = set(range(1, n + 1)) - cover
        return is_set

    best_is = set()
    best_size = 0
    
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    # Run multiple iterations with random perturbations
    # Perturbation: Add random noise to degrees to break ties differently
    for i in range(10000):
        if time.time() - start_time > timeout:
            break
            
        # Create a copy of degrees with random noise for tie-breaking
        # Or just use the standard greedy. 
        # Standard greedy is deterministic. 
        # To get different results, we can randomly remove a few vertices from the "candidate" set before starting?
        # No, that reduces the size.
        
        # Better: Randomized Greedy.
        # When picking max degree, pick from the top K vertices with max degree randomly.
        
        # Let's implement a randomized version inside the loop.
        
        remaining = set(range(1, n + 1))
        # Initialize deg
        deg = {v: len(adj[v]) for v in remaining}
        
        # We will use a list and sort, or heap.
        # For randomization, let's just shuffle the vertices and use a stable selection?
        # No, let's use the heap but with random noise added to the key.
        # key = -deg[v] + random.uniform(0, 0.1)
        # This effectively breaks ties randomly and slightly favors higher degrees but allows lower ones if noise is high?
        # No, -deg is the main term. Noise is small.
        
        heap = []
        for v in remaining:
            # Add small random noise to break ties
            noise = random.uniform(0, 0.01)
            heapq.heappush(heap, (-deg[v] + noise, v))
            
        cover = set()
        
        while heap:
            key, v = heapq.heappop(heap)
            if v not in remaining:
                continue
            # The key includes noise, so we can't directly compare key to deg[v].
            # We need to store the actual degree separately or recompute?
            # This is getting complicated.
            
            # Simpler Randomization:
            # 1. Compute degrees.
            # 2. Find max degree D.
            # 3. Collect all vertices with degree == D.
            # 4. Pick one randomly.
            # 5. Remove and update.
            # This is O(N) per step to find max and collect. N=512.
            # 512 * 512 * 10000 iterations = 2.6 * 10^9 ops. Too slow in Python.
            
            # Let's stick to the deterministic heap-based greedy first.
            # It is very fast.
            # If it's too slow, we reduce restarts.
            # If it's fast, we can try to improve.
            
            # Let's just run the deterministic one multiple times? 
            # It will give the same result.
            # So we need a different heuristic.
            
            # Heuristic 2: Greedy Independent Set (Min Degree)
            # Pick vertex with min degree, add to IS, remove it and neighbors.
            
            # Let's implement Min Degree Greedy IS.
            
            # Reset
            remaining = set(range(1, n + 1))
            deg = {v: len(adj[v]) for v in remaining}
            
            # Min-heap: (deg, v)
            # Use noise for randomization
            heap = []
            for v in remaining:
                noise = random.uniform(0, 0.1)
                heapq.heappush(heap, (deg[v] + noise, v))
                
            is_set = set()
            
            while heap:
                key, v = heapq.heappop(heap)
                if v not in remaining:
                    continue
                # Check if degree is roughly correct?
                # Since we added noise, we can't verify easily.
                # But if v is in remaining, it's a valid candidate.
                # Is it the min degree?
                # The heap property ensures we pop the smallest (deg+noise).
                # This is a valid randomized greedy.
                
                is_set.add(v)
                remaining.remove(v)
                
                # Remove neighbors
                for u in adj[v]:
                    if u in remaining:
                        remaining.remove(u)
                        # We don't need to update deg[u] in heap if we just remove u from remaining.
                        # The "if v not in remaining" check will skip it later.
                        # But we should push nothing.
                        # Wait, if we remove u, we don't need its degree anymore.
                        # So we just mark it removed.
            
            size = len(is_set)
            if size > best_size:
                best_size = size
                best_is = set(is_set)
        
    # Final check: ensure no 4-AP
    # The construction guarantees it, but a sanity check is good.
    # For a in best_is:
    #   for d in ...:
    #      if all(x in best_is for x in [a, a+d, a+2d, a+3d]): return [] # Fallback
    
    return sorted(best_is)

# EVOLVE-BLOCK-END
