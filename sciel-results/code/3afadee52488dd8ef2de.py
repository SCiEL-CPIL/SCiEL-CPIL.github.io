import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    m = 65
    
    best_score = float('inf')
    best_edges = None
    
    # Try all bipartite splits
    for a in range(1, n // 2 + 1):
        b = n - a
        # We can put internal edges in A or B. 
        # Usually better to put in the larger partition to dilute degree increases.
        # But let's try both or just the larger one if k is small.
        # Actually, if k is large, we might need edges in both, but here m is close to max bipartite.
        
        max_edges_bipartite = a * b
        if m > max_edges_bipartite:
            k = m - max_edges_bipartite
            # We need k internal edges.
            # We can add them to A or B.
            # Let's try adding to the larger partition first, as it has more vertices to spread load.
            # If k is too large for one partition, we'd need a more complex structure, 
            # but for n=16, m=65, the optimal is likely near balanced.
            
            for target_partition_size in [a, b]:
                if k > target_partition_size * (target_partition_size - 1) // 2:
                    continue
                    
                # The other partition size is n - target_partition_size
                other_size = n - target_partition_size
                
                # Base degrees:
                # Vertices in target partition have base degree 'other_size'
                # Vertices in other partition have base degree 'target_partition_size'
                
                # We add k edges to target partition.
                # Let d[i] be the number of internal edges incident to vertex i in target partition.
                # Degree of vertex i in target partition = other_size + d[i]
                # Degree of vertex j in other partition = target_partition_size
                
                # Triangles are formed by an internal edge (u, v) in target partition 
                # and any vertex w in the other partition.
                # Sum = (other_size + d[u]) + (other_size + d[v]) + target_partition_size
                # Sum = 2 * other_size + target_partition_size + d[u] + d[v]
                
                base_sum = 2 * other_size + target_partition_size
                
                # We need to place k edges in a graph of size target_partition_size 
                # to minimize max(d[u] + d[v]) over all edges (u,v).
                
                # Greedy placement:
                # Maintain list of degrees d, initially 0.
                # Maintain list of edges added.
                # At each step, find pair (u,v) not connected, minimizing d[u] + d[v].
                # If tie, pick arbitrary.
                
                if k == 0:
                    current_max_add = 0
                else:
                    d = [0] * target_partition_size
                    edges_added = []
                    # Precompute possible edges
                    # To speed up, we can just iterate
                    for _ in range(k):
                        best_pair = None
                        best_sum = float('inf')
                        # Optimization: only check pairs with low current degrees
                        # Since k is small, O(V^2 * k) is fine for V=16
                        for u in range(target_partition_size):
                            for v in range(u + 1, target_partition_size):
                                # Check if edge exists
                                # For small k, linear scan of edges_added is fine
                                exists = False
                                for e in edges_added:
                                    if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                                        exists = True
                                        break
                                if not exists:
                                    s = d[u] + d[v]
                                    if s < best_sum:
                                        best_sum = s
                                        best_pair = (u, v)
                        
                        if best_pair is None:
                            break # Should not happen if k is valid
                            
                        u, v = best_pair
                        edges_added.append((u, v))
                        d[u] += 1
                        d[v] += 1
                        
                    # The max sum for a triangle is base_sum + max(d[u] + d[v] for (u,v) in edges_added)
                    if edges_added:
                        max_d_sum = max(d[u] + d[v] for u, v in edges_added)
                    else:
                        max_d_sum = 0
                        
                    current_max_add = max_d_sum
                
                total_score = base_sum + current_max_add
                
                if total_score < best_score:
                    best_score = total_score
                    # Construct the actual edge list
                    # Vertices 0..a-1 are partition A, a..n-1 are partition B
                    # We need to map the "target partition" indices to actual vertex IDs.
                    
                    # Determine which partition is the target
                    # In the loop, we iterated over target_partition_size in [a, b].
                    # We need to know which one it was.
                    # Let's re-derive based on the loop variable.
                    # This is tricky because we lost the context of 'a' vs 'b' in the inner loop.
                    # Let's refactor to explicitly try "Internal in A" and "Internal in B".
                    
                    pass # Placeholder, we will refactor the loop structure below.

    # Refactored Loop Structure for Clarity and Correctness
    
    best_score = float('inf')
    best_edges = None
    
    for a in range(1, n // 2 + 1):
        b = n - a
        bipartite_edges = a * b
        if m < bipartite_edges:
            continue # We need exactly m edges, and we are assuming core is K_{a,b}
            # If m < a*b, we would remove edges from K_{a,b}. 
            # But removing edges from K_{a,b} doesn't create triangles, it just reduces degrees.
            # The task says "has triangles", implying m > max bipartite edges?
            # floor(16^2/4) = 64. m=65. So m > 64.
            # So we are always in the case m > a*b for the optimal a,b.
            
        k = m - bipartite_edges
        
        # Option 1: Internal edges in A (size a)
        if k <= a * (a - 1) // 2:
            # Base degrees: A vertices have b, B vertices have a.
            # Triangle sum = (b + d[u]) + (b + d[v]) + a = 2b + a + d[u] + d[v]
            base_sum = 2 * b + a
            # Minimize max(d[u] + d[v]) for k edges in A
            d = [0] * a
            edges_added = []
            for _ in range(k):
                best_pair = None
                best_sum = float('inf')
                for u in range(a):
                    for v in range(u + 1, a):
                        exists = any((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u) for e in edges_added)
                        if not exists:
                            s = d[u] + d[v]
                            if s < best_sum:
                                best_sum = s
                                best_pair = (u, v)
                if best_pair:
                    u, v = best_pair
                    edges_added.append((u, v))
                    d[u] += 1
                    d[v] += 1
            
            max_d_sum = max((d[u] + d[v] for u, v in edges_added), default=0)
            score = base_sum + max_d_sum
            
            if score < best_score:
                best_score = score
                # Build edges
                # A: 0..a-1, B: a..n-1
                edges = []
                # Bipartite
                for u in range(a):
                    for v in range(a, n):
                        edges.append([u, v])
                # Internal in A
                for u, v in edges_added:
                    edges.append([u, v])
                best_edges = edges
                
        # Option 2: Internal edges in B (size b)
        if k <= b * (b - 1) // 2:
            # Base degrees: A vertices have b, B vertices have a.
            # Triangle sum = (a + d[u]) + (a + d[v]) + b = 2a + b + d[u] + d[v]
            base_sum = 2 * a + b
            # Minimize max(d[u] + d[v]) for k edges in B
            d = [0] * b
            edges_added = []
            for _ in range(k):
                best_pair = None
                best_sum = float('inf')
                for u in range(b):
                    for v in range(u + 1, b):
                        exists = any((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u) for e in edges_added)
                        if not exists:
                            s = d[u] + d[v]
                            if s < best_sum:
                                best_sum = s
                                best_pair = (u, v)
                if best_pair:
                    u, v = best_pair
                    edges_added.append((u, v))
                    d[u] += 1
                    d[v] += 1
            
            max_d_sum = max((d[u] + d[v] for u, v in edges_added), default=0)
            score = base_sum + max_d_sum
            
            if score < best_score:
                best_score = score
                # Build edges
                # A: 0..a-1, B: a..n-1
                # Internal edges in B are between vertices a..n-1
                # Map local index i to global index a+i
                edges = []
                # Bipartite
                for u in range(a):
                    for v in range(a, n):
                        edges.append([u, v])
                # Internal in B
                for u, v in edges_added:
                    edges.append([a + u, a + v])
                best_edges = edges

    if time.time() > deadline:
         # Return best found so far or fallback
         if best_edges is None:
             # Fallback to K8,8 + 1 edge
             edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
             return {'n': 16, 'edges': edges}
    
    if best_edges is None:
        # Fallback
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    else:
        edges = best_edges

    return {'n': 16, 'edges': edges}
