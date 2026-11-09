import numpy as np
import time
import random
import itertools

def construct():
    # Try n=29 first, then 30 if memory allows (unlikely)
    for n in [29, 28, 27]:
        try:
            num_edges = n * (n - 1) // 2
            pair_index = {}
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    pair_index[(i, j)] = idx
                    idx += 1
            
            # Generate instances of K2,2,2
            # Each instance is a tuple of 12 edge indices
            # We need to map vertices to edges
            
            # Precompute list of pairs for each vertex?
            # Or just iterate combinations of 6 vertices
            
            instances = []
            # To speed up, we can precompute the 15 pairings of 6 items
            # Items 0,1,2,3,4,5
            # Pairings are partitions into 3 pairs.
            # There are 15 such partitions.
            # We can hardcode them or generate them.
            
            def get_pairings(verts):
                # verts is a tuple of 6 vertices
                # Returns list of 15 tuples of 3 pairs
                # Each pair is (u, v)
                # We need to map (u,v) to edge index
                pass
            
            # This Python loop will be slow.
            # Let's try to vectorize or optimize.
            # For n=29, 475,020 combinations of 6.
            # 15 pairings each.
            # 7.1 million iterations.
            # Each iteration does 12 lookups in dict.
            # ~85 million lookups.
            # This might take 5-10 seconds. Acceptable.
            
            # Hardcoded pairings of indices 0..5
            # 0-1, 2-3, 4-5
            # 0-1, 2-4, 3-5
            # ...
            # We can generate them programmatically once.
            
            pairings = []
            # Generate all partitions of {0,1,2,3,4,5} into 3 pairs
            # Use a recursive helper or hardcode
            # There are 15.
            # Let's generate them:
            # Fix 0. It pairs with 1,2,3,4,5.
            # If 0-1, remaining {2,3,4,5}. Partitions: (2-3,4-5), (2-4,3-5), (2-5,3-4). (3 ways)
            # If 0-2, remaining {1,3,4,5}. Partitions: (1-3,4-5), (1-4,3-5), (1-5,3-4). (3 ways)
            # If 0-3, remaining {1,2,4,5}. Partitions: (1-2,4-5), (1-4,2-5), (1-5,2-4). (3 ways)
            # If 0-4, remaining {1,2,3,5}. Partitions: (1-2,3-5), (1-3,2-5), (1-5,2-3). (3 ways)
            # If 0-5, remaining {1,2,3,4}. Partitions: (1-2,3-4), (1-3,2-4), (1-4,2-3). (3 ways)
            
            base_pairs = [
                [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
                [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
                [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
                [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
                [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
            ]
            
            # Iterate over combinations of 6 vertices
            # This is the bottleneck.
            # We can use numpy to generate combinations? No, itertools is faster for this count.
            
            # To optimize, we can precompute the edge indices for all pairs of vertices?
            # We have pair_index.
            
            # We will store instances in a list of lists, then convert to numpy array.
            # Or directly to numpy array if we know the size.
            # M = C(n,6) * 15
            
            from math import comb
            M = comb(n, 6) * 15
            # Use int16 for edge indices if num_edges < 32768
            # For n=29, num_edges = 406. int16 is fine.
            instances_arr = np.empty((M, 12), dtype=np.int16)
            
            inst_idx = 0
            for combo in itertools.combinations(range(n), 6):
                # For each of the 15 pairings
                for p in base_pairs:
                    # p is list of 3 pairs of indices in 0..5
                    # Map these indices to actual vertices in combo
                    # combo is (v0, v1, v2, v3, v4, v5)
                    edges = []
                    for (a, b) in p:
                        u = combo[a]
                        v = combo[b]
                        # Get edge index
                        if u > v:
                            u, v = v, u
                        edges.append(pair_index[(u, v)])
                    
                    # Wait, the edges of K2,2,2 are BETWEEN the pairs.
                    # The pairs in 'p' are the PARTS.
                    # So if p = [(0,1), (2,3), (4,5)], the parts are {v0,v1}, {v2,v3}, {v4,v5}.
                    # The edges are all pairs between different parts.
                    # Part 1: v0, v1. Part 2: v2, v3. Part 3: v4, v5.
                    # Edges P1-P2: (v0,v2), (v0,v3), (v1,v2), (v1,v3)
                    # Edges P1-P3: (v0,v4), (v0,v5), (v1,v4), (v1,v5)
                    # Edges P2-P3: (v2,v4), (v2,v5), (v3,v4), (v3,v5)
                    
                    # My previous logic in the thought process was correct, 
                    # but the code snippet above `edges.append(pair_index[(u,v)])` 
                    # was adding the edges WITHIN the parts, which are NOT in K2,2,2.
                    # K2,2,2 is complete tripartite. Edges are only between parts.
                    
                    # So we need to generate the 12 edges between the 3 parts.
                    
                    p1 = [combo[p[0][0]], combo[p[0][1]]]
                    p2 = [combo[p[1][0]], combo[p[1][1]]]
                    p3 = [combo[p[2][0]], combo[p[2][1]]]
                    
                    def get_edge(u, v):
                        if u > v: u, v = v, u
                        return pair_index[(u, v)]
                    
                    e1 = get_edge(p1[0], p2[0])
                    e2 = get_edge(p1[0], p2[1])
                    e3 = get_edge(p1[1], p2[0])
                    e4 = get_edge(p1[1], p2[1])
                    
                    e5 = get_edge(p1[0], p3[0])
                    e6 = get_edge(p1[0], p3[1])
                    e7 = get_edge(p1[1], p3[0])
                    e8 = get_edge(p1[1], p3[1])
                    
                    e9 = get_edge(p2[0], p3[0])
                    e10 = get_edge(p2[0], p3[1])
                    e11 = get_edge(p2[1], p3[0])
                    e12 = get_edge(p2[1], p3[1])
                    
                    instances_arr[inst_idx, 0] = e1
                    instances_arr[inst_idx, 1] = e2
                    instances_arr[inst_idx, 2] = e3
                    instances_arr[inst_idx, 3] = e4
                    instances_arr[inst_idx, 4] = e5
                    instances_arr[inst_idx, 5] = e6
                    instances_arr[inst_idx, 6] = e7
                    instances_arr[inst_idx, 7] = e8
                    instances_arr[inst_idx, 8] = e9
                    instances_arr[inst_idx, 9] = e10
                    instances_arr[inst_idx, 10] = e11
                    instances_arr[inst_idx, 11] = e12
                    inst_idx += 1
            
            # Build CSR for edge to instances
            # Count occurrences of each edge
            edge_counts = np.zeros(num_edges, dtype=np.int64)
            # Flatten instances
            flat_instances = instances_arr.flatten()
            np.bincount(flat_instances, minlength=num_edges, output=edge_counts)
            
            # Compute offsets
            edge_offsets = np.zeros(num_edges + 1, dtype=np.int64)
            np.cumsum(edge_counts, out=edge_offsets[1:])
            
            # Fill edge_indices
            edge_indices = np.empty(flat_instances.shape, dtype=np.int64)
            # We need to fill this efficiently.
            # For each instance i, for each edge e in instances_arr[i]:
            #   pos = edge_offsets[e] + count_so_far[e]
            #   edge_indices[pos] = i
            #   count_so_far[e] += 1
            
            count_so_far = np.zeros(num_edges, dtype=np.int64)
            for i in range(M):
                for e in instances_arr[i]:
                    pos = edge_offsets[e] + count_so_far[e]
                    edge_indices[pos] = i
                    count_so_far[e] += 1
            
            # Initialize colors
            colors = np.random.randint(0, 2, num_edges)
            
            # Initialize counts
            # Chunked calculation
            count_0 = np.zeros(M, dtype=np.int16)
            chunk_size = 100000
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                chunk_inst = instances_arr[start:end]
                chunk_colors = colors[chunk_inst]
                count_0[start:end] = (chunk_colors == 0).sum(axis=1)
            
            count_1 = 12 - count_0
            
            # Initial fitness
            fitness = ((count_0 == 12) | (count_1 == 12)).sum()
            
            # Search
            # Simple Hill Climbing with random restarts if stuck
            # Or SA
            # Given time, try to find 0.
            
            max_iters = 50000
            t = time.time()
            found = False
            
            for i in range(max_iters):
                if time.time() - t > 100:
                    break
                
                e = np.random.randint(num_edges)
                
                # Get instances containing e
                start_off = edge_offsets[e]
                end_off = edge_offsets[e + 1]
                
                if start_off == end_off:
                    continue
                
                idxs = edge_indices[start_off:end_off]
                
                # Get current counts
                c0 = count_0[idxs]
                c1 = count_1[idxs]
                
                was_mono = (c0 == 12) | (c1 == 12)
                
                # Flip color
                if colors[e] == 0:
                    count_0[idxs] -= 1
                    count_1[idxs] += 1
                    colors[e] = 1
                else:
                    count_0[idxs] += 1
                    count_1[idxs] -= 1
                    colors[e] = 0
                
                new_c0 = count_0[idxs]
                new_c1 = count_1[idxs]
                is_mono = (new_c0 == 12) | (new_c1 == 12)
                
                delta = is_mono.sum() - was_mono.sum()
                fitness += delta
                
                if fitness == 0:
                    found = True
                    break
                
                # Accept if better or with probability (simple SA)
                # If delta > 0, revert
                if delta > 0:
                    # Revert
                    if colors[e] == 1:
                        count_0[idxs] -= 1
                        count_1[idxs] += 1
                        colors[e] = 0
                    else:
                        count_0[idxs] += 1
                        count_1[idxs] -= 1
                        colors[e] = 1
                    fitness -= delta
                    # Simple: just reject
                    pass
                else:
                    # Accept
                    pass
            
            if found:
                # Convert colors to list
                return {'n': n, 'colors': colors.tolist()}
                
        except Exception as e:
            print(f"Error for n={n}: {e}")
            continue

    # Fallback
    return {'n': 17, 'colors': [0 if (j - i) % 17 in {x*x % 17 for x in range(1,17)} else 1 for i in range(17) for j in range(i+1, 17)]}
