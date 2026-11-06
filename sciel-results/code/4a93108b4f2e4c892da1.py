"""Program for the graph construction task."""
import time
import random

DEADLINE = 145  # safety margin before 150s hard kill

def _compute_score_fast(adj, degrees):
    """
    Compute the largest degree sum over triangles using bitmask adjacency.
    adj: list of 16 integers, where bit i is set if vertex i is a neighbor.
    degrees: list of 16 integers, degree of each vertex.
    """
    n = 16
    max_sum = 0
    for u in range(n):
        au = adj[u]
        du = degrees[u]
        # Only consider v > u to avoid duplicate edge processing
        # We can iterate over set bits in au that are > u
        # But simpler: loop v from u+1 to 15
        for v in range(u + 1, n):
            if (au >> v) & 1:
                dv = degrees[v]
                common = au & adj[v]
                # Iterate over common neighbors w
                # We only care about w > v to count each triangle once
                # Mask out bits <= v
                # (1 << (v+1)) - 1 gives mask for bits 0..v
                # We want bits > v
                # So common & ~((1 << (v+1)) - 1)
                mask = ~((1 << (v + 1)) - 1)
                c = common & mask
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    s = du + dv + degrees[w]
                    if s > max_sum:
                        max_sum = s
                    c &= c - 1
    return max_sum

def _mask_to_edges(adj):
    """Convert bitmask adjacency to list of edges."""
    edges = []
    for u in range(16):
        au = adj[u]
        for v in range(u + 1, 16):
            if (au >> v) & 1:
                edges.append([u, v])
    return edges

def _edges_to_mask(edges):
    """Convert list of edges to bitmask adjacency and degrees."""
    adj = [0] * 16
    degrees = [0] * 16
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
    return adj, degrees

def _initial_constructions():
    """Generate diverse initial constructions."""
    constructions = []
    
    # 1. K_{8,8} + (0,1)
    edges1 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    constructions.append(edges1)
    
    # 2. K_{7,9} + 2 internal edges in the 9-part (vertices 7-15)
    # Part A: 0-6 (7 vertices), Part B: 7-15 (9 vertices)
    edges2 = [[u, v] for u in range(7) for v in range(7, 16)]
    # Add (7,8) and (9,10) in B
    edges2.append([7, 8])
    edges2.append([9, 10])
    constructions.append(edges2)
    
    # 3. K_{7,9} + 2 internal edges in the 7-part (vertices 0-6)
    edges3 = [[u, v] for u in range(7) for v in range(7, 16)]
    edges3.append([0, 1])
    edges3.append([2, 3])
    constructions.append(edges3)
    
    # 4. K_{8,8} - 1 cross edge + 2 internal edges
    # Remove (0,8), add (0,1) in A, (8,9) in B
    edges4 = [[u, v] for u in range(8) for v in range(8, 16) if not (u==0 and v==8)]
    edges4.append([0, 1])
    edges4.append([8, 9])
    constructions.append(edges4)

    return constructions

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_adj = None
    best_degrees = None
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def _time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Evaluate initial constructions
    for edges in _initial_constructions():
        if _time_left() < 10:
            break
        adj, degrees = _edges_to_mask(edges)
        score = _compute_score_fast(adj, degrees)
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_degrees = degrees.copy()
            
        # Local Search / Simulated Annealing from this start
        current_adj = adj.copy()
        current_degrees = degrees.copy()
        current_score = score
        
        T = 1.0
        cooling = 0.999
        no_improve = 0
        max_no_improve = 5000
        
        it = 0
        while _time_left() > 5 and no_improve < max_no_improve and it < 500000:
            it += 1
            
            # Pick random edge to remove
            # Find a random set bit in a random vertex
            u1 = random.randint(0, n - 1)
            neighbors = current_adj[u1]
            if neighbors == 0:
                continue
            # Pick random bit from neighbors
            # Convert to list? No, use random index of set bits
            # For N=16, we can just pick random v and check
            v1 = random.randint(0, n - 1)
            if v1 == u1 or not ((neighbors >> v1) & 1):
                # Try a few times to find a valid edge
                found = False
                for _ in range(10):
                    v1 = random.randint(0, n - 1)
                    if v1 != u1 and ((neighbors >> v1) & 1):
                        found = True
                        break
                if not found:
                    continue
            
            # Pick random non-edge to add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            
            # Check if it's a non-edge and not the same as removed
            if (u2 == u1 and v2 == v1) or ((current_adj[u2] >> v2) & 1):
                # Try a few times
                found = False
                for _ in range(10):
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2: continue
                    if u2 > v2: u2, v2 = v2, u2
                    if (u2 == u1 and v2 == v1): continue
                    if not ((current_adj[u2] >> v2) & 1):
                        found = True
                        break
                if not found:
                    continue

            # Perform swap
            new_adj = current_adj.copy()
            new_degrees = current_degrees.copy()
            
            # Remove (u1, v1)
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_degrees[u1] -= 1
            new_degrees[v1] -= 1
            
            # Add (u2, v2)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_degrees[u2] += 1
            new_degrees[v2] += 1
            
            new_score = _compute_score_fast(new_adj, new_degrees)
            
            delta = new_score - current_score
            if delta < 0 or (T > 0.01 and random.random() < __import__('math').exp(-delta / T)):
                current_adj = new_adj
                current_degrees = new_degrees
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    best_degrees = current_degrees.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cooling

    # Final polish: Hill climbing on best_adj
    if best_adj is not None and _time_left() > 10:
        current_adj = best_adj.copy()
        current_degrees = best_degrees.copy()
        current_score = _compute_score_fast(current_adj, current_degrees)
        
        improved = True
        while improved and _time_left() > 5:
            improved = False
            # Try all single edge swaps
            for rem_u, rem_v in all_pairs:
                if _time_left() < 5:
                    break
                if not ((current_adj[rem_u] >> rem_v) & 1):
                    continue
                
                for add_u, add_v in all_pairs:
                    if _time_left() < 5:
                        break
                    if add_u == rem_u and add_v == rem_v:
                        continue
                    if (current_adj[add_u] >> add_v) & 1:
                        continue
                    
                    test_adj = current_adj.copy()
                    test_degrees = current_degrees.copy()
                    
                    test_adj[rem_u] &= ~(1 << rem_v)
                    test_adj[rem_v] &= ~(1 << rem_u)
                    test_degrees[rem_u] -= 1
                    test_degrees[rem_v] -= 1
                    
                    test_adj[add_u] |= (1 << add_v)
                    test_adj[add_v] |= (1 << add_u)
                    test_degrees[add_u] += 1
                    test_degrees[add_v] += 1
                    
                    test_score = _compute_score_fast(test_adj, test_degrees)
                    
                    if test_score < current_score:
                        current_adj = test_adj
                        current_degrees = test_degrees
                        current_score = test_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = current_adj.copy()
                            best_degrees = current_degrees.copy()
                        break # Restart loop with new best

    best_edges = _mask_to_edges(best_adj)
    return {'n': n, 'edges': best_edges}
