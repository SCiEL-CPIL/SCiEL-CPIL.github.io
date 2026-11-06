import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 130.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    # Precompute all possible edges as packed ints: u*n + v (u < v)
    all_edge_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edge_set = set(u * n + v for u, v in all_edge_pairs)
    
    # Precompute triangle triples (u, v, w) with u < v < w
    # Only include triples where all three pairs are in the complete graph
    # We'll check at runtime which ones are actual triangles
    triangle_triples = [(u, v, w) for u in range(n) for v in range(u+1, n) for w in range(v+1, n)]
    
    # Precompute bit masks for each vertex
    bit = [1 << i for i in range(n)]
    
    def calc_score_fast(adj, degs):
        """Calculate max triangle degree sum using precomputed triples."""
        max_sum = 0
        for (u, v, w) in triangle_triples:
            # Check if (u,v,w) is a triangle
            if (adj[u] & bit[v]) and (adj[v] & bit[w]) and (adj[w] & bit[u]):
                s = degs[u] + degs[v] + degs[w]
                if s > max_sum:
                    max_sum = s
        return max_sum
    
    # Initialize with K_{8,8} + (0,1)
    # 0-7 in A, 8-15 in B
    adj = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= bit[v]
            adj[v] |= bit[u]
    adj[0] |= bit[1]
    adj[1] |= bit[0]
    
    def get_degrees(adj):
        return [bin(x).count('1') for x in adj]
    
    def get_edge_set_packed(adj):
        es = set()
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    es.add(u * n + v)
                mask -= lsb
        return es
    
    current_adj = adj[:]
    current_degs = get_degrees(current_adj)
    current_score = calc_score_fast(current_adj, current_degs)
    best_adj = current_adj[:]
    best_score = current_score
    best_degs = current_degs[:]
    
    current_edges = get_edge_set_packed(current_adj)
    
    # SA Parameters
    T = 2.0
    T_min = 0.05
    cooling = 0.995
    
    # For move generation: track which edges are in/out for fast rejection
    # Use a list of packed edge ints for current edges
    current_edge_list = list(current_edges)
    
    # Precompute for each vertex, the list of edges incident to it (packed)
    # This helps with biased move generation
    
    def do_swap(e1_packed, e2_packed):
        """Apply swap: remove e1, add e2. Returns None on success, or reverts."""
        u1 = e1_packed // n
        v1 = e1_packed % n
        u2 = e2_packed // n
        v2 = e2_packed % n
        
        # Remove e1
        current_adj[u1] &= ~bit[v1]
        current_adj[v1] &= ~bit[u1]
        current_degs[u1] -= 1
        current_degs[v1] -= 1
        current_edges.discard(e1_packed)
        
        # Add e2
        current_adj[u2] |= bit[v2]
        current_adj[v2] |= bit[u2]
        current_degs[u2] += 1
        current_degs[v2] += 1
        current_edges.add(e2_packed)
    
    def undo_swap(e1_packed, e2_packed):
        """Revert swap: add e1 back, remove e2."""
        u1 = e1_packed // n
        v1 = e1_packed % n
        u2 = e2_packed // n
        v2 = e2_packed % n
        
        # Re-add e1
        current_adj[u1] |= bit[v1]
        current_adj[v1] |= bit[u1]
        current_degs[u1] += 1
        current_degs[v1] += 1
        current_edges.add(e1_packed)
        
        # Remove e2
        current_adj[u2] &= ~bit[v2]
        current_adj[v2] &= ~bit[u2]
        current_degs[u2] -= 1
        current_degs[v2] -= 1
        current_edges.discard(e2_packed)
    
    # Restart mechanism
    num_restarts = 0
    max_restarts = 10
    
    while time.time() - start_time < time_limit:
        # Generate move: pick a random existing edge and a random non-existing edge
        if not current_edge_list:
            break
        
        # Biased selection: prefer edges incident to vertices with high degree
        # This targets the triangles that likely achieve the max score
        # Simple approach: just pick randomly from current edges
        e1_packed = random.choice(current_edge_list)
        
        # Pick random non-existing edge
        # Pre-generate a list of non-edges for faster sampling
        # But that's expensive to maintain. Instead, rejection sampling.
        e2_packed = None
        attempts = 0
        while e2_packed is None:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            e2_candidate = u2 * n + v2
            if e2_candidate not in current_edges:
                e2_packed = e2_candidate
            attempts += 1
            if attempts > 100:
                break
        
        if e2_packed is None:
            # Graph is nearly complete, try a different approach
            # Just pick from all edges not in current
            non_edges = all_edge_set - current_edges
            if not non_edges:
                break
            e2_packed = random.choice(list(non_edges))
        
        # Apply swap
        do_swap(e1_packed, e2_packed)
        new_score = calc_score_fast(current_adj, current_degs)
        
        # Acceptance
        if new_score < current_score:
            accept = True
        elif new_score == current_score:
            accept = random.random() < 0.3
        else:
            delta = new_score - current_score
            if T > 0.01:
                accept = random.random() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
                best_degs = current_degs[:]
                # Update current_edge_list
                current_edge_list = list(current_edges)
        else:
            # Revert
            undo_swap(e1_packed, e2_packed)
            # Rebuild edge list if it got out of sync
            if e1_packed not in current_edge_list:
                current_edge_list = list(current_edges)
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Periodic restart: if stuck, reheat
        if T <= T_min * 1.01 and num_restarts < max_restarts:
            # Check if we've been stuck for a while by checking if best hasn't improved
            # Simple heuristic: reheat and continue
            T = 1.0
            num_restarts += 1
    
    # Final polish: local search on best_adj
    # Try all single edge swaps to see if we can improve
    polish_adj = best_adj[:]
    polish_degs = get_degrees(polish_adj)
    polish_edges = get_edge_set_packed(polish_adj)
    polish_score = calc_score_fast(polish_adj, polish_degs)
    
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        edge_list = list(polish_edges)
        random.shuffle(edge_list)
        for e1_packed in edge_list:
            if time.time() - start_time > time_limit - 5:
                break
            u1 = e1_packed // n
            v1 = e1_packed % n
            
            # Try all possible non-edges
            for u2 in range(n):
                for v2 in range(u2 + 1, n):
                    e2_packed = u2 * n + v2
                    if e2_packed in polish_edges:
                        continue
                    
                    # Apply swap
                    polish_adj[u1] &= ~bit[v1]
                    polish_adj[v1] &= ~bit[u1]
                    polish_degs[u1] -= 1
                    polish_degs[v1] -= 1
                    polish_edges.discard(e1_packed)
                    
                    polish_adj[u2] |= bit[v2]
                    polish_adj[v2] |= bit[u2]
                    polish_degs[u2] += 1
                    polish_degs[v2] += 1
                    polish_edges.add(e2_packed)
                    
                    new_s = calc_score_fast(polish_adj, polish_degs)
                    
                    if new_s < polish_score:
                        polish_score = new_s
                        improved = True
                        break
                    else:
                        # Revert
                        polish_adj[u1] |= bit[v1]
                        polish_adj[v1] |= bit[u1]
                        polish_degs[u1] += 1
                        polish_degs[v1] += 1
                        polish_edges.add(e1_packed)
                        
                        polish_adj[u2] &= ~bit[v2]
                        polish_adj[v2] &= ~bit[u2]
                        polish_degs[u2] -= 1
                        polish_degs[v2] -= 1
                        polish_edges.discard(e2_packed)
            
            if improved:
                break
    
    if polish_score < best_score:
        best_adj = polish_adj[:]
        best_score = polish_score
    
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v > u:
                edges.append([u, v])
            mask -= lsb
            
    return {'n': 16, 'edges': edges}
