import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Helper: Check validity of a graph represented by adjacency bitmasks
    def is_valid(adj):
        # Compute degrees
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        # For each edge (u, v), check all paths s-u-v-t
        # s in N(u)\{v}, t in N(v)\{u}, s != t
        # If deg(s) == deg(t), invalid.
        for u in range(n):
            au = adj[u]
            for v in range(n):
                if not (au & (1 << v)):
                    continue
                if u > v:
                    continue
                # Edge (u, v) exists
                s_mask = au & ~(1 << v)
                t_mask = adj[v] & ~(1 << u)
                
                # Optimization: Pre-compute degrees of t's in a set?
                # For n=15, simple loop is fast enough if optimized.
                sm = s_mask
                while sm:
                    s_bit = sm & -sm
                    s = s_bit.bit_length() - 1
                    sm ^= s_bit
                    ds = degrees[s]
                    
                    # Check if any t in t_mask (t != s) has degree ds
                    tm = t_mask & ~(1 << s)
                    while tm:
                        t_bit = tm & -tm
                        t = t_bit.bit_length() - 1
                        tm ^= t_bit
                        if degrees[t] == ds:
                            return False
        return True

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # Strategy 1: Construct K_7,8 (Known valid, 56 edges)
    best_adj = [0] * n
    for u in range(7):
        for v in range(7, 15):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
    
    best_count = count_edges(best_adj)
    
    # Strategy 2: Local search with specific move sets
    # Move: Remove k cross-edges, add k+1 intra-edges
    # We will try k=1, 2, 3
    
    adj = best_adj[:]
    count = best_count
    
    random.seed(42)
    
    # Precompute lists of edges for faster access
    def get_edges(adj):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                if ai & (1 << j):
                    edges.append((i, j))
        return edges

    # We will perform a more structured search
    # Try to improve by swapping edges
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        trial_adj = adj[:]
        
        # Randomly select 1 or 2 cross-edges to remove
        # Cross edges are between 0-6 and 7-14
        # Intra edges are within 0-6 or 7-14
        
        # Identify current cross edges
        cross_edges = []
        for u in range(7):
            au = trial_adj[u]
            # Mask for B part
            mask_b = 0
            for v in range(7, 15):
                mask_b |= (1 << v)
            au_b = au & mask_b
            while au_b:
                bit = au_b & -au_b
                v = bit.bit_length() - 1
                au_b ^= bit
                cross_edges.append((u, v))
        
        if not cross_edges:
            continue
            
        # Pick 1 or 2 cross edges to remove
        num_remove = random.choice([1, 2])
        if len(cross_edges) < num_remove:
            continue
            
        removed = random.sample(cross_edges, num_remove)
        
        for u, v in removed:
            trial_adj[u] &= ~(1 << v)
            trial_adj[v] &= ~(1 << u)
            
        # Try to add num_remove + 1 intra-edges
        added = 0
        target_add = num_remove + 1
        
        # Generate candidate intra-edges
        intra_candidates_a = []
        for i in range(7):
            for j in range(i+1, 7):
                if not (trial_adj[i] & (1 << j)):
                    intra_candidates_a.append((i, j))
                    
        intra_candidates_b = []
        for i in range(7, 15):
            for j in range(i+1, 15):
                if not (trial_adj[i] & (1 << j)):
                    intra_candidates_b.append((i, j))
                    
        candidates = intra_candidates_a + intra_candidates_b
        
        if len(candidates) < target_add:
            continue
            
        added_edges = random.sample(candidates, target_add)
        
        for u, v in added_edges:
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            added += 1
            
        new_count = count - num_remove + added
        
        if new_count <= best_count:
            continue
            
        if is_valid(trial_adj):
            adj = trial_adj
            count = new_count
            best_count = count
            best_adj = adj[:]
            
    # Final result from best_adj
    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([int(i), int(j)])
    
    return final_edges
