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
    
    # Strategy 2: Try to improve via local search on edge additions/swaps
    # We know simple addition to K_7,8 fails. 
    # Let's try a "Greedy Removal" from a denser graph to see if we can land on >56.
    # Start with K_15 and remove edges to fix violations.
    
    # However, K_15 is far from valid. 
    # Alternative: Start with K_7,8 and try "Swaps".
    # Remove edge (a,b) from bipartite, add edge (c,d) within partition?
    # This is complex.
    
    # Let's try a simple local search on the K_7,8 structure:
    # Try to add edges. If invalid, try removing a conflicting edge and adding a different one?
    # Given the proof that adding intra-partition edges to K_7,8 fails,
    # we must look for a different topology.
    
    # Let's try to construct a graph with 3 partitions (Turan-like) but optimized.
    # Or just run a randomized local search starting from K_7,8 allowing edge swaps.
    
    adj = best_adj[:]
    count = best_count
    
    # Local Search: Try to add an edge. If invalid, try to "repair" by removing another edge?
    # Or just accept that K_7,8 might be optimal for this constraint.
    
    # Let's do a quick check: Is there a graph with 57 edges?
    # We will perform a brief simulated annealing or random restart local search 
    # starting from K_7,8, allowing moves: Add edge, Remove edge, Swap edges.
    
    # To keep it fast and deterministic-ish, we will try a few random swaps.
    # But randomization might not beat the deterministic K_7,8 if it's the global max.
    
    # Let's verify if K_7,8 is indeed the best bipartite.
    # K_6,9 -> 54 edges.
    # K_5,10 -> 50 edges.
    # K_8,7 -> 56 edges (same).
    
    # What if we take K_7,8 and remove one edge to allow adding two intra-partition edges?
    # Remove (a,b) where a in A, b in B.
    # deg(a) becomes 7, deg(b) becomes 6.
    # Now add (a1, a2) in A. deg(a1), deg(a2) increase.
    # This breaks the symmetry.
    
    # Let's try a specific "Swap" heuristic:
    # 1. Pick a random edge (u,v) in K_7,8.
    # 2. Remove it.
    # 3. Try to add 2 random intra-partition edges.
    # 4. Check validity.
    # 5. If valid and count > best, keep.
    
    # Given the 60s limit, we can do ~1000 such trials.
    
    random.seed(42) # For reproducibility in search
    
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        trial_adj = adj[:]
        # Pick a random bipartite edge to remove
        a_idx = random.randint(0, 6)
        b_idx = random.randint(7, 14)
        
        # Remove (a_idx, b_idx)
        trial_adj[a_idx] &= ~(1 << b_idx)
        trial_adj[b_idx] &= ~(1 << a_idx)
        
        # Try to add 1 or 2 intra-partition edges
        added = 0
        for _ in range(2):
            if random.random() < 0.5:
                # Add in A
                i = random.randint(0, 6)
                j = random.randint(0, 6)
                if i != j and not (trial_adj[i] & (1 << j)):
                    trial_adj[i] |= (1 << j)
                    trial_adj[j] |= (1 << i)
                    added += 1
            else:
                # Add in B
                i = random.randint(7, 14)
                j = random.randint(7, 14)
                if i != j and not (trial_adj[i] & (1 << j)):
                    trial_adj[i] |= (1 << j)
                    trial_adj[j] |= (1 << i)
                    added += 1
        
        new_count = count - 1 + added
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
