"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with a good base: Turan graph T(20, 5) or similar dense structure
    # But K2,3-free requires careful construction.
    # Let's start with a known K2,3-free construction or use greedy.
    
    n = 20
    # Use a greedy approach with random restarts
    best_edges = []
    best_count = 0
    
    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) would create a K2,3."""
        # A K2,3 exists if there are two vertices with 3 common neighbors.
        # When we add edge (u,v), we need to check if this creates a new K2,3.
        # The new K2,3 would involve u and v as the two vertices in K2, and 3 common neighbors.
        # Or it could involve u (or v) and some other vertex w, where the common neighbors increase.
        
        # Actually, the simplest check: after adding edge (u,v), for every pair (a,b),
        # count common neighbors. But that's O(n^2 * n) per edge addition.
        
        # Better: only check pairs that might be affected.
        # Adding edge (u,v) increases common neighbors for pairs involving u or v.
        # Specifically:
        # 1. Pair (u, v): common neighbors are N(u) ∩ N(v). If |N(u) ∩ N(v)| >= 3, K2,3 exists.
        # 2. For any w, pair (u, w): common neighbors are N(u) ∩ N(w). 
        #    The new edge (u,v) means v is now in N(u). So if v was in N(w), then v becomes a new common neighbor.
        #    So we need |N(u) ∩ N(w)| < 3 after addition.
        #    Before addition, |N(u) ∩ N(w)| was some value. After, it increases by 1 if v in N(w) and v not already in N(u)∩N(w).
        #    Since v was not in N(u) before (we're adding u-v), v is in N(u)∩N(w) after iff v in N(w).
        #    So if v in N(w), the count increases by 1.
        #    We need the new count < 3, so old count must be < 2 if v in N(w).
        #    Similarly for pair (v, w).
        
        # Let's just do the direct check for efficiency in small n=20.
        # For each pair (a,b), compute common neighbors.
        # This is O(n^2 * n) = 20^3 = 8000 operations, which is fast.
        
        # But we can optimize: only check pairs where the count might reach 3.
        
        # For simplicity and correctness, let's check all pairs.
        for a in range(n):
            for b in range(a+1, n):
                # Count common neighbors
                common = 0
                for c in range(n):
                    if c != a and c != b and adj[a][c] and adj[b][c]:
                        common += 1
                        if common >= 3:
                            return False
        return True
    
    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [[False]*n for _ in range(n)]
        edges = []
        
        # Order of adding edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            # Try adding edge
            # Check if it violates K2,3
            # Fast check: 
            # 1. Check pair (u,v): common neighbors
            common_uv = sum(1 for c in range(n) if c != u and c != v and adj[u][c] and adj[v][c])
            if common_uv >= 3:
                continue
            
            # 2. Check pairs (u,w) for all w != u,v
            valid = True
            for w in range(n):
                if w == u or w == v:
                    continue
                # Pair (u, w): 
                # Common neighbors before: adj[u][c] and adj[w][c]
                # After adding (u,v): v is in N(u). If adj[w][v], then v is new common neighbor.
                # So new common count = old common count + (1 if adj[w][v] else 0)
                old_common = sum(1 for c in range(n) if c != u and c != w and adj[u][c] and adj[w][c])
                new_common = old_common + (1 if adj[w][v] else 0)
                if new_common >= 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # 3. Check pairs (v,w) for all w != u,v
            for w in range(n):
                if w == u or w == v:
                    continue
                # Pair (v, w):
                # Common neighbors before: adj[v][c] and adj[w][c]
                # After adding (u,v): u is in N(v). If adj[w][u], then u is new common neighbor.
                old_common = sum(1 for c in range(n) if c != v and c != w and adj[v][c] and adj[w][c])
                new_common = old_common + (1 if adj[w][u] else 0)
                if new_common >= 3:
                    valid = False
                    break
            
            if valid:
                adj[u][v] = True
                adj[v][u] = True
                edges.append([u, v])
        
        return edges
    
    # Try multiple random restarts
    for seed in range(100):
        if time.time() > deadline:
            break
        edges = greedy_build(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # If we haven't found anything good, use a known construction
    # For now, return the best found
    if best_edges:
        return best_edges
    
    # Fallback: five disjoint K4s
    edges = []
    for k in range(5):
        edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    return edges
# EVOLVE-BLOCK-END
