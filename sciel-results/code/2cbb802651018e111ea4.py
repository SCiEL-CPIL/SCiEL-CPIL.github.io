import time
import random
from itertools import combinations

N = 18

def construct():
    deadline = time.time() + 110  # 110s budget, 40s safety margin (limit 150s)
    
    # We work with the complement graph H (G_bar).
    # G is K4-free <=> alpha(H) <= 3.
    # Largest triangle-free induced subgraph in G <=> omega(H).
    # Goal: Minimize omega(H) subject to alpha(H) <= 3.
    
    # State: H as a list of bitmasks adj[i]
    # adj[i] has bit j set if (i,j) is an edge in H.
    
    def get_non_adj(adj):
        """Return list of bitmasks for non-adjacency in H (including self? No, exclude self for independence check logic ease, or handle carefully)."""
        # non_adj[i] should contain j if i!=j and not adj[i][j].
        non_adj = [0] * N
        full_mask = (1 << N) - 1
        for i in range(N):
            # non-neighbors in H are neighbors in G.
            # non_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)
            non_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)
        return non_adj

    def has_independent_set_4(adj):
        """Check if H has an independent set of size 4."""
        non_adj = get_non_adj(adj)
        for a in range(N):
            # Candidates for b are non-neighbors of a, greater than a
            # We can just iterate all b > a in non_adj[a]
            mask_b = non_adj[a] & ((1 << N) - 1) & ~((1 << (a + 1)) - 1) # b > a
            while mask_b:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                
                # Candidates for c are non-neighbors of a AND b, greater than b
                # c must be in non_adj[a] AND non_adj[b]
                mask_c = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1) # c > b
                while mask_c:
                    c = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    
                    # Candidates for d are non-neighbors of a, b, c, greater than c
                    # d must be in non_adj[a] AND non_adj[b] AND non_adj[c]
                    mask_d = non_adj[a] & non_adj[b] & non_adj[c] & ~((1 << (c + 1)) - 1) # d > c
                    if mask_d:
                        return True
        return False

    def has_k4_in_H(adj):
        """Check if H has a clique of size 4 (K4)."""
        # Equivalent to checking if G has an independent set of size 4.
        # But we already have has_independent_set_4 for H.
        # Wait. K4 in H is a clique in H.
        # Independent set in H is a clique in G.
        # We need to check for K4 in H.
        # This is the same as checking for independent set of size 4 in G.
        # But G is the complement of H.
        # So we can just use the same logic on the complement?
        # No, let's implement a direct K4 check on H using adj.
        
        for a in range(N):
            # Neighbors of a in H
            mask_b = adj[a] & ~((1 << (a + 1)) - 1) # b > a
            while mask_b:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                
                # Common neighbors of a and b in H
                # c must be neighbor of a and b, and c > b
                mask_c = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                while mask_c:
                    c = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    
                    # d must be neighbor of a, b, c, and d > c
                    # d in adj[a] & adj[b] & adj[c]
                    mask_d = adj[a] & adj[b] & adj[c] & ~((1 << (c + 1)) - 1)
                    if mask_d:
                        return True
        return False

    def calculate_omega_H(adj):
        """Calculate max clique size in H."""
        # Since N=18, we can just check 4, 5, ...
        # If no K4, omega <= 3.
        # We can just return 3 if no K4, else find max.
        # For speed, we assume we are looking for 3 vs 4+.
        # If has_k4_in_H is False, return 3 (assuming omega >= 3, which is true for non-trivial graphs).
        # Actually, if H is empty, omega=1. But alpha=18.
        # If H is complete, omega=18, alpha=1.
        # We will track the actual max clique if it's > 3.
        
        if not has_k4_in_H(adj):
            return 3 # Or 2, but 3 is better for score? No, we want to MINIMIZE. 
                     # If omega is 2, score is 2. If omega is 3, score is 3.
                     # So we should distinguish 2 and 3.
                     # But alpha <= 3 implies H is not too sparse.
                     # Let's just compute it properly if it's small.
            
            # Check for K3
            has_k3 = False
            for a in range(N):
                mask_b = adj[a] & ~((1 << (a + 1)) - 1)
                while mask_b:
                    b = (mask_b & -mask_b).bit_length() - 1
                    mask_b &= mask_b - 1
                    if adj[a] & adj[b] & (1 << b): # Wait, b is in adj[a]
                        # Check if any common neighbor
                        if adj[a] & adj[b]:
                            has_k3 = True
                            break
                if has_k3: break
            
            if has_k3:
                return 3
            else:
                # Max clique is 2?
                if any(adj[i] for i in range(N)):
                    return 2
                else:
                    return 1
        else:
            # Has K4. Find max clique.
            # Simple B&B
            best = [4]
            
            def expand(clique, candidates):
                if not candidates:
                    return
                if len(clique) > best[0]:
                    best[0] = len(clique)
                    return
                
                # Upper bound
                if len(clique) + len(candidates) <= best[0]:
                    return
                
                v = next(iter(candidates))
                
                # Include v
                new_cand = candidates - {v}
                # New candidates must be neighbors of v
                new_cand = {u for u in new_cand if (adj[v] >> u) & 1}
                expand(clique | {v}, new_cand)
                
                # Exclude v
                expand(clique, candidates - {v})
                
            expand(set(), set(range(N)))
            return best[0]

    def score_H(adj):
        if has_independent_set_4(adj):
            return 100 # Invalid
        return calculate_omega_H(adj)

    # Initialization
    # Start with a random graph
    random.seed(42)
    
    best_score = 100
    best_adj = None
    current_adj = [0] * N
    
    # Try to build a valid H
    # Start with a dense graph to ensure alpha <= 3
    # Random edges with p=0.5
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < 0.5:
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
                
    # Enforce alpha <= 3
    # While alpha > 3, add edges to break independent sets
    # This is a repair step
    for _ in range(100): # Max 100 repairs
        if not has_independent_set_4(current_adj):
            break
        # Find an independent set of size 4
        # Reuse logic from has_independent_set_4 to find one
        non_adj = get_non_adj(current_adj)
        found = None
        for a in range(N):
            mask_b = non_adj[a] & ~((1 << (a + 1)) - 1)
            while mask_b and not found:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                mask_c = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
                while mask_c and not found:
                    c = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    mask_d = non_adj[a] & non_adj[b] & non_adj[c] & ~((1 << (c + 1)) - 1)
                    if mask_d:
                        d = (mask_d & -mask_d).bit_length() - 1
                        found = (a, b, c, d)
        
        if found:
            a, b, c, d = found
            # Add an edge between two of them to break the independent set
            # Randomly pick a pair
            pair = random.choice([(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)])
            u, v = pair
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
    if not has_independent_set_4(current_adj):
        best_score = calculate_omega_H(current_adj)
        best_adj = current_adj[:]
    
    # Local Search
    t_start = time.time()
    
    while time.time() - t_start < 100:
        # Pick random edge to flip
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v: continue
        
        old_val = (current_adj[u] >> v) & 1
        # Flip
        if old_val:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        else:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
        # Check validity
        if has_independent_set_4(current_adj):
            # Revert
            if old_val:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            continue
            
        # Calculate score
        new_score = calculate_omega_H(current_adj)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = current_adj[:]
            # If we reached 3, we can stop? 
            # 3 is the theoretical minimum for this problem class? 
            # If we can get 2, that would be better.
            # But alpha <= 3 and omega <= 2 implies triangle-free and no indep set 4.
            # If omega <= 2, H is triangle-free.
            # If H is triangle-free, alpha(H) can be large.
            # We need alpha <= 3.
            # So H must be a triangle-free graph with alpha <= 3.
            # By Ramsey R(3,3)=6, any graph on 6 vertices has triangle or indep 3.
            # If H is triangle-free, it must have alpha >= 3?
            # Actually, if H is triangle-free, max clique is 2.
            # Can we have alpha <= 3?
            # Yes, e.g. C5 has alpha 2.
            # So a union of C5s?
            # 18 vertices. 3 * C5 = 15. Plus 3 vertices?
            # If we can find such a graph, score is 2.
            # So we should keep searching for 2.
            
        # Acceptance: 
        # If new_score <= best_score, keep.
        # If new_score > best_score, revert? 
        # Simple hill climbing: only accept if better or equal.
        if new_score > best_score:
            # Revert
            if old_val:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        # If equal, keep (to explore plateaus)
        
        if best_score == 2:
            break

    # Convert best_adj (H) to G (complement)
    # G edges are non-edges of H
    edges = []
    if best_adj:
        for i in range(N):
            for j in range(i+1, N):
                if not ((best_adj[i] >> j) & 1):
                    edges.append([i, j])
    else:
        # Fallback: Random K4-free graph
        # Just return empty graph? No, empty graph has alpha 18.
        # Return complete graph? No, K4.
        # Return a random valid one.
        # For safety, return a known valid structure if search failed
        # e.g. 3 parts of 6, complete 3-partite (K6,6,6)
        # This G has alpha 6 (one from each part).
        # Triangle-free subgraph? 
        # In K6,6,6, any 2 vertices from different parts form an edge.
        # Triangle-free means no 3 pairwise connected.
        # In K6,6,6, a triangle must have vertices from 3 different parts.
        # So a set of 2 vertices from different parts is triangle-free.
        # A set of 2 vertices from same part is independent (no edge), so triangle-free.
        # Max triangle-free?
        # Take all 6 from part 1. Independent. Size 6.
        # So score 6.
        
        # Let's just generate a random K4-free graph if best_adj is None
        adj_G = [[0]*N for _ in range(N)]
        # Random edges
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    adj_G[i][j] = 1
                    adj_G[j][i] = 1
        # Remove K4s
        # Simple: just return the edges of K6,6,6
        parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.append([a, b])
                        
    return edges
