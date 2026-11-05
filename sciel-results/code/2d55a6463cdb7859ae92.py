"""Combined approach: C9 union baseline + bitmask K4-free checking + local search."""
import time
import random
import math
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 18
    full_mask = (1 << N) - 1
    
    # Helper: Check if a bitmask represents an independent set in graph defined by adj (list of bitmasks)
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True
    
    # Helper: Check if adding edge (u,v) keeps graph K4-free
    # A K4 is formed if u,v have common neighbors w,x such that w-x is an edge.
    # So common neighbors of u and v must form an independent set.
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True  # Already an edge
        common = adj[u] & adj[v]
        return is_independent(common, adj)
    
    # Helper: Check if removing edge (u,v) is safe (always safe for K4-free)
    # Removing edges never creates a K4, so always allowed
    
    # Exact B&B for largest triangle-free induced subgraph
    # A triangle-free induced subgraph S means: for any v in S, N(v) ∩ S is an independent set
    # Equivalently: no three vertices in S form a triangle
    def largest_triangle_free(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic to get a good initial lower bound
        # Try multiple random orders
        for _ in range(5):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                # Check if v can be added: N(v) ∩ S must be independent
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = bin(S).count('1')
            if sz > best[0]:
                best[0] = sz
        
        if best[0] == N:
            return N
        
        # Branch and bound
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            
            # Prune 1: Even adding all candidates can't beat best
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Prune 2: Remove candidates that cannot be added to current S
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb
            
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return
            
            C_mask = new_C
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            # Branch: pick a vertex from C_mask
            # Use the first set bit for determinism, or random for diversity
            lsb = C_mask & -C_mask
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            bnb(S_mask | lsb, C_mask ^ lsb)
            
            # Branch 2: Exclude v
            bnb(S_mask, C_mask ^ lsb)
        
        bnb(0, full_mask)
        return best[0]
    
    # Get edge list from adjacency bitmasks
    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Create C9 ∪ C9 baseline
    # Two disjoint 9-cycles: vertices 0-8 form one cycle, 9-17 form another
    def make_c9_union_adj():
        adj = [0] * N
        # First cycle: 0-1-2-3-4-5-6-7-8-0
        for i in range(9):
            a = i
            b = (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        # Second cycle: 9-10-11-12-13-14-15-16-17-9
        for i in range(9):
            a = 9 + i
            b = 9 + (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        return adj
    
    # Verify K4-free (should always be true for C9∪C9, but check for safety)
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j + 1, N):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    for l in range(k + 1, N):
                        if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                            return False
        return True
    
    random.seed(42)
    
    # Start with C9 ∪ C9
    current_adj = make_c9_union_adj()
    current_score = largest_triangle_free(current_adj, deadline - 5.0)
    best_adj = current_adj[:]
    best_score = current_score
    
    # Local search: try edge flips (add or remove)
    def local_search(adj, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible single edge flips
            for i in range(N):
                if time.time() > deadline:
                    return best_score
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        return best_score
                    old_val = (adj[i] >> j) & 1
                    
                    if old_val == 0:
                        # Try adding edge
                        if not can_add_edge(adj, i, j):
                            continue
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        # Try removing edge
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    
                    score = largest_triangle_free(adj, deadline - 5.0)
                    
                    if score < best_score:
                        best_score = score
                        improved = True
                    else:
                        # Revert
                        if old_val == 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        
        return best_score
    
    # Run local search from C9 ∪ C9
    best_score = local_search(current_adj, best_score, deadline)
    if best_score < current_score:
        best_adj = current_adj[:]
    
    # Random restarts with perturbation
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate a random K4-free graph by greedy edge addition
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Verify K4-free (should be, but just in case)
        if not is_k4_free(adj):
            continue
        
        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            # Local search from this point
            best_score = local_search(adj, best_score, deadline)
            if best_score < score:
                best_adj = adj[:]
        
        # Also try perturbing the C9 union
        if attempts % 3 == 0:
            adj2 = make_c9_union_adj()
            # Randomly add some edges while maintaining K4-free
            num_adds = random.randint(3, 15)
            for _ in range(num_adds):
                u, v = random.sample(range(N), 2)
                if can_add_edge(adj2, u, v):
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
            
            if is_k4_free(adj2):
                score2 = largest_triangle_free(adj2, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_adj = adj2[:]
                    best_score = local_search(adj2, best_score, deadline)
                    if best_score < score2:
                        best_adj = adj2[:]
        
        # Also try perturbing by removing edges from current best
        if attempts % 5 == 0 and time.time() < deadline - 15.0:
            adj3 = best_adj[:]
            num_removals = random.randint(1, 5)
            current_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj3[i] >> j) & 1]
            if current_edges:
                for _ in range(min(num_removals, len(current_edges))):
                    u, v = random.choice(current_edges)
                    adj3[u] &= ~(1 << v)
                    adj3[v] &= ~(1 << u)
                    current_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj3[i] >> j) & 1]
                    if not current_edges:
                        break
                
                score3 = largest_triangle_free(adj3, deadline - 5.0)
                if score3 < best_score:
                    best_score = score3
                    best_adj = adj3[:]
    
    return get_edge_list(best_adj)

# EVOLVE-BLOCK-END
