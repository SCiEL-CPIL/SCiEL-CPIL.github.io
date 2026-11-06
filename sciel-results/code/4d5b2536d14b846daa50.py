import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)

    def count_violations(n, adj):
        """
        Efficiently count the number of 4-sets without a common dominator.
        adj is an n x n boolean matrix (1 if i->j).
        """
        # Precompute out-masks for fast bit operations
        # For n <= 90, we can use Python integers as bitsets
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        viol_count = 0
        # Iterate through all 4-sets
        # Optimization: If we find a dominator, we stop checking others for that set
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            found = False
            # Check if any vertex dominates this quad
            # A vertex v dominates if out_mask[v] contains all bits in quad_mask
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                viol_count += 1
        return viol_count

    def local_search(n, adj, max_iters=50000):
        """
        Local search to minimize violations.
        Uses delta cost calculation for edge flips.
        """
        # Precompute out-masks
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        current_cost = count_violations(n, adj)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        if current_cost == 0:
            return best_adj, 0

        # Precompute quad masks and indices for faster delta calc?
        # For n up to ~60, C(n,4) is manageable. For n=90, it's ~2.5M.
        # Let's stick to a simpler delta calc:
        # Flip edge (u, v).
        # Affected quads are those containing u or v.
        # We need to know if u or v were dominators.
        
        # To speed up, we can precompute for each vertex v, the list of quads it dominates?
        # That's too much memory.
        
        # Alternative: Random edge flip, recompute cost for affected quads only.
        # A quad is affected if it contains u or v.
        # Number of such quads is ~ 4 * n^2. For n=60, ~14400 quads per flip.
        # Recomputing dominance for 14400 quads takes ~ 14400 * n operations.
        # 14400 * 60 = 864,000 ops. ~0.5ms in Python? Maybe too slow for 50k iters.
        
        # Let's use a smarter heuristic:
        # 1. Identify "bad" quads (violating ones).
        # 2. For each bad quad, try to flip an edge to create a dominator.
        
        bad_quads = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                bad_quads.append(quad)
                
        if not bad_quads:
            return best_adj, 0

        T = 1.0
        cooling = 0.9995
        no_improve = 0
        
        for i in range(max_iters):
            if time_left() < 5:
                break
                
            # Pick a random bad quad
            quad = random.choice(bad_quads)
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            # Try to fix this quad by flipping an edge involving one of its vertices
            # or an edge from a potential dominator to one of its vertices.
            # Potential dominators are vertices v such that v->3 of the 4 vertices.
            
            candidates = []
            for v in range(n):
                if v in quad:
                    continue
                # Count how many of a,b,c,d are dominated by v
                cnt = 0
                if adj[v][a]: cnt += 1
                if adj[v][b]: cnt += 1
                if adj[v][c]: cnt += 1
                if adj[v][d]: cnt += 1
                
                if cnt == 3:
                    # v dominates 3, needs to dominate the 4th.
                    # The 4th is the one v does NOT point to.
                    # If we flip the edge from the 4th vertex to v? No, v needs to point to the 4th.
                    # So we need to flip edge (x, v) where x is the 4th vertex, from x->v to v->x.
                    # Wait, if v does not point to x, then x->v.
                    # We want v->x. So we flip (x, v).
                    # This changes out_mask of v and x.
                    # It might break other quads.
                    
                    # Identify x
                    if not adj[v][a]: x = a
                    elif not adj[v][b]: x = b
                    elif not adj[v][c]: x = c
                    else: x = d
                    
                    # Flip edge (x, v)
                    # Current: x -> v (adj[x][v]=1, adj[v][x]=0)
                    # Target: v -> x (adj[v][x]=1, adj[x][v]=0)
                    
                    # Calculate delta cost?
                    # This is complex. Let's just do a random walk on edges if stuck.
                    pass
            
            # Simpler: Random edge flip with acceptance based on full cost re-evaluation of affected quads
            # But full cost re-eval is slow.
            
            # Let's try a different approach:
            # Greedy: Find an edge flip that reduces cost.
            # Sample random edges.
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Calculate delta
            # Affected quads: those containing u or v.
            # We need to know which quads are currently violating and which will be.
            
            # This is getting too complex for a quick script.
            # Let's rely on the constructive initialization being good enough,
            # and use a simple "repair" phase:
            # If a quad is bad, flip a random edge in it.
            
            if bad_quads:
                quad = random.choice(bad_quads)
                a, b, c, d = quad
                # Pick two vertices in the quad to flip
                pair = random.sample(list(quad), 2)
                x, y = pair
                # Flip edge (x, y)
                adj[x][y] = 1 - adj[x][y]
                adj[y][x] = 1 - adj[x][y]
                
                # Update out_masks
                out_mask[x] ^= (1 << y)
                out_mask[y] ^= (1 << x)
                
                # Update bad_quads list?
                # This is expensive. Let's just recompute cost periodically.
                
                if i % 100 == 0:
                    current_cost = count_violations(n, adj)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = [row[:] for row in adj]
                        no_improve = 0
                        # Rebuild bad_quads
                        bad_quads = []
                        for q in combinations(range(n), 4):
                            a, b, c, d = q
                            qm = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            found = False
                            for v in range(n):
                                if (out_mask[v] & qm) == qm:
                                    found = True
                                    break
                            if not found:
                                bad_quads.append(q)
                        if not bad_quads:
                            break
                    else:
                        no_improve += 1
                        if no_improve > 5000:
                            # Restart with random perturbation
                            for _ in range(10):
                                u = random.randint(0, n-1)
                                v = random.randint(0, n-1)
                                if u != v:
                                    adj[u][v] = 1 - adj[u][v]
                                    adj[v][u] = 1 - adj[v][u]
                                    out_mask[u] ^= (1 << v)
                                    out_mask[v] ^= (1 << u)
                            no_improve = 0
                            current_cost = count_violations(n, adj)
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_adj = [row[:] for row in adj]
            
            T *= cooling
            
        return best_adj, best_cost

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try Constructive Heuristic for smaller n
    # Start from lower bound 31, go up to 82
    for n in range(31, 83):
        if time_left() < 10:
            break
            
        # Try a few seeds
        for seed in range(5):
            if time_left() < 5:
                break
                
            random.seed(seed)
            
            # Constructive Initialization
            # Split into Hubs and Non-Hubs
            # Let k = number of hubs.
            # We want Hubs to dominate Non-Hubs.
            # If we have k hubs, and n-k non-hubs.
            # Each non-hub needs to be dominated by at least one hub?
            # No, the condition is on 4-sets.
            # If a 4-set has 4 non-hubs, it must be dominated by a hub (since non-hubs don't dominate each other? Not necessarily).
            # Let's assume Hubs form a "dominating set" of the tournament.
            # i.e., for every non-hub v, there is a hub h such that h->v.
            # This doesn't guarantee 4-sets of non-hubs are dominated.
            
            # Better: Random tournament, but bias edges towards a random subset of "hubs".
            # Let's just use random init + local search, but with a tighter loop.
            
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                    else:
                        adj[j][i] = 1
                        
            best_adj, best_cost = local_search(n, adj, max_iters=20000)
            
            if best_cost == 0:
                best_n = n
                best_orient = adj_to_orient(n, best_adj)
                break
                
        if best_n < 83:
            break
            
    return {'n': best_n, 'orientation': best_orient}
