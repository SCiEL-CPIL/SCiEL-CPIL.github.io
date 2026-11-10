import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Finite-field-based construction: use Z_23 (prime) and map 20 vertices to a subset
    # Paley graph on Z_p: edge (u,v) iff (u-v) is a quadratic residue mod p
    # For p=23, the Paley graph has 23 vertices, each with degree 11
    # Common neighbors in Paley graphs are bounded
    
    # Quadratic residues mod 23
    QR_23 = set()
    for i in range(1, 23):
        QR_23.add((i*i) % 23)
    
    # Map 20 vertices to 20 of the 23 field elements (skip 0, and two others for variety)
    # Try different subsets to find good ones
    def make_paley_subset(skip_elements):
        """Create adjacency from Paley graph on Z_23 restricted to given subset."""
        elements = [x for x in range(23) if x not in skip_elements][:20]
        adj = [0]*n
        for i in range(20):
            for j in range(i+1, 20):
                diff = (elements[i] - elements[j]) % 23
                if diff in QR_23:
                    adj[i] |= 1<<j
                    adj[j] |= 1<<i
        return adj
    
    def build_common_and_bad(adj):
        """Build common-neighbor counts and bad bitmask from scratch."""
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u]>>v)&1: continue
                c = 0
                for w in range(n):
                    if w==u or w==v: continue
                    if (adj[u]>>w)&1 and (adj[v]>>w)&1:
                        c += 1
                common[u][v] = c
                common[v][u] = c
        
        bad = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u]>>v)&1: continue
                feasible = True
                if common[u][v] >= 3:
                    feasible = False
                else:
                    for w in range(n):
                        if w==u or w==v: continue
                        if (adj[v]>>w)&1 and common[u][w] >= 2:
                            feasible = False; break
                        if (adj[u]>>w)&1 and common[v][w] >= 2:
                            feasible = False; break
                if not feasible:
                    bad[u] |= 1<<v
                    bad[v] |= 1<<u
        return common, bad
    
    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj)//2
    
    def do_add(adj, common, bad, u, v):
        """Add edge (u,v) and update bad for all affected pairs."""
        adj[u] |= 1<<v; adj[v] |= 1<<u
        
        au = adj[u] & ~(1<<v)
        av = adj[v] & ~(1<<u)
        for w in range(n):
            if (au>>w)&1:
                common[v][w]+=1; common[w][v]+=1
            if (av>>w)&1:
                common[u][w]+=1; common[w][u]+=1
        
        affected = set()
        for x in range(n):
            if x == u or x == v: continue
            if (adj[u]>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (adj[v]>>y)&1:
                        affected.add((min(x,y), max(x,y)))
            if (adj[v]>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (adj[u]>>y)&1:
                        affected.add((min(x,y), max(x,y)))
        
        for x in range(n):
            if x == u or x == v: continue
            if not ((adj[u]>>x)&1):
                affected.add((min(u,x), max(u,x)))
            if not ((adj[v]>>x)&1):
                affected.add((min(v,x), max(v,x)))
        
        for (a,b) in affected:
            if (adj[a]>>b)&1:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
                continue
            feasible = True
            if common[a][b] >= 3:
                feasible = False
            else:
                for w in range(n):
                    if w == a or w == b: continue
                    if (adj[b]>>w)&1 and common[a][w] >= 2:
                        feasible = False; break
                    if (adj[a]>>w)&1 and common[b][w] >= 2:
                        feasible = False; break
            if not feasible:
                bad[a] |= 1<<b
                bad[b] |= 1<<a
            else:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
    
    def do_remove(adj, common, bad, u, v):
        """Remove edge (u,v) and update bad for all affected pairs."""
        au = adj[u] & ~(1<<v)
        av = adj[v] & ~(1<<u)
        
        for w in range(n):
            if (au>>w)&1:
                common[v][w]-=1; common[w][v]-=1
            if (av>>w)&1:
                common[u][w]-=1; common[w][u]-=1
        
        adj[u] &= ~(1<<v); adj[v] &= ~(1<<u)
        
        affected = set()
        for x in range(n):
            if x == u or x == v: continue
            if (au>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (av>>y)&1:
                        affected.add((min(x,y), max(x,y)))
            if (av>>x)&1:
                for y in range(n):
                    if y == x or y == u or y == v: continue
                    if (au>>y)&1:
                        affected.add((min(x,y), max(x,y)))
        
        for x in range(n):
            if x == u or x == v: continue
            if not ((au>>x)&1):
                affected.add((min(u,x), max(u,x)))
            if not ((av>>x)&1):
                affected.add((min(v,x), max(v,x)))
        
        affected.add((min(u,v), max(u,v)))
        
        for (a,b) in affected:
            if (adj[a]>>b)&1:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
                continue
            feasible = True
            if common[a][b] >= 3:
                feasible = False
            else:
                for w in range(n):
                    if w == a or w == b: continue
                    if (adj[b]>>w)&1 and common[a][w] >= 2:
                        feasible = False; break
                    if (adj[a]>>w)&1 and common[b][w] >= 2:
                        feasible = False; break
            if not feasible:
                bad[a] |= 1<<b
                bad[b] |= 1<<a
            else:
                bad[a] &= ~(1<<b)
                bad[b] &= ~(1<<a)
    
    def local_improve(adj, common, bad):
        changed = True
        while changed:
            changed = False
            for (u,v) in all_edges:
                if not ((adj[u]>>v)&1) and not ((bad[u]>>v)&1):
                    do_add(adj, common, bad, u, v)
                    changed = True
        return adj, common, bad
    
    def perturbation_search(adj, common, bad, rng, iters=3):
        """Remove some random edges, then re-optimize."""
        adj2 = adj[:]
        common2 = [row[:] for row in common]
        bad2 = bad[:]
        
        edges_now = [(u,v) for (u,v) in all_edges if (adj2[u]>>v)&1]
        if not edges_now:
            return adj2, common2, bad2
        
        num_remove = min(3, len(edges_now))
        for _ in range(num_remove):
            if not edges_now:
                break
            u,v = rng.choice(edges_now)
            do_remove(adj2, common2, bad2, u, v)
            edges_now = [(a,b) for (a,b) in all_edges if (adj2[a]>>b)&1]
        
        adj2, common2, bad2 = local_improve(adj2, common2, bad2)
        return adj2, common2, bad2
    
    best = None
    best_count = -1
    rng = random.Random(42)
    
    # Phase 1: Try multiple Paley-based initializations with different vertex subsets
    # Skip different pairs of elements from Z_23 to get diverse starting points
    skip_pairs = []
    for a in range(23):
        for b in range(a+1, 23):
            skip_pairs.append((a, b))
    rng.shuffle(skip_pairs)
    
    # Also include the old 5xK4 seed for comparison
    def seed_graph_5k4():
        adj = [0]*n
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    u = 4*k+a; v = 4*k+b
                    adj[u] |= 1<<v; adj[v] |= 1<<u
        return adj
    
    # Generate diverse initializations
    initializations = []
    
    # 5xK4 seed
    adj0 = seed_graph_5k4()
    common0, bad0 = build_common_and_bad(adj0)
    initializations.append((adj0, common0, bad0))
    
    # Paley-based seeds (try first 30 skip pairs)
    for skip_pair in skip_pairs[:30]:
        if time.time() > deadline - 20.0:
            break
        adj_p = make_paley_subset(set(skip_pair))
        common_p, bad_p = build_common_and_bad(adj_p)
        initializations.append((adj_p, common_p, bad_p))
    
    # Random seeds
    for _ in range(20):
        if time.time() > deadline - 20.0:
            break
        adj_r = [0]*n
        for (u,v) in all_edges:
            if rng.random() < 0.4:
                adj_r[u] |= 1<<v
                adj_r[v] |= 1<<u
        common_r, bad_r = build_common_and_bad(adj_r)
        initializations.append((adj_r, common_r, bad_r))
    
    # Phase 2: Optimize each initialization
    for adj_init, common_init, bad_init in initializations:
        if time.time() > deadline - 15.0:
            break
        
        adj_c = adj_init[:]
        common_c = [row[:] for row in common_init]
        bad_c = bad_init[:]
        
        adj_c, common_c, bad_c = local_improve(adj_c, common_c, bad_c)
        cnt = edge_count(adj_c)
        if cnt > best_count:
            best_count = cnt
            best = (adj_c[:], [row[:] for row in common_c], bad_c[:])
    
    # Phase 3: Perturbation search from best solution
    if best is not None:
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            
            adj_p, common_p, bad_p = perturbation_search(
                best[0], best[1], best[2], rng, iters=1
            )
            cnt_p = edge_count(adj_p)
            if cnt_p > best_count:
                best_count = cnt_p
                best = (adj_p[:], [row[:] for row in common_p], bad_p[:])
    
    # Phase 4: More aggressive perturbation - remove more edges
    if best is not None:
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            adj_a = best[0][:]
            common_a = [row[:] for row in best[1]]
            bad_a = best[2][:]
            
            # Remove 5-8 random edges
            edges_now = [(u,v) for (u,v) in all_edges if (adj_a[u]>>v)&1]
            num_remove = min(rng.randint(5, 8), len(edges_now))
            for _ in range(num_remove):
                if not edges_now:
                    break
                u,v = rng.choice(edges_now)
                do_remove(adj_a, common_a, bad_a, u, v)
                edges_now = [(a,b) for (a,b) in all_edges if (adj_a[a]>>b)&1]
            
            adj_a, common_a, bad_a = local_improve(adj_a, common_a, bad_a)
            cnt_a = edge_count(adj_a)
            if cnt_a > best_count:
                best_count = cnt_a
                best = (adj_a[:], [row[:] for row in common_a], bad_a[:])
    
    # Phase 5: Final local improvement passes
    if best is not None:
        adj_f = best[0][:]
        common_f = [row[:] for row in best[1]]
        bad_f = best[2][:]
        adj_f, common_f, bad_f = local_improve(adj_f, common_f, bad_f)
        cnt_f = edge_count(adj_f)
        if cnt_f > best_count:
            best_count = cnt_f
            best = (adj_f[:], [row[:] for row in common_f], bad_f[:])
    
    # Fallback
    if best is None:
        adj = seed_graph_5k4()
    else:
        adj = best[0]
    
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u]>>v)&1:
                res.append([u,v])
    return res
