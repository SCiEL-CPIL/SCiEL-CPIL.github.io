import time
import random
import itertools

N = 15

def _degrees(inc):
    d = [0]*N
    for i in range(N):
        d[i] = len(inc[i])
    return d

def _valid(inc, deg):
    # violation: exists edge (u,v) and s in N(u)\{v}, t in N(v)\{u}, s!=t, deg(s)==deg(t)
    for u in range(N):
        for v in inc[u]:
            if v < u:
                continue
            # u<v, edge exists
            du = deg[u]; dv = deg[v]
            # s in N(u) not v, t in N(v) not u, s!=t
            # collect degrees of N(u)\{v} and N(v)\{u}
            su = [deg[x] for x in inc[u] if x != v]
            tv = [deg[x] for x in inc[v] if x != u]
            # need s!=t and deg equal
            # check by degree value
            set_u = set(su)
            for x in inc[u]:
                if x == v:
                    continue
                # x is candidate s
                # need t in inc[v], t!=u, t!=x, deg[t]==deg[x]
                for y in inc[v]:
                    if y == u or y == x:
                        continue
                    if deg[y] == deg[x]:
                        return False
    return True

def _violation_count(inc, deg):
    cnt = 0
    for u in range(N):
        for v in inc[u]:
            if v < u:
                continue
            for x in inc[u]:
                if x == v:
                    continue
                for y in inc[v]:
                    if y == u or y == x:
                        continue
                if deg[y] == deg[x]:
                    cnt += 1
    return cnt

def _edge_count(inc):
    return sum(len(inc[i]) for i in range(N)) // 2

def _edges_list(inc):
    out = []
    for u in range(N):
        for v in inc[u]:
            if v > u:
                out.append([u, v])
    return out

def _add_edge(inc, u, v):
    inc[u].append(v)
    inc[v].append(u)

def _remove_edge(inc, u, v):
    inc[u].remove(v)
    inc[v].remove(u)

def _random_bipartite():
    # random partition into 2+ parts, each a complete bipartite
    # assign each vertex a color; within same color, no edges; across colors, all edges
    # but we need bipartite per component. Let's do: split into groups, each group is K_{a,b}
    inc = [set() for _ in range(N)]
    # random number of components
    # assign vertices to components
    comp = []
    for i in range(N):
        comp.append(i)  # start each own
    # random merges
    for _ in range(6):
        a = random.randrange(N)
        b = random.randrange(N)
        if a == b:
            continue
        # merge b's group into a's group
        for i in range(N):
            if comp[i] == b:
                comp[i] = a
    # now for each group, split into two sides randomly
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(N):
        groups[comp[i]].append(i)
    for g, verts in groups.items():
        if len(verts) < 2:
            continue
        # random split
        random.shuffle(verts)
        half = len(verts)//2
        A = set(verts[:half])
        B = set(verts[half:])
        if not A or not B:
            continue
        for a in A:
            for b in B:
                inc[a].add(b)
                inc[b].add(a)
    return [sorted(s) for s in inc]

def _local_search(inc, deg, max_iters=200000, seed=0):
    rng = random.Random(seed)
    inc = [list(s) for s in inc]
    deg = _degrees(inc)
    cur_vc = _violation_count(inc, deg)
    cur_ec = _edge_count(inc)
    best_inc = [list(s) for s in inc]
    best_ec = cur_ec
    best_vc = cur_vc
    t0 = time.time()
    it = 0
    while it < max_iters:
        it += 1
        if time.time() - t0 > 25:
            break
        if cur_vc == 0:
            # try to add edges
            improved = False
            pairs = list(itertools.combinations(range(N), 2))
            rng.shuffle(pairs)
            for (u, v) in pairs:
                if v in inc[u]:
                    continue
                # try add
                _add_edge(inc, u, v)
                nd = _degrees(inc)
                nvc = _violation_count(inc, nd)
                nec = cur_ec + 1
                if nvc <= cur_vc and nec > cur_ec:
                    cur_vc = nvc
                    cur_ec = nec
                    deg = nd
                    improved = True
                    if nec > best_ec and nvc < best_vc:
                        best_ec = nec
                        best_vc = nvc
                        best_inc = [list(s) for s in inc]
                    if nvc == 0:
                        # keep trying
                        continue
                    break
                else:
                    _remove_edge(inc, u, v)
            if improved and cur_vc == 0:
                continue
            if not improved:
                # stuck, try swap
                pass
        # swap: remove edge (a,b), add edge (c,d)
        # choose random existing edge to remove
        if cur_ec == 0:
            break
        # pick random edge
        u = rng.randrange(N)
        if not inc[u]:
            continue
        v = rng.choice(inc[u])
        # pick random non-edge to add
        c = rng.randrange(N)
        d = rng.randrange(N)
        if c == d:
            continue
        if d in inc[c]:
            continue
        # also ensure we don't remove the same edge we add
        if (c == u and d == v) or (c == v and d == u):
            continue
        # apply
        _remove_edge(inc, u, v)
        _add_edge(inc, c, d)
        nd = _degrees(inc)
        nvc = _violation_count(inc, nd)
        nec = cur_ec  # same edge count
        # accept if fewer violations or (same violations and we like it)
        if nvc < cur_vc or (nvc == cur_vc and rng.random() < 0.1):
            cur_vc = nvc
            deg = nd
            if nvc < best_vc and nec > best_ec:
                best_ec = nec
                best_vc = nvc
                best_inc = [list(s) for s in inc]
        else:
            # revert
            _remove_edge(inc, c, d)
            _add_edge(inc, u, v)
            deg = _degrees(inc)
    # return best
    if best_vc < cur_vc or (best_vc == cur_vc and best_ec >= cur_ec):
        return best_inc, best_ec, best_vc
    return inc, cur_ec, cur_vc

def _initial_construction():
    # Build a strong initial graph: disjoint union of complete bipartite components,
    # then break within-component path violations by removing a matching.
    # Strategy: components K_{a_i,b_i} with distinct degree values.
    # For n=15, try components that partition 15.
    # We want many edges. K_{a,b} has a*b edges, degrees a (b of them) and b (a of them).
    # Within K_{a,b}, a path s-u-v-t with s,t same side => deg equal => violation.
    # To fix, remove edges so that no such path exists.
    # Simplest: make each component a star or a matching? 
    # Actually, let's use components that are complete bipartite but then we must ensure
    # that for any edge (u,v), there's no s in N(u)\v and t in N(v)\u with deg equal.
    # In K_{a,b}, u in A (deg b), v in B (deg a). N(u)\v = A\{u} (all deg b). N(v)\u = B\{u}... 
    # wait v in B, N(v) = A. N(v)\u = A\{u}. So s in A\{u}, t in A\{u}. deg(s)=b, deg(t)=b. 
    # So any edge with a>=2 and b>=2 has violations (since |A\{u}|>=1 and |A\{u}|>=1).
    # So K_{a,b} with a>=2,b>=2 always has violations. 
    # We need to remove edges to break this.
    # 
    # Alternative: use components that are stars K_{1,b}. Then A={center} (deg b), B=leaves (deg 1).
    # Edge (center, leaf): u=center (deg b), v=leaf (deg 1). N(u)\v = other leaves (deg 1). N(v)\u = {} (leaf only connected to center). 
    # So no t. No violation! Stars are valid.
    # So a star K_{1,b} is valid. Multiple stars: degrees are {b_i, 1}. 
    # Path s-u-v-t: within one star, u=center, v=leaf, N(v)\u empty. Across stars, no edges.
    # So disjoint union of stars is valid!
    # To maximize edges with n=15: use stars. A star on k vertices has k-1 edges.
    # Partition 15 into stars: total edges = 15 - (number of stars).
    # To maximize, minimize number of stars. One star K_{1,14} has 14 edges.
    # But we can do better by mixing: a star K_{1,b} uses b+1 vertices, b edges.
    # Actually a single star K_{1,14} gives 14 edges. Can we get more?
    # What about a component that's not a star but still valid?
    # Consider K_{2,2} = C4. Degrees all 2. Edge (u,v): u in A, v in B. N(u)\v = A\{u} (1 vertex, deg 2). N(v)\u = B\{u}... v in B, N(v)=A, N(v)\u=A\{u} (1 vertex deg 2). 
    # s in A\{u}, t in A\{u}, s!=t? |A\{u}|=1, so s=t, not allowed. So no violation! 
    # Wait, s and t must be different. A\{u} has 1 element, so s=t, invalid. So C4 is valid!
    # Let me double check. C4: vertices 0,1,2,3, edges 0-1,1-2,2-3,3-0. deg all 2.
    # Edge (0,1): s in N(0)\{1}={3}, t in N(1)\{0}={2}. s=3,t=2, s!=t, deg(3)=2,deg(2)=2. VIOLATION!
    # I made an error. N(1) = {0,2}. N(1)\{0} = {2}. N(0)={1,3}. N(0)\{1}={3}. So s=3, t=2, both deg 2. Violation.
    # So C4 is NOT valid. My earlier reasoning was wrong: u in A, v in B, N(u)\v = A\{u}, N(v)\u = A\{u} (since v in B, N(v)=A). 
    # So s in A\{u}, t in A\{u}. If |A\{u}|>=2, then s!=t possible. If |A\{u}|=1, s=t, no violation.
    # For C4, A={0,2}, u=0, A\{u}={2}, size 1. So s=t=2, no violation for edge (0,1)? 
    # But I computed N(0)\{1}={3}, not {2}. Let me recompute. 
    # C4 edges: 0-1, 1-2, 2-3, 3-0. So N(0)={1,3}, N(1)={0,2}, N(2)={1,3}, N(3)={2,0}.
    # This is a 4-cycle, but it's bipartite with A={0,2}, B={1,3}. 
    # Edge (0,1): 0 in A, 1 in B. N(0)\{1} = {3}. N(1)\{0} = {2}. s=3 (in B!), t=2 (in A). 
    # deg(3)=2, deg(2)=2. Violation!
    # So my bipartite analysis was wrong: N(v) for v in B is A, but N(1)={0,2} which is A={0,2}. Yes. N(1)\{0}={2}. 
    # And N(0)\{1}: N(0)={1,3}. 1 in B, 3 in B. So N(0)\{1}={3}, which is in B, not A!
    # I see my error: in K_{a,b}, N(u) for u in A is exactly B. So N(0) should be B={1,3}. Yes N(0)={1,3}=B. 
    # So N(0)\{1} = B\{1} = {3}. And {3} is in B, not A. 
    # So s in B\{v}, t in A\{u}. deg(s)=a (since s in B, deg = |A| = a = 2), deg(t)=b (t in A, deg=|B|=b=2). 
    # Both 2. Violation.
    # So in K_{a,b}, for edge (u in A, v in B): s in B\{v} (deg a), t in A\{u} (deg b). 
    # Violation if a==b and |B\{v}|>=1 and |A\{u}|>=1, i.e., a==b and a>=2 and b>=2.
    # If a != b, then deg(s)=a, deg(t)=b, a!=b, no violation!
    # So K_{a,b} with a != b is VALID!
    # Let me verify with K_{1,3} (star): a=1,b=3. Edge (center in A, leaf in B). s in B\{v} (other leaves, deg a=1), t in A\{u} (empty, since |A|=1). No t. Valid.
    # K_{2,3}: a=2,b=3. Edge (u in A, v in B). s in B\{v} (2 vertices, deg 2), t in A\{u} (1 vertex, deg 3). deg(s)=2,deg(t)=3, not equal. Valid!
    # Great. So K_{a,b} with a!=b is valid.
    # Now, disjoint union of K_{a_i,b_i} with a_i != b_i for each: 
    # Cross-component paths: no edges between components, so no path crosses. 
    # So the whole graph is valid if each component K_{a_i,b_i} has a_i != b_i.
    # Maximize total edges = sum a_i*b_i, subject to sum (a_i+b_i) = 15, a_i,b_i >= 1, a_i != b_i.
    # Let's search over partitions.
    # 
    # To maximize sum a_i*b_i with sum(a_i+b_i)=15.
    # For a single component K_{a,b}, a+b=s, a*b max when a=b=s/2, but a!=b.
    # For s=15: a+b=15, a!=b. Max a*b = 7*8=56. So K_{7,8} has 56 edges!
    # But wait, can we do better with multiple components? 
    # sum a_i*b_i with sum s_i=15 where s_i=a_i+b_i.
    # a_i*b_i <= (s_i/2)^2. Sum <= sum (s_i/2)^2. By convexity, max when one s_i=15, rest 0. 
    # So single component K_{7,8} (or K_{8,7}) gives 56 edges, which is the max possible for a single component.
    # But is K_{7,8} valid? a=7,b=8, a!=b. Yes.
    # So 56 edges is achievable!
    # Let me verify K_{7,8} is valid more carefully.
    # A has 7 vertices (deg 8 each), B has 8 vertices (deg 7 each).
    # Edge (u in A, v in B). s in N(u)\{v} = B\{v} (7 vertices, deg 7). t in N(v)\{u} = A\{u} (6 vertices, deg 8).
    # deg(s)=7, deg(t)=8. 7!=8. No violation.
    # What about s in A? s must be in N(u)\{v} = B\{v}, so s in B. t in N(v)\{u}=A\{u}, so t in A.
    # So s always in B (deg 7), t always in A (deg 8). Never equal. Valid!
    # 
    # So K_{7,8} on 15 vertices has 56 edges and is valid.
    # Can we get more than 56? Total possible edges = 15*14/2 = 105.
    # But K_{7,8} is not complete. Can we add edges between A and B? No, K_{7,8} already has all A-B edges.
    # Can we add edges within A or within B? If we add an edge within A, then degrees change.
    # Let's think: start with K_{7,8}, add an edge within A. Now two vertices in A have deg 9, rest 8. 
    # This might create violations. Let's not go there.
    # 
    # Actually, can we do better than 56? Let's consider: the constraint is quite loose.
    # Maybe we can have a denser graph that's not a disjoint union of complete bipartite.
    # But 56 is already quite high. Let's use K_{7,8} as a strong baseline and try to improve with local search.
    #
    # Actually, let me reconsider. Can we have a graph where some vertices have the same degree but no path connects them with equal-degree endpoints?
    # The constraint is: for every path s-u-v-t, deg(s) != deg(t).
    # This is equivalent to: there is no path of length 3 whose endpoints have the same degree.
    # 
    # K_{7,8} gives 56. Let's see if we can get more.
    # Consider making the graph more connected. 
    # 
    # For now, K_{7,8} = 56 edges is a strong starting point.
    inc = [set() for _ in range(N)]
    A = list(range(7))
    B = list(range(7, 15))
    for a in A:
        for b in B:
            inc[a].add(b)
            inc[b].add(a)
    return [sorted(s) for s in inc]

def construct():
    t_start = time.time()
    deadline = 50.0  # safety margin before 60s
    best_edges = 0
    best_inc = None
    
    # Start with K_{7,8}
    inc = _initial_construction()
    deg = _degrees(inc)
    ec = _edge_count(inc)
    vc = _violation_count(inc, deg)
    if vc == 0:
        best_edges = ec
        best_inc = [list(s) for s in inc]
    
    # Local search to improve
    if time.time() - t_start < deadline - 5:
        inc2, ec2, vc2 = _local_search(inc, deg, max_iters=1000000, seed=42)
        if vc2 == 0 and ec2 > best_edges:
            best_edges = ec2
            best_inc = inc2
    
    # Try more random restarts
    seed = 0
    while time.time() - t_start < deadline - 3:
        seed += 1
        inc_r = _random_bipartite()
        deg_r = _degrees(inc_r)
        ec_r = _edge_count(inc_r)
        vc_r = _violation_count(inc_r, deg_r)
        if vc_r == 0:
            # local search from here
            if time.time() - t_start < deadline - 3:
                inc_ls, ec_ls, vc_ls = _local_search(inc_r, deg_r, max_iters=100000, seed=seed)
                if vc_ls == 0 and ec_ls > best_edges:
                    best_edges = ec_ls
                    best_inc = inc_ls
        # Also try local search even if invalid, to repair
        if time.time() - t_start < deadline - 3:
            inc_ls, ec_ls, vc_ls = _local_search(inc_r, deg_r, max_iters=50000, seed=seed+1000)
            if vc_ls == 0 and ec_ls > best_edges:
                best_edges = ec_ls
                best_inc = inc_ls
    
    if best_inc is None:
        # fallback: star
        inc = [set() for _ in range(N)]
        for v in range(1, N):
            inc[0].add(v)
            inc[v].add(0)
        best_inc = [sorted(s) for s in inc]
        best_edges = _edge_count(best_inc)
    
    return _edges_list(best_inc)
