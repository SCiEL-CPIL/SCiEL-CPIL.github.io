def construct():
    t0 = time.time()
    deadline = t0 + 85
    random.seed(12345)
    
    base = [(i, (i+1)%N) for i in range(N)]
    base_set = frozenset(frozenset(e) for e in base)
    all_pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
    candidates = [e for e in all_pairs if frozenset(e) not in base_set]
    
    def adj_of(edge_list):
        adj = [[] for _ in range(N)]
        aset = set()
        for (u,v) in edge_list:
            adj[u].append(v); adj[v].append(u)
            aset.add((u,v)); aset.add((v,u))
        return adj, aset
    
    def find_cycle(adj, aset, L, node_limit):
        for start in range(N):
            path = [start]
            used = [False]*N
            used[start] = True
            found = [None]
            cnt = [0]
            def dfs():
                if found[0] is not None: return
                if len(path) == L:
                    if (path[-1], start) in aset:
                        found[0] = path[:]
                    return
                cnt[0] += 1
                if cnt[0] > node_limit: return
                v = path[-1]
                for w in adj[v]:
                    if w == start or used[w] or w < start: continue
                    used[w]=True; path.append(w)
                    dfs()
                    if found[0] is not None: return
                    path.pop(); used[w]=False
            dfs()
            if found[0] is not None:
                return found[0]
        return None
    
    def covered(edge_list, node_limit):
        adj, aset = adj_of(edge_list)
        cov = set()
        for L in range(3, N):  # 3..39
            if find_cycle(adj, aset, L, node_limit) is not None:
                cov.add(L)
        cov.add(N)  # 40 always (base cycle)
        return cov
    
    target = set(range(3, N+1))  # 3..40
    
    # Greedy + local search with random restarts
    best_edges = None
    best_count = len(base) + len(candidates)  # init high
    
    def evaluate_and_local(edge_list):
        # returns (edge_list, covered) after local search (removals)
        cov = covered(edge_list, 50000)
        # local search: remove redundant chords
        changed = True
        while changed:
            changed = False
            for i in range(len(edge_list)):
                if edge_list[i] in base_set: continue  # don't remove base
                trial = edge_list[:i] + edge_list[i+1:]
                c2 = covered(trial, 50000)
                if target.issubset(c2):
                    edge_list = trial
                    cov = c2
                    changed = True
                    break
        return edge_list, cov
    
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # start with base + random chords
        if restarts == 1:
            chords = []
        else:
            k = random.randint(6, 12)
            chords = random.sample(candidates, k)
        edge_list = base + chords
        # greedy add
        cov = covered(edge_list, 50000)
        guard = 0
        while not target.issubset(cov) and guard < 30 and time.time() < deadline - 5:
            guard += 1
            # pick random candidate chord not in edge_list
            cur_set = set(frozenset(e) for e in edge_list)
            avail = [c for c in candidates if frozenset(c) not in cur_set]
            if not avail: break
            # try a few random, pick best
            best_c = None; best_cov = cov
            for _ in range(min(20, len(avail))):
                c = random.choice(avail)
                trial = edge_list + [c]
                c2 = covered(trial, 50000)
                if len(c2) > len(best_cov):
                    best_cov = c2; best_c = c
            if best_c is None:
                # pick random
                best_c = random.choice(avail)
                best_cov = covered(edge_list + [best_c], 50000)
            edge_list = edge_list + [best_c]
            cov = best_cov
        # local search removal
        edge_list, cov = evaluate_and_local(edge_list)
        if target.issubset(cov):
            cnt = len(edge_list)
            if cnt < best_count:
                best_count = cnt
                best_edges = edge_list[:]
    
    # Final: verify best_edges with high node limit, get cycles
    if best_edges is not None:
        adj, aset = adj_of(best_edges)
        cycles = {}
        ok = True
        for L in range(3, N+1):
            if L == N:
                cycles[L] = list(range(N))  # base cycle 0-1-...-39-0
            else:
                c = find_cycle(adj, aset, L, 500000)
                if c is None:
                    ok = False; break
                cycles[L] = c
        if ok:
            # verify all cycles valid
            for L in range(3, N+1):
                cyc = cycles[L]
                if len(cyc) != L: ok=False; break
                if len(set(cyc)) != L: ok=False; break
                for i in range(L):
                    a,b = cyc[i], cyc[(i+1)%L]
                    if (a,b) not in aset and (b,a) not in aset:
                        ok=False; break
                if not ok: break
            if ok:
                edges_out = [tuple(sorted(e)) for e in best_edges]
                cycles_out = [cycles[L] for L in range(3, N+1)]
                return {'edges': edges_out, 'cycles': cycles_out}
    
    # Fallback: seed
    edges = [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)]
    cycles = [list(range(k)) for k in range(3,N+1)]
    return {'edges': edges, 'cycles': cycles}
