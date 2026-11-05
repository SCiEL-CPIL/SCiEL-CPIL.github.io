def _fast_local_search(adj, deg, cur_score, t0, rng, iters_budget):
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]
    edge_set = set(edges)

    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]

    start_temp = 4.0
    end_temp = 0.02
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    stall = 0
    stall_limit = 5000
    worst_tri = None
    while iters < iters_budget:
        if (iters % 100 == 0) and (time.time() - t0 > DEADLINE - 3.0):
            break

        if not edges or not nonedges:
            break

        # If we don't have a worst triangle, find one
        if worst_tri is None:
            cur_score, worst_tri = _score_and_worst(adj, deg)
            if cur_score < best_score:
                best_score = cur_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]

        # Biased move selection: 90% chance to target worst triangle
        if worst_tri is not None and rng.random() < 0.9:
            i, j, k = worst_tri
            tri_edges = [(i, j), (i, k), (j, k)]
            er = tri_edges[rng.randrange(3)]
            if er not in edge_set:
                er = edges[rng.randrange(len(edges))]
            
            # Bias adding edge to low degree vertices
            # Calculate weights for nonedges based on deg[x]+deg[y]
            # Lower sum is better
            weights = [1.0 / (1.0 + deg[x] + deg[y]) for (x, y) in nonedges]
            total_w = sum(weights)
            if total_w == 0:
                ea = nonedges[rng.randrange(len(nonedges))]
            else:
                r = rng.random() * total_w
                cumulative = 0
                ea = nonedges[-1]
                for idx, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        ea = nonedges[idx]
                        break
        else:
            er = edges[rng.randrange(len(edges))]
            ea = nonedges[rng.randrange(len(nonedges))]

        if er == ea:
            iters += 1
            continue
        u, v = er
        x, y = ea

        # apply
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1; deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1; deg[y] += 1

        new_score, new_worst_tri = _score_and_worst(adj, deg)
        worst_tri = new_worst_tri

        accept = False
        if new_score < cur_score:
            accept = True
            stall = 0
        elif new_score == cur_score:
            accept = rng.random() < 0.35
            stall += 1
        else:
            d = new_score - cur_score
            if temp < 1e-9:
                accept = False
                stall += 1
            else:
                accept = math.exp(-d / temp) > rng.random()
                if accept:
                    stall += 1
                else:
                    stall += 1

        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = cur_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            # update lists
            edge_set.discard(er)
            edge_set.add(ea)
            edges.remove(er)
            edges.append(ea)
            nonedges.remove(ea)
            nonedges.append(er)
        else:
            # undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1; deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1; deg[v] += 1

        iters += 1
        temp *= cool

        # If stalled too long, restart with higher temp briefly
        if stall > stall_limit:
            temp = max(temp, 2.0)
            stall = 0
            worst_tri = None # Force re-evaluation

    return best_adj, best_deg, best_score
