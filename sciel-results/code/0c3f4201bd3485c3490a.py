"""Rewritten program: greedy + tabu local search for densest valid graph."""
import random
import time

# EVOLVE-BLOCK-START
P = {'n': 15}


def _deg_and_check(adj, n):
    """Return (degrees, valid_bool). adj: list of sets."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # path s-u-v-t
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return deg, False
    return deg, True


def _valid(adj, n):
    return _deg_and_check(adj, n)[1]


def _build_greedy(n, rng, max_edges=None):
    """Add edges in random order, keep if valid."""
    adj = [set() for _ in range(n)]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    count = 0
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
        if not _valid(adj, n):
            adj[i].discard(j)
            adj[j].discard(i)
        else:
            count += 1
        if max_edges is not None and count >= max_edges:
            break
    return adj, count


def _tabu(adj, n, rng, iters, deadline, tabu_len=6):
    """Local search: add missing edges, swap edges. Return best (adj, count)."""
    cur_adj = [set(s) for s in adj]
    cur_count = sum(len(s) for s in cur_adj) // 2
    best_adj = [set(s) for s in cur_adj]
    best_count = cur_count
    tabu = []  # list of (frozenset of changed edges, expiry_iter)
    no_improve = 0
    for it in range(iters):
        if time.time() > deadline:
            break
        # candidate moves
        moves = []
        # 1) add a missing edge
        for i in range(n):
            for j in range(i + 1, n):
                if j not in cur_adj[i]:
                    moves.append(('add', i, j))
        # 2) swap: remove (a,b), add (c,d)
        for a in range(n):
            for b in cur_adj[a]:
                if a >= b:
                    continue
                for c in range(n):
                    for d in range(c + 1, n):
                        if d in cur_adj[c]:
                            continue
                        if (c == a and d == b) or (c == b and d == a):
                            continue
                        moves.append(('swap', a, b, c, d))
        rng.shuffle(moves)
        improved = False
        for mv in moves:
            if time.time() > deadline:
                break
            if mv[0] == 'add':
                _, i, j = mv
                cur_adj[i].add(j)
                cur_adj[j].add(i)
                ok = _valid(cur_adj, n)
                new_count = cur_count + 1
                key = frozenset([(i, j)])
            else:
                _, a, b, c, d = mv
                cur_adj[a].discard(b)
                cur_adj[b].discard(a)
                cur_adj[c].add(d)
                cur_adj[d].add(c)
                ok = _valid(cur_adj, n)
                new_count = cur_count
                key = frozenset([(a, b), (c, d)])
            if key in [t[0] for t in tabu if t[1] > it]:
                # undo and skip
                if mv[0] == 'add':
                    cur_adj[i].discard(j)
                    cur_adj[j].discard(i)
                else:
                    cur_adj[a].add(b)
                    cur_adj[b].add(a)
                    cur_adj[c].discard(d)
                    cur_adj[d].discard(c)
                continue
            if ok:
                if new_count > cur_count:
                    cur_count = new_count
                    improved = True
                    tabu.append((key, it + tabu_len))
                    if len(tabu) > 20:
                        tabu.pop(0)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = [set(s) for s in cur_adj]
                        no_improve = 0
                    break
                else:
                    # undo
                    if mv[0] == 'add':
                        cur_adj[i].discard(j)
                        cur_adj[j].discard(i)
                    else:
                        cur_adj[a].add(b)
                        cur_adj[b].add(a)
                        cur_adj[c].discard(d)
                        cur_adj[d].discard(c)
            else:
                # undo
                if mv[0] == 'add':
                    cur_adj[i].discard(j)
                    cur_adj[j].discard(i)
                else:
                    cur_adj[a].add(b)
                    cur_adj[b].add(a)
                    cur_adj[c].discard(d)
                    cur_adj[d].discard(c)
        if not improved:
            no_improve += 1
            if no_improve > 40:
                break
    return best_adj, best_count


def construct():
    n = P['n']
    start = time.time()
    deadline = start + 50.0  # safety margin under 60/90s
    rng = random.Random(12345)

    best_adj = None
    best_count = -1

    # multiple greedy starts
    n_starts = 60
    for k in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        adj, cnt = _build_greedy(n, rng)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
        # refine with tabu
        adj2, cnt2 = _tabu(adj, n, rng, iters=300, deadline=deadline - 15.0, tabu_len=5)
        if cnt2 > best_count:
            best_count = cnt2
            best_adj = [set(s) for s in adj2]

    # final polish on the best
    if best_adj is not None and time.time() < deadline - 5.0:
        adj3, cnt3 = _tabu(best_adj, n, rng, iters=800, deadline=deadline - 2.0, tabu_len=6)
        if cnt3 > best_count:
            best_count = cnt3
            best_adj = [set(s) for s in adj3]

    if best_adj is None:
        best_adj = [set() for _ in range(n)]

    # build edge list
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    return edges
# EVOLVE-BLOCK-END
