import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    if k != 3:
        return _fallback_algebraic(k, g, deadline)
    return _construct_repair(3, g, deadline)


def _fallback_algebraic(k, g, deadline):
    # Simple algebraic construction: voltage graph over K4
    m = 257
    voltages = [1, 3, 7, 13, 29, 61]
    n = 4 * m
    edges = []
    edge_map = {(0,1):0,(1,0):0,(0,2):1,(2,0):1,(0,3):2,(3,0):2,
                (1,2):3,(2,1):3,(1,3):4,(3,1):4,(2,3):5,(3,2):5}
    for u in range(4):
        for w in range(4):
            if u == w:
                continue
            vol = voltages[edge_map[(u,w)]]
            for v in range(m):
                a = v * 4 + u
                b = ((v + vol) % m) * 4 + w
                if a < b:
                    edges.append((a, b))
                else:
                    edges.append((b, a))
    return n, list(set(edges))


def _random_3reg(n, rng):
    if n % 2 != 0:
        n += 1
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i+1]
        if a == b:
            return None
        key = (min(a,b), max(a,b))
        if key in edges:
            return None
        edges.add(key)
    # Verify simple + 3-regular
    deg = [0]*n
    for a,b in edges:
        deg[a]+=1; deg[b]+=1
    if any(d != 3 for d in deg):
        return None
    return list(edges)


def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for a,b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _find_short_cycle(adj, n, g):
    # Find a cycle of length < g if exists; return list of vertices or None
    # Use BFS from each vertex with parent tracking, early stop
    for start in range(n):
        parent = [-1]*n
        level = [-1]*n
        level[start] = 0
        q = deque([start])
        found = None
        while q:
            u = q.popleft()
            if level[u] >= (g-1)//2:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if level[v] == -1:
                    level[v] = level[u]+1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and parent[v] != u:
                    cl = level[u] + level[v] + 1
                    if 3 <= cl < g:
                        # reconstruct
                        path_u = [u]
                        while path_u[-1] != start:
                            path_u.append(parent[path_u[-1]])
                        path_u.reverse()  # start..u
                        path_v = [v]
                        while path_v[-1] != start:
                            path_v.append(parent[path_v[-1]])
                        path_v.reverse()  # start..v
                        cycle = path_u + path_v[1:]
                        found = cycle
                        break
            if found:
                break
        if found:
            return found
    return None


def _try_break_cycle(adj, edges_set, n, cycle, rng, g):
    # cycle is list of vertices [v0, v1, ..., v_{L-1}] with edges (vi, v_{i+1 mod L})
    L = len(cycle)
    if L < 3 or L >= g:
        return False
    # For each edge in the cycle (a,b), try to find a 2-switch:
    #   a is adjacent to x (x != b, x != prev), b adjacent to y (y != a, y != next)
    #   swap (a,b) and (x,y) -> (a,y),(x,b)  if both new edges exist? No, we add them.
    # Actually 2-switch: remove (a,b) and (x,y), add (a,y) and (x,b) or (a,b)->(a,y),(x,b)
    # We need (x,y) to be an existing edge not in the cycle.
    # After switch, check we don't create a shorter cycle or duplicate.
    for i in range(L):
        a = cycle[i]
        b = cycle[(i+1) % L]
        # neighbors of a except b
        cand_a = [x for x in adj[a] if x != b]
        for x in cand_a:
            # neighbors of b except a
            for y in adj[b]:
                if y == a:
                    continue
                if y == x:
                    continue
                if (min(x,y), max(x,y)) not in edges_set:
                    continue
                # proposed new edges (a,y) and (x,b)
                if a == y or x == b:
                    continue
                if (min(a,y), max(a,y)) in edges_set:
                    continue
                if (min(x,b), max(x,b)) in edges_set:
                    continue
                # perform switch
                edges_set.discard((min(a,b), max(a,b)))
                edges_set.discard((min(x,y), max(x,y)))
                edges_set.add((min(a,y), max(a,y)))
                edges_set.add((min(x,b), max(x,b)))
                # update adj
                adj[a].remove(b); adj[b].remove(a)
                adj[x].remove(y); adj[y].remove(x)
                adj[a].append(y); adj[y].append(a)
                adj[x].append(b); adj[b].append(x)
                return True
    return False


def _construct_repair(k, g, deadline):
    rng = random.Random(42)
    # Try several starting n values
    for n0 in [512, 514, 516, 518, 520, 524, 528, 532, 536]:
        if time.time() > deadline - 10:
            break
        edges_list = _random_3reg(n0, rng)
        if edges_list is None:
            continue
        edges_set = set(edges_list)
        adj = _build_adj(edges_list, n0)
        # Repair loop
        for it in range(200000):
            if time.time() > deadline - 15:
                break
            cycle = _find_short_cycle(adj, n0, g)
            if cycle is None:
                return n0, list(edges_set)
            if not _try_break_cycle(adj, edges_set, n0, cycle, rng, g):
                # restart with new random graph
                break
    # Fallback: larger n
    for n0 in [540, 548, 556, 564, 572, 580, 588, 600]:
        if time.time() > deadline - 10:
            break
        edges_list = _random_3reg(n0, rng)
        if edges_list is None:
            continue
        edges_set = set(edges_list)
        adj = _build_adj(edges_list, n0)
        for it in range(200000):
            if time.time() > deadline - 15:
                break
            cycle = _find_short_cycle(adj, n0, g)
            if cycle is None:
                return n0, list(edges_set)
            if not _try_break_cycle(adj, edges_set, n0, cycle, rng, g):
                break
    # Last resort algebraic
    return _fallback_algebraic(k, g, deadline)


def _has_girth_at_least(n, edges, g):
    adj = _build_adj(edges, n)
    return _find_short_cycle(adj, n, g) is None
