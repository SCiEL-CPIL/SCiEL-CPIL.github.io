import time
import random

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    n = 21
    edges = [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]

    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
        adj[a].add(b)
        adj[b].add(a)

    def is_connected(edge_set_arg):
        if not edge_set_arg:
            return False
        temp_adj = [set() for _ in range(n)]
        for (u, v) in edge_set_arg:
            temp_adj[u].add(v)
            temp_adj[v].add(u)
        visited = set([0])
        stack = [0]
        while stack:
            u = stack.pop()
            for v in temp_adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return len(visited) == n

    def find_ham_cycle(avail, start, randomize=False, max_steps=300000):
        if time.time() > deadline:
            return None
        if not is_connected(avail):
            return None

        path = [start]
        visited = {start}
        used = set()
        steps = [0]

        def bt():
            if time.time() > deadline:
                return None
            steps[0] += 1
            if steps[0] > max_steps:
                return None
            if len(path) == n:
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in avail and key not in used:
                    return list(path)
                return None
            last = path[-1]
            nbrs = []
            for v in adj[last]:
                if v not in visited:
                    key = (min(last, v), max(last, v))
                    if key in avail and key not in used:
                        nbrs.append(v)
            if not nbrs:
                return None
            if randomize:
                random.shuffle(nbrs)
            def ndeg(v):
                c = 0
                for u in adj[v]:
                    if u not in visited and u != last:
                        key = (min(u, v), max(u, v))
                        if key in avail and key not in used:
                            c += 1
                return c
            nbrs.sort(key=ndeg)
            for nb in nbrs:
                key = (min(last, nb), max(last, nb))
                if key not in avail or key in used:
                    continue
                used.add(key)
                path.append(nb)
                visited.add(nb)
                res = bt()
                if res is not None:
                    return res
                path.pop()
                visited.discard(nb)
                used.discard(key)
            return None

        return bt()

    cycles = []
    available = set(edge_set)

    # Find first cycle
    for start in range(n):
        if time.time() > deadline:
            break
        c = find_ham_cycle(available, start, False, 200000)
        if c is not None:
            cycles.append(c)
            for i in range(len(c)):
                a, b = c[i], c[(i+1) % len(c)]
                available.discard((min(a,b), max(a,b)))
            break

    # Find second cycle
    if len(cycles) < 3 and time.time() < deadline:
        if is_connected(available):
            for start in range(n):
                if time.time() > deadline:
                    break
                c = find_ham_cycle(available, start, False, 200000)
                if c is not None:
                    cycles.append(c)
                    for i in range(len(c)):
                        a, b = c[i], c[(i+1) % len(c)]
                        available.discard((min(a,b), max(a,b)))
                    break

    # Find third cycle - remaining should be 2-regular, check if it's a single cycle
    if len(cycles) < 3 and time.time() < deadline:
        if is_connected(available) and len(available) == n:
            # 2-regular connected graph on n vertices is a Hamiltonian cycle
            # Reconstruct it
            rem_adj = [set() for _ in range(n)]
            for (u,v) in available:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            # Verify each vertex has degree 2
            valid = all(len(rem_adj[i]) == 2 for i in range(n))
            if valid:
                # Find the cycle
                c = [0]
                prev = None
                cur = 0
                while True:
                    nxts = list(rem_adj[cur])
                    nxt = [x for x in nxts if x != prev][0]
                    c.append(nxt)
                    prev, cur = cur, nxt
                    if cur == 0 and len(c) == n + 1:
                        break
                    if cur == 0:
                        break
                c = c[:-1]  # Remove the repeated 0
                if len(c) == n and len(set(c)) == n:
                    cycles.append(c)
                    for i in range(len(c)):
                        a, b = c[i], c[(i+1) % len(c)]
                        available.discard((min(a,b), max(a,b)))

    # If we still don't have 3, try randomized
    if len(cycles) < 3 and time.time() < deadline:
        for _ in range(30):
            if time.time() > deadline:
                break
            if len(cycles) >= 3:
                break
            if not is_connected(available):
                break
            start = random.randint(0, n-1)
            c = find_ham_cycle(available, start, True, 100000)
            if c is not None:
                cycles.append(c)
                for i in range(len(c)):
                    a, b = c[i], c[(i+1) % len(c)]
                    available.discard((min(a,b), max(a,b)))

    if len(cycles) == 0:
        available = set(edge_set)
        for s in range(n):
            if time.time() > deadline:
                break
            c = find_ham_cycle(available, s, False, 100000)
            if c is not None:
                cycles = [c]
                break

    cycles = cycles[:10]
    result = [[int(v) for v in c] for c in cycles]
    return result
