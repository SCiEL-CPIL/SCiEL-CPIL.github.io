"""Improved program: hill-climb to maximize edges subject to no s-u-v-t path with deg(s)==deg(t)."""
import time
import random

N = 15
DEADLINE = 80.0  # safety margin under 90s

def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    def get_degrees(adj):
        return [sum(adj[i]) for i in range(N)]

    def check_valid(adj):
        deg = get_degrees(adj)
        # For every edge (u,v) (the middle edge of the 3-edge path s-u-v-t):
        #   s in N(u), t in N(v), s != t, and we need deg[s] != deg[t].
        # If any s,t exist with deg[s]==deg[t], invalid.
        # Also need s-u-v-t to be a simple path: s!=v, t!=u, s!=t (s!=t already checked)
        # s != v: s is neighbor of u, s could be v? If s==v then path is v-u-v-t which is not simple.
        # t != u: t is neighbor of v, t could be u? If t==u then s-u-v-u not simple.
        # So exclude s==v and t==u.
        for u in range(N):
            for v in range(N):
                if adj[u][v] == 0:
                    continue
                for s in range(N):
                    if s == u or s == v:
                        continue
                    if adj[s][u] == 0:
                        continue
                    for t in range(N):
                        if t == u or t == v:
                            continue
                        if t == s:
                            continue
                        if adj[v][t] == 0:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def edge_count(adj):
        return sum(sum(row) for row in adj) // 2

    def get_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    best_adj = None
    best_ec = -1

    for restart in range(50):
        if elapsed() > DEADLINE:
            break

        # Random initial graph
        adj = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.4:
                    adj[i][j] = 1
                    adj[j][i] = 1

        ec = edge_count(adj)
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            for i in range(N):
                if elapsed() > DEADLINE:
                    break
                for j in range(i+1, N):
                    if elapsed() > DEADLINE:
                        break
                    if adj[i][j] == 1:
                        continue
                    # Try adding (i, j)
                    adj[i][j] = 1
                    adj[j][i] = 1
                    if check_valid(adj):
                        improved = True
                        ec += 1
                        # Keep
                    else:
                        adj[i][j] = 0
                        adj[j][i] = 0

        if ec > best_ec:
            best_ec = ec
            best_adj = [row[:] for row in adj]

    if best_adj is None:
        # Fallback star
        best_adj = [[0]*N for _ in range(N)]
        for i in range(1, N):
            best_adj[0][i] = 1
            best_adj[i][0] = 1

    return get_edges(best_adj)
