import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def time_ok():
        return (time.time() - start_time) < DEADLINE

    def try_solve(n):
        if not time_ok():
            return None

        total_edges = n * (n - 1) // 2
        colors = [-1] * total_edges
        adj_list = [[[] for _ in range(n)] for _ in range(4)]
        adj_mat = [[[[0] * n for _ in range(n)] for _ in range(4)]]
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def check_c6(u, v, c):
            """Check if there's a simple path u-a-b-c2-d-v of length 5 in color-c subgraph."""
            for a in adj_list[c][u]:
                if a == v:
                    continue
                for b in adj_list[c][a]:
                    if b == u or b == v or b == a:
                        continue
                    for c2 in adj_list[c][b]:
                        if c2 == u or c2 == v or c2 == a or c2 == b:
                            continue
                        for d in adj_list[c][c2]:
                            if d == u or d == v or d == a or d == b or d == c2:
                                continue
                            if adj_mat[c][d][v]:
                                return True
            return False

        def solve(idx):
            if not time_ok():
                return False
            if idx == total_edges:
                return True

            u, v = edge_list[idx]

            for c in range(4):
                if check_c6(u, v, c):
                    continue

                colors[idx] = c
                adj_list[c][u].append(v)
                adj_list[c][v].append(u)
                adj_mat[c][u][v] = 1
                adj_mat[c][v][u] = 1

                if solve(idx + 1):
                    return True

                colors[idx] = -1
                adj_list[c][u].remove(v)
                adj_list[c][v].remove(u)
                adj_mat[c][u][v] = 0
                adj_mat[c][v][u] = 0

            return False

        if solve(0):
            return {'n': n, 'colors': colors}
        return None

    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if not time_ok():
            break
        result = try_solve(n)
        if result is not None:
            return result

    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
