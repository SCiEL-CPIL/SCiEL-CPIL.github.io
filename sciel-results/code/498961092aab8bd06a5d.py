"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    n = 20
    num_colors = 20

    # edge_color[(u,v)] = c with u < v, or absent
    edge_color = {}
    adj = [set() for _ in range(n)]

    def get_color(u, v):
        if u > v:
            u, v = v, u
        return edge_color.get((u, v), -1)

    def set_edge(u, v, c):
        if u > v:
            u, v = v, u
        edge_color[(u, v)] = c
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) in edge_color:
            del edge_color[(u, v)]
            adj[u].discard(v)
            adj[v].discard(u)

    def is_valid_with_new_edge(u, v, c):
        """Check if adding edge (u,v) with colour c keeps all 4-cycles rainbow.
        Only new 4-cycles containing edge (u,v) need checking."""
        common = adj[u] & adj[v]
        if not common:
            return True
        clist = list(common)
        for i in range(len(clist)):
            x = clist[i]
            cvx = get_color(v, x)
            if cvx == -1:
                continue
            for j in range(len(clist)):
                if i == j:
                    continue
                y = clist[j]
                cxy = get_color(x, y)
                if cxy == -1:
                    continue
                cyu = get_color(y, u)
                if cyu == -1:
                    continue
                colors = (c, cvx, cxy, cyu)
                if colors[0] == colors[1] or colors[0] == colors[2] or colors[0] == colors[3] \
                   or colors[1] == colors[2] or colors[1] == colors[3] \
                   or colors[2] == colors[3]:
                    return False
        return True

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(12345)
    random.shuffle(all_edges)

    # Pass 1: greedy add
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if is_valid_with_new_edge(u, v, c):
                set_edge(u, v, c)
                break

    # Pass 2: retry missing edges
    missing = [(i, j) for (i, j) in all_edges if (i, j) not in edge_color]
    random.shuffle(missing)
    for (u, v) in missing:
        if time.time() > deadline:
            break
        for c in range(num_colors):
            if is_valid_with_new_edge(u, v, c):
                set_edge(u, v, c)
                break

    # Pass 3: try to add more by removing some edges and re-adding
    # (local search) - only if time permits
    if time.time() < deadline - 30:
        # Try to improve by swapping colours
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            # Try to add a missing edge by recolouring
            missing_now = [(i, j) for (i, j) in all_edges if (i, j) not in edge_color]
            if not missing_now:
                break
            random.shuffle(missing_now)
            for (u, v) in missing_now[:20]:
                for c in range(num_colors):
                    if is_valid_with_new_edge(u, v, c):
                        set_edge(u, v, c)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    result = []
    for (u, v), c in edge_color.items():
        result.append([u, v, c])

    return result

# EVOLVE-BLOCK-END
