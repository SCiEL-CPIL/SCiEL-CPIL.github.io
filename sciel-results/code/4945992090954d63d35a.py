import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Bitmask adjacency
    adj = [0] * n
    # common[u][v] = number of common neighbors of u and v
    common = [[0] * n for _ in range(n)]

    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        # Check 1: common neighbors of u and v must be <= 2
        if common[u][v] > 2:
            return False
        # Check 2: For all y in N(v), common[u][y] must be <= 1
        # (because v will become a common neighbor of u and y)
        vw = adj[v]
        while vw:
            lsb = vw & -vw
            y = lsb.bit_length() - 1
            vw ^= lsb
            if y == u:
                continue
            if common[u][y] > 1:
                return False
        # Check 3: For all y in N(u), common[v][y] must be <= 1
        uw = adj[u]
        while uw:
            lsb = uw & -uw
            y = lsb.bit_length() - 1
            uw ^= lsb
            if y == v:
                continue
            if common[v][y] > 1:
                return False
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update common counts
        # For y in N(v) (excluding u): common[u][y] += 1
        vw = adj[v] & ~(1 << u)
        while vw:
            lsb = vw & -vw
            y = lsb.bit_length() - 1
            vw ^= lsb
            common[u][y] += 1
            common[y][u] += 1
        # For y in N(u) (excluding v): common[v][y] += 1
        uw = adj[u] & ~(1 << v)
        while uw:
            lsb = uw & -uw
            y = lsb.bit_length() - 1
            uw ^= lsb
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(u, v):
        # Revert common updates BEFORE removing from adj
        vw = adj[v] & ~(1 << u)
        while vw:
            lsb = vw & -vw
            y = lsb.bit_length() - 1
            vw ^= lsb
            common[u][y] -= 1
            common[y][u] -= 1
        uw = adj[u] & ~(1 << v)
        while uw:
            lsb = uw & -uw
            y = lsb.bit_length() - 1
            uw ^= lsb
            common[v][y] -= 1
            common[y][v] -= 1
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count():
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edges():
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if u < v:
                    edges.append([u, v])
        return edges

    def reset():
        for i in range(n):
            adj[i] = 0
        for i in range(n):
            for j in range(n):
                common[i][j] = 0

    def build_clebsch_seed():
        """Build Clebsch graph on 16 vertices + 4 new vertices connected to disjoint pairs."""
        reset()
        # Clebsch graph: vertices 0-15, adjacency if Hamming distance 1 or 4
        for u in range(16):
            for v in range(u + 1, 16):
                diff = u ^ v
                pc = bin(diff).count('1')
                if pc == 1 or pc == 4:
                    add_edge(u, v)
        # Find 4 disjoint edges in the Clebsch graph
        edges_in_clebsch = [(u, v) for u in range(16) for v in range(u + 1, 16)
                            if (adj[u] >> v) & 1]
        used = set()
        selected = []
        for u, v in edges_in_clebsch:
            if u not in used and v not in used:
                selected.append((u, v))
                used.add(u)
                used.add(v)
                if len(selected) == 4:
                    break
        # Connect new vertices 16-19 to these pairs
        for i, (u, v) in enumerate(selected):
            new_v = 16 + i
            add_edge(new_v, u)
            add_edge(new_v, v)
        # Connect new vertices among themselves
        for i in range(4):
            for j in range(i + 1, 4):
                u = 16 + i
                v = 16 + j
                if can_add(u, v):
                    add_edge(u, v)

    def greedy_fill():
        """Try to add any valid edge repeatedly."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if can_add(u, v):
                        add_edge(u, v)
                        improved = True

    def local_search_1for1():
        """Try to add a single edge. Return True if added."""
        for u in range(n):
            for v in range(u + 1, n):
                if can_add(u, v):
                    add_edge(u, v)
                    return True
        return False

    def local_search_1for2():
        """Try to remove one edge and add two. Return True if net +1."""
        if time.time() > deadline:
            return False
        edges = get_edges()
        if not edges:
            return False
        random.shuffle(edges)
        for u, v in edges[:30]:  # limit tries
            if time.time() > deadline:
                return False
            remove_edge(u, v)
            added = 0
            added_edges = []
            for a in range(n):
                for b in range(a + 1, n):
                    if can_add(a, b):
                        add_edge(a, b)
                        added_edges.append((a, b))
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    break
            if added >= 2:
                return True
            else:
                # Revert
                for a, b in added_edges:
                    remove_edge(a, b)
                add_edge(u, v)
        return False

    def local_search_1for3():
        """Try to remove one edge and add three. Net +2 gain. Return True if net +2."""
        if time.time() > deadline:
            return False
        edges = get_edges()
        if not edges:
            return False
        random.shuffle(edges)
        for u, v in edges[:20]:  # limit tries
            if time.time() > deadline:
                return False
            remove_edge(u, v)
            added = 0
            added_edges = []
            for a in range(n):
                for b in range(a + 1, n):
                    if can_add(a, b):
                        add_edge(a, b)
                        added_edges.append((a, b))
                        added += 1
                        if added >= 3:
                            break
                if added >= 3:
                    break
            if added >= 3:
                return True
            else:
                # Revert
                for a, b in added_edges:
                    remove_edge(a, b)
                add_edge(u, v)
        return False

    def local_search_2for2():
        """Try to remove two edges and add two. Net 0 but may enable further additions."""
        if time.time() > deadline:
            return False
        edges = get_edges()
        if len(edges) < 2:
            return False
        random.shuffle(edges)
        for i in range(min(10, len(edges))):
            if time.time() > deadline:
                return False
            e1 = edges[i]
            e2 = edges[i + 1] if i + 1 < len(edges) else None
            if e2 is None or e2 == e1:
                continue
            remove_edge(e1[0], e1[1])
            remove_edge(e2[0], e2[1])
            added = 0
            added_edges = []
            for a in range(n):
                for b in range(a + 1, n):
                    if can_add(a, b):
                        add_edge(a, b)
                        added_edges.append((a, b))
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    break
            if added >= 2:
                # Try to add more
                for a in range(n):
                    for b in range(a + 1, n):
                        if can_add(a, b):
                            add_edge(a, b)
                return True
            else:
                for a, b in added_edges:
                    remove_edge(a, b)
                add_edge(e1[0], e1[1])
                add_edge(e2[0], e2[1])
        return False

    best_count = 0
    best_edges = None

    # Phase 1: Build Clebsch seed and greedily fill
    build_clebsch_seed()
    greedy_fill()
    cnt = edge_count()
    if cnt > best_count:
        best_count = cnt
        best_edges = get_edges()

    # Phase 2: Local search with various moves
    while time.time() < deadline:
        improved = False
        # Try 1-for-1 (add edge)
        if local_search_1for1():
            improved = True
        # Try 1-for-2
        if local_search_1for2():
            improved = True
        # Try 1-for-3 (net +2)
        if local_search_1for3():
            improved = True
        # Try 2-for-2
        if local_search_2for2():
            improved = True

        cnt = edge_count()
        if cnt > best_count:
            best_count = cnt
            best_edges = get_edges()

        if not improved:
            # Stuck, try a random perturbation: remove a random edge and refill
            if time.time() > deadline:
                break
            edges = get_edges()
            if edges:
                u, v = random.choice(edges)
                remove_edge(u, v)
                greedy_fill()
                cnt = edge_count()
                if cnt > best_count:
                    best_count = cnt
                    best_edges = get_edges()

    # Final: try to add any remaining edges
    if time.time() < deadline:
        greedy_fill()
        cnt = edge_count()
        if cnt > best_count:
            best_count = cnt
            best_edges = get_edges()

    if best_edges is None:
        # Fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
