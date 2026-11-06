import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under 150s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges):
        adj = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def compute_forbid(edges, adj):
        """forbid[i] = endpoints of edge i + all neighbours of those endpoints"""
        m = len(edges)
        forbid = []
        for (u, v) in edges:
            s = {u, v}
            s.update(adj.get(u, ()))
            s.update(adj.get(v, ()))
            forbid.append(s)
        return forbid

    def greedy_colour(edges, adj, order, forbid):
        """Assign colours greedily following order. Returns (colour_list, num_classes)."""
        m = len(edges)
        colour = [-1] * m
        classes = []       # list of edge-index lists
        class_forbid = []  # class_forbid[c] = set of vertices (endpoints) in class c

        for i in order:
            fi = forbid[i]
            best_c = -1
            # Prefer the smallest-index compatible class to keep colours compact
            for c in range(len(classes)):
                if fi.isdisjoint(class_forbid[c]):
                    best_c = c
                    break
            if best_c >= 0:
                colour[i] = best_c
                classes[best_c].append(i)
                class_forbid[best_c].update(edges[i])
            else:
                c = len(classes)
                colour[i] = c
                classes.append([i])
                class_forbid.append(set(edges[i]))
        return colour, len(classes), classes, class_forbid

    def merge_classes(edges, adj, colour, k, classes, class_forbid):
        """Try to merge classes. Returns (new_colour, new_k, new_classes, new_class_forbid)."""
        m = len(edges)
        # Precompute forbid_union for each class
        def get_forbid_union(c):
            s = set()
            for i in classes[c]:
                u, v = edges[i]
                s.add(u); s.add(v)
                s.update(adj.get(u, ()))
                s.update(adj.get(v, ()))
            return s

        forb_union = [get_forbid_union(c) for c in range(k)]
        
        changed = True
        while changed and time.time() < DEADLINE - 2:
            changed = False
            # Recompute if needed (we update in-place below)
            for a in range(k):
                if a >= k:
                    continue
                for b in range(a + 1, k):
                    if b >= k:
                        continue
                    # Can we merge class b into class a?
                    # Need: forb_union[a] disjoint from class_forbid[b]
                    #       AND forb_union[b] disjoint from class_forbid[a]
                    if forb_union[a].isdisjoint(class_forbid[b]) and \
                       forb_union[b].isdisjoint(class_forbid[a]):
                        # Merge b into a
                        for i in range(m):
                            if colour[i] == b:
                                colour[i] = a
                        classes[a].extend(classes[b])
                        class_forbid[a] |= class_forbid[b]
                        # Remove class b
                        classes[b] = []
                        class_forbid[b] = set()
                        forb_union[b] = set()
                        k -= 1
                        changed = True
                        break
                if changed:
                    break
            if changed:
                # Renumber remaining classes compactly
                new_colour = [-1] * m
                mapping = {}
                new_k = 0
                for c in range(k + 1):  # original k before decrement, but we removed one
                    # Actually, we need to track which class indices are still valid
                    pass
                # Simpler: rebuild
                seen = {}
                new_k = 0
                for i in range(m):
                    if colour[i] not in seen:
                        seen[colour[i]] = new_k
                        new_k += 1
                    new_colour[i] = seen[colour[i]]
                
                # Rebuild classes and class_forbid
                new_classes = [[] for _ in range(new_k)]
                new_class_forbid = [set() for _ in range(new_k)]
                for i in range(m):
                    c = new_colour[i]
                    new_classes[c].append(i)
                    new_class_forbid[c].update(edges[i])
                
                return new_colour, new_k, new_classes, new_class_forbid
        return colour, k, classes, class_forbid

    def make_order(edges, adj, mode, rng):
        m = len(edges)
        idx = list(range(m))
        deg = {}
        for (u, v) in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        
        if mode == 0:
            # Descending sum of endpoint degrees
            idx.sort(key=lambda i: (-(deg[edges[i][0]] + deg[edges[i][1]]), i))
        elif mode == 1:
            # Random
            rng.shuffle(idx)
        elif mode == 2:
            # Ascending sum of endpoint degrees
            idx.sort(key=lambda i: (deg[edges[i][0]] + deg[edges[i][1]], i))
        elif mode == 3:
            # Descending product
            idx.sort(key=lambda i: (-(deg[edges[i][0]] * max(1, deg[edges[i][1]])), i))
        elif mode == 4:
            # Descending max endpoint degree
            idx.sort(key=lambda i: (-max(deg[edges[i][0]], deg[edges[i][1]]), i))
        else:
            idx.sort(key=lambda i: i)
        return idx

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        adj = build_adj(edges)
        m = len(edges)
        forbid = compute_forbid(edges, adj)
        
        best_k = m + 1
        best_colour = None
        rng = Random(12345 + seed)
        
        # Try multiple orderings and merge
        modes = [0, 3, 4, 2, 1]
        for mode in modes:
            if time.time() > DEADLINE - 5:
                break
            for restart in range(3):
                if time.time() > DEADLINE - 5:
                    break
                order = make_order(edges, adj, mode, rng)
                colour, k, classes, class_forbid = greedy_colour(edges, adj, order, forbid)
                
                # Apply merging
                colour, k, classes, class_forbid = merge_classes(edges, adj, colour, k, classes, class_forbid)
                
                if k < best_k:
                    best_k = k
                    best_colour = colour[:]
        
        # Fallback: if best_colour is None, use a simple greedy
        if best_colour is None:
            order = make_order(edges, adj, 0, rng)
            best_colour, best_k, _, _ = greedy_colour(edges, adj, order, forbid)
        
        return best_colour

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        if time.time() > DEADLINE - 10:
            # Emergency: return minimal valid coloring (each edge own colour)
            edges = gen_instance(n, d, seed)
            results.append(list(range(len(edges))))
            continue
        results.append(solve_instance(n, d, seed))
    
    return results
