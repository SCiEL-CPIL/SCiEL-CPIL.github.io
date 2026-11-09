import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n = 16
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Start with a dense random graph
    random.seed(42)
    edges = set()
    for e in all_edges:
        if random.random() < 0.85:
            edges.add(e)

    # Adjacency as bitmasks
    adj = [0] * n

    def rebuild_adj():
        for i in range(n):
            adj[i] = 0
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    rebuild_adj()

    def has_q3():
        """Check if graph contains Q3. Return True if found."""
        # Q3 is bipartite with parts of size 4.
        # For each 8-set, check if it contains a Q3.
        # Optimization: Q3 has 12 edges. If induced subgraph has < 12 edges, skip.
        for combo in itertools.combinations(range(n), 8):
            # Count edges in induced subgraph
            ec = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        ec += 1
            if ec < 12:
                continue

            # Try all ways to split 8 vertices into two groups of 4
            # There are C(8,4)/2 = 35 unique splits (but we check all 70, 
            # since A,B and B,A give different matchings)
            vlist = list(combo)
            for i in range(8):
                for j in range(i + 1, 8):
                    for k in range(j + 1, 8):
                        for l in range(k + 1, 8):
                            A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                            B = [x for x in vlist if x not in A]

                            # Check if missing edges in A×B form a partial matching
                            # Missing edges must be ≤ 4 and form a matching
                            rows_used = set()
                            cols_used = set()
                            missing_count = 0
                            valid = True
                            for ri, a in enumerate(A):
                                for ci, b in enumerate(B):
                                    if not ((adj[a] >> b) & 1):
                                        if ri in rows_used or ci in cols_used:
                                            valid = False
                                            break
                                        rows_used.add(ri)
                                        cols_used.add(ci)
                                        missing_count += 1
                                if not valid:
                                    break
                            if valid and missing_count <= 4:
                                return True
        return False

    # Phase 1: Remove edges to make Q3-free
    while time.time() < deadline - 20:
        if not has_q3():
            break
        # Find a Q3 and remove one of its edges
        # We need to find which edges to remove
        # Re-do: find Q3 and return an edge to remove
        found = False
        for combo in itertools.combinations(range(n), 8):
            ec = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        ec += 1
            if ec < 12:
                continue
            vlist = list(combo)
            for i in range(8):
                for j in range(i + 1, 8):
                    for k in range(j + 1, 8):
                        for l in range(k + 1, 8):
                            A = [vlist[i], vlist[j], vlist[k], vlist[l]]
                            B = [x for x in vlist if x not in A]
                            rows_used = set()
                            cols_used = set()
                            missing_count = 0
                            valid = True
                            for ri, a in enumerate(A):
                                for ci, b in enumerate(B):
                                    if not ((adj[a] >> b) & 1):
                                        if ri in rows_used or ci in cols_used:
                                            valid = False
                                            break
                                        rows_used.add(ri)
                                        cols_used.add(ci)
                                        missing_count += 1
                                if not valid:
                                    break
                            if valid and missing_count <= 4:
                                # Found Q3. Remove a random edge from A×B that exists
                                existing = []
                                for a in A:
                                    for b in B:
                                        if (adj[a] >> b) & 1:
                                            existing.append((a, b))
                                if existing:
                                    e_to_remove = random.choice(existing)
                                    if e_to_remove in edges:
                                        edges.remove(e_to_remove)
                                        adj[e_to_remove[0]] &= ~(1 << e_to_remove[1])
                                        adj[e_to_remove[1]] &= ~(1 << e_to_remove[0])
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break

    # Phase 2: Try to add edges
    while time.time() < deadline - 5:
        added = False
        non_edges = [e for e in all_edges if e not in edges]
        random.shuffle(non_edges)
        for (u, v) in non_edges:
            # Try adding (u,v)
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3():
                added = True
                break
            else:
                edges.remove((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        if not added:
            break

    return [list(e) for e in edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
