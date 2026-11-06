"""Stronger decomposition: max 2-factor + greedy long cycles + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, time_budget):
        if not edges:
            return []

        # ---- Phase A: maximum 2-factor (vertex-disjoint cycles, max vertices) ----
        n = max(max(u, v) for u, v in edges) + 1
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Try multiple randomized orderings; keep the one covering most vertices
        best_assignment = None
        best_count = 0
        rng = random.Random(777)
        num_trials = 40
        for trial in range(num_trials):
            if time.time() > time_budget - 2:
                break
            order = list(range(n))
            rng.shuffle(order)
            assignment = {}
            for u in order:
                # candidates: neighbors not yet assigned
                cands = [v for v in adj[u] if v not in assignment]
                if not cands:
                    continue
                # prefer candidates that have fewest free neighbors (tightest)
                cands.sort(key=lambda v: sum(1 for w in adj[v] if w not in assignment))
                # small random perturbation
                if len(cands) > 1 and rng.random() < 0.3:
                    k = 1 + int(rng.random() * min(3, len(cands) - 1))
                    v = cands[k]
                else:
                    v = cands[0]
                assignment[u] = v
                assignment[v] = u

            # extract cycles
            cycles = []
            visited = set()
            for u in assignment:
                if u in visited:
                    continue
                path = []
                cur = u
                while cur not in visited:
                    visited.add(cur)
                    path.append(cur)
                    cur = assignment[cur]
                if len(path) >= 3:
                    cycles.append(path)
            count = sum(len(c) for c in cycles)
            if count > best_count:
                best_count = count
                best_assignment = assignment
                best_cycles = [c[:] for c in cycles]

        # Build the 2-factor edge set from best cycles
        factor_edges = set()
        best_cycles = best_cycles if best_cycles else []
        for cyc in best_cycles:
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                factor_edges.add(tuple(sorted((u, v))))

        # ---- Phase B: decompose leftover edges into long cycles ----
        leftover = set(edges) - factor_edges
        pieces = [c[:] for c in best_cycles]

        def get_adj(rem):
            a = defaultdict(set)
            for u, v in rem:
                a[u].add(v)
                a[v].add(u)
            return a

        def find_long_cycle(rem, adj, max_attempts=60):
            if not rem:
                return None
            verts = []
            for u, v in rem:
                verts.append(u)
                verts.append(v)
            verts = list(set(verts))
            if not verts:
                return None
            best_cycle = None
            best_len = 0
            for _ in range(max_attempts):
                if time.time() > time_budget - 2:
                    break
                start = rng.choice(verts)
                path = [start]
                visited = {start}
                cur = start
                while True:
                    nbs = []
                    for nb in adj.get(cur, []):
                        e = tuple(sorted((cur, nb)))
                        if e not in rem:
                            continue
                        if nb == start and len(path) >= 3:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                            return best_cycle
                        if nb not in visited:
                            nbs.append(nb)
                    if not nbs:
                        break
                    # prefer low remaining-degree neighbors
                    nbs.sort(key=lambda nb: len(adj.get(nb, set())) - 1)
                    if len(nbs) > 1 and rng.random() < 0.25:
                        nxt = nbs[1]
                    else:
                        nxt = nbs[0]
                    path.append(nxt)
                    visited.add(nxt)
                    cur = nxt
            return best_cycle

        while leftover:
            if time.time() > time_budget - 2:
                break
            adj = get_adj(leftover)
            cyc = find_long_cycle(leftover, adj, max_attempts=80)
            if cyc:
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    leftover.discard(tuple(sorted((u, v))))
                pieces.append(cyc[:])
            else:
                # no cycle: take a single edge (prefer low-degree endpoint)
                e = min(leftover, key=lambda e: len(adj.get(e[0], set())) + len(adj.get(e[1], set())))
                leftover.discard(e)
                pieces.append(list(e))

        # ---- Phase C: local search to shrink leftover single-edges ----
        # pieces list: cycles (len>=3) and single edges (len==2)
        # We try to merge two single edges into a cycle, or absorb a single edge into a cycle.
        # Represent current state: edge set per piece.
        def piece_edges(p):
            es = []
            if len(p) >= 3:
                for i in range(len(p)):
                    u = p[i]; v = p[(i+1) % len(p)]
                    es.append(tuple(sorted((u, v))))
            else:
                es.append(tuple(sorted((p[0], p[1]))))
            return es

        # Build edge->piece index
        edge_to_piece = {}
        for idx, p in enumerate(pieces):
            for e in piece_edges(p):
                edge_to_piece[e] = idx

        improved = True
        while improved and time.time() < time_budget - 3:
            improved = False
            single_edges = [(idx, pieces[idx]) for idx, p in enumerate(pieces) if len(p) == 2]
            # try to form a cycle from 2 single edges sharing a vertex + a closing edge
            # Actually a 3-cycle needs 3 edges. Two single edges can't form a cycle alone.
            # Instead: try to convert a single edge (u,v) into a cycle by finding a path u..v in other pieces' edges.
            # Simpler, high-value move: find any 3-cycle in the union of leftover single edges + cycle edges
            # that is not already fully in one piece, and merge.
            # We'll do a targeted move: for each pair of single edges, see if they can be extended.
            # Keep it cheap: look for a triangle in the set of all current edges that spans 2+ pieces.
            all_edges = set()
            for idx, p in enumerate(pieces):
                for e in piece_edges(p):
                    all_edges.add(e)
            # candidate triangles
            adj2 = defaultdict(set)
            for u, v in all_edges:
                adj2[u].add(v)
                adj2[v].add(u)
            found = None
            for u in adj2:
                for v in adj2[u]:
                    if v <= u:
                        continue
                    for w in adj2[v]:
                        if w <= v:
                            continue
                        if w in adj2[u]:
                            e1 = tuple(sorted((u, v)))
                            e2 = tuple(sorted((v, w)))
                            e3 = tuple(sorted((u, w)))
                            p1 = edge_to_piece.get(e1)
                            p2 = edge_to_piece.get(e2)
                            p3 = edge_to_piece.get(e3)
                            # triangle spanning >=2 pieces -> merge reduces piece count
                            if p1 is not None and len({p1, p2, p3}) >= 2:
                                found = (u, v, w, [e1, e2, e3, p1, p2, p3])
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                u, v, w, info = found
                e1, e2, e3, p1, p2, p3 = info
                # remove edges from old pieces
                new_pieces = []
                for idx, p in enumerate(pieces):
                    pe = piece_edges(p)
                    pe = [e for e in pe if e not in (e1, e2, e3)]
                    if pe:
                        # rebuild piece from remaining edges
                        # if it was a cycle and we removed edges, it may break -> keep as list of edges? 
                        # To stay valid, if a cycle loses edges we must split into edges.
                        if len(p) >= 3 and len(pe) < len(p):
                            for e in pe:
                                new_pieces.append(list(e))
                        elif len(p) >= 3:
                            new_pieces.append(p)
                        else:
                            # single edge that survived
                            new_pieces.append(p)
                new_pieces.append([u, v, w])
                # rebuild edge_to_piece
                edge_to_piece = {}
                for idx, p in enumerate(new_pieces):
                    for e in piece_edges(p):
                        edge_to_piece[e] = idx
                pieces = new_pieces
                improved = True

        # ---- Phase D: final cleanup: any leftover edges ----
        for e in leftover:
            pieces.append(list(e))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 8:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        # budget split: give each graph a fair share
        remaining_budget = deadline - time.time()
        pieces = decompose_graph(edges, time.time() + remaining_budget / 3.0 - 1)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
