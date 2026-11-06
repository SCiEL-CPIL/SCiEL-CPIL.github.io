import time

# EVOLVE-BLOCK-START

def construct():
    d = 7
    n = 1 << d          # 128 vertices
    deadline = time.time() + 140.0   # safety margin under 150s kill

    # ---- Precompute all edges as (u, v, bit) with u < v ----
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y, i))
    E = len(edges)  # 448

    # edge index by (min, max)
    edge_index = {}
    for idx, (u, v, b) in enumerate(edges):
        edge_index[(u, v)] = idx

    # For each edge, the 6 "partner" edges that with it form a 4-cycle.
    # A 4-cycle uses two dimensions i and j. Given edge (x,y) in dim i,
    # the opposite edge is (x^bitj, y^bitj) in dim i. The two "side" edges
    # connect x-x^bitj and y-y^bitj.
    # Actually a 4-cycle = {e, opp, side1, side2}. We need all 4.
    # Store for each edge the set of edges that form a C4 with it (3 others).
    # But a C4 has 4 edges; for a given edge, there are (d-1) possible C4s.
    # For each such C4, the other 3 edges. We store the 3-edge complement.
    # We'll store c4_complement[e] = list of (3-tuples) of edge indices.
    c4_comp = [[] for _ in range(E)]
    for idx, (x, y, i) in enumerate(edges):
        for j in range(d):
            if j == i:
                continue
            bi = 1 << i
            bj = 1 << j
            # opposite edge in dim i: (x^bj, y^bj)
            ox, oy = x ^ bj, y ^ bj
            o = (min(ox, oy), max(ox, oy))
            # side edges: (x, x^bj) dim j and (y, y^bj) dim j
            s1 = (min(x, x ^ bj), max(x, x ^ bj))
            s2 = (min(y, y ^ bj), max(y, y ^ bj))
            comp = [edge_index[o], edge_index[s1], edge_index[s2]]
            c4_comp[idx].append(comp)

    # ---- Greedy construction ----
    selected = [False] * E
    # count how many selected partners in each C4 for quick check
    # We'll just do the standard check: edge can be added if no C4 has all 3 comps selected.
    chosen = 0
    for idx in range(E):
        # can we add?
        ok = True
        for comp in c4_comp[idx]:
            a, b, c = comp
            if selected[a] and selected[b] and selected[c]:
                ok = False
                break
        if ok:
            selected[idx] = True
            chosen += 1

    # ---- Local search polish ----
    # Represent selected as a set of indices for fast membership.
    sel_set = set(i for i in range(E) if selected[i])

    def can_add(idx):
        for comp in c4_comp[idx]:
            a, b, c = comp
            if a in sel_set and b in sel_set and c in sel_set:
                return False
        return True

    def add_edge(idx):
        sel_set.add(idx)
        selected[idx] = True

    def remove_edge(idx):
        sel_set.discard(idx)
        selected[idx] = False

    best_count = len(sel_set)

    # Build list of currently-forbidden (unselected) edges for addition attempts
    # We iterate improvement rounds.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1) single-edge additions
        for idx in range(E):
            if selected[idx]:
                continue
            if can_add(idx):
                add_edge(idx)
                best_count += 1
                improved = True
        if time.time() >= deadline:
            break
        # 2) 2-for-1 swaps: remove 2 selected, add 3 unselected -> net +1
        #    Try pairs of selected edges to remove, then see how many unselected
        #    become addable. We do a lightweight version: for each pair of selected
        #    edges that share a C4, removing them may free additions.
        #    To keep it fast, we sample.
        sel_list = list(sel_set)
        # try to find a beneficial 2-for-1: remove 2, add 3
        # We'll iterate over unselected edges that are "blocked" by exactly 2 selected comps,
        # then see if removing those 2 unlocks 3 additions.
        # Simpler heuristic: for each unselected edge, find the minimal set of selected
        # edges whose removal makes it addable (its blocking comps). Try removing up to 2.
        for idx in range(E):
            if selected[idx]:
                continue
            # collect blocking selected edges
            blockers = set()
            for comp in c4_comp[idx]:
                for a in comp:
                    if a in sel_set:
                        blockers.add(a)
            if len(blockers) > 2:
                continue
            # try removing all blockers (<=2)
            if not blockers:
                continue
            bl = list(blockers)
            # save
            saved = []
            for b in bl:
                remove_edge(b)
                saved.append(b)
            # now try to add idx and any other unselected edges that become addable
            gained = 0
            newly_added = []
            if can_add(idx):
                add_edge(idx)
                gained += 1
                newly_added.append(idx)
            # scan a few other unselected edges
            for idx2 in range(E):
                if idx2 in sel_set:
                    continue
                if can_add(idx2):
                    add_edge(idx2)
                    gained += 1
                    newly_added.append(idx2)
            net = gained - len(bl)
            if net > 0:
                best_count += net
                improved = True
            else:
                # revert
                for a in newly_added:
                    remove_edge(a)
                for b in bl:
                    add_edge(b)
        if time.time() >= deadline:
            break
        # 3) 3-for-1 swaps: remove 3, add 4
        for idx in range(E):
            if selected[idx]:
                continue
            blockers = set()
            for comp in c4_comp[idx]:
                for a in comp:
                    if a in sel_set:
                        blockers.add(a)
            if len(blockers) > 3:
                continue
            if len(blockers) < 3:
                continue
            bl = list(blockers)
            for b in bl:
                remove_edge(b)
            gained = 0
            newly_added = []
            if can_add(idx):
                add_edge(idx)
                gained += 1
                newly_added.append(idx)
            for idx2 in range(E):
                if idx2 in sel_set:
                    continue
                if can_add(idx2):
                    add_edge(idx2)
                    gained += 1
                    newly_added.append(idx2)
            net = gained - len(bl)
            if net > 0:
                best_count += net
                improved = True
            else:
                for a in newly_added:
                    remove_edge(a)
                for b in bl:
                    add_edge(b)
        if time.time() >= deadline:
            break

    # ---- Final: build output list ----
    out = []
    for idx in range(E):
        if selected[idx]:
            u, v, b = edges[idx]
            out.append([u, v])
    return out

# EVOLVE-BLOCK-END
