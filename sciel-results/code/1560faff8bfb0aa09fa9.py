"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        # Build cross-neighborhood: for each i in [k, n), which K-vertices it connects to
        rng = Random(seed)
        # We must replicate the exact edge generation to know cross edges
        # K-internal edges: all pairs in [0,k)
        # Cross edges: for v in [k,n), for u in [0,k), edge if rng.random() < p
        # Note: rng is called in a specific order: first K-internal (no rng), then cross
        # Actually K-internal edges don't use rng. Cross edges use rng in order:
        # for v in range(k, n): for u in range(k): rng.random()
        cross = {}  # i -> set of K-neighbors
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            if nbrs:
                cross[v] = nbrs

        active = list(cross.keys())
        # U = union of all N_K(i)
        U = set()
        for i in active:
            U |= cross[i]

        # W = K-vertices that have at least one uncovered K-edge
        # Edge (a,b) in K is uncovered if no active i has both a,b in cross[i]
        # A vertex v is in W if there exists u != v such that (v,u) uncovered
        # (v,u) uncovered iff not exists i with v in cross[i] and u in cross[i]
        # Precompute for each K-vertex v, the set of active i that contain v
        # Then (v,u) covered iff exists i with v in cross[i] and u in cross[i]
        # i.e., the intersection of {i: v in cross[i]} and {i: u in cross[i]} is non-empty

        # For each K-vertex, list of active i containing it
        k_to_i = {v: [] for v in range(k)}
        for i in active:
            for v in cross[i]:
                k_to_i[v].append(i)

        # W: vertex v is in W if exists u != v with no common i
        W = set()
        for v in range(k):
            if not k_to_i[v]:
                # v not in any cross[i], so all K-edges incident to v are uncovered
                # v is in W if k >= 2 (there's another vertex)
                if k >= 2:
                    W.add(v)
                continue
            # Check if v has any u with no common i
            # v's i-set: k_to_i[v]
            # For v to NOT be in W, every u != v must share an i with v
            # Equivalently, union of k_to_i[u] for u != v covers... no.
            # (v,u) covered iff k_to_i[v] ∩ k_to_i[u] != empty
            # v in W iff exists u != v with k_to_i[v] ∩ k_to_i[u] == empty
            # If k_to_i[v] is non-empty, and for all u != v, k_to_i[u] intersects k_to_i[v]
            # We can check: is there any u whose i-set is disjoint from k_to_i[v]?
            # Optimization: if k_to_i[v] contains all active i, then it intersects everything
            all_i = set(active)
            if set(k_to_i[v]) == all_i:
                continue  # v intersects all
            # Check each u
            found = False
            for u in range(k):
                if u == v:
                    continue
                if not k_to_i[u]:
                    found = True
                    break
                # check intersection
                # small sets, just do it
                if not (set(k_to_i[v]) & set(k_to_i[u])):
                    found = True
                    break
            if found:
                W.add(v)

        # Build cliques
        cliques = []
        # I-vertex cliques
        for i in active:
            clique = [i] + sorted(cross[i])
            cliques.append(clique)
        # K-clique for uncovered
        if W:
            W_sorted = sorted(W)
            if len(W_sorted) >= 2:
                cliques.append(W_sorted)
            else:
                # W has 1 vertex, but a single vertex isn't a clique (need >=2)
                # But if W has 1 vertex v, then v has uncovered edges to others
                # Wait, if W={v}, then v has uncovered edges to some u, but u not in W?
                # Contradiction: if (v,u) uncovered, then u also in W (symmetric)
                # So W can't have exactly 1 vertex. Let's handle: if len(W)==1, 
                # actually it shouldn't happen. But to be safe, if there are uncovered edges,
                # we need a clique. Let me reconsider.
                # Actually if W has 1 vertex, that vertex has no uncovered edges (since 
                # uncovered is symmetric). So W with 1 vertex means no uncovered edges incident,
                # contradiction with how W is built. So W is either empty or >=2.
                pass

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Fallback: return seed-style answer (each edge as clique) to be valid
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            result.append([list(e) for e in edges])
        else:
            cliques = solve_one(*spec)
            # Validate: each clique is a list of >=2 distinct ints
            # And every edge is in exactly one clique
            # Let's verify quickly
            n, k, p, seed = spec
            all_edges = set(tuple(sorted(e)) for e in instance(n, k, p, seed))
            covered = set()
            valid = True
            for c in cliques:
                if len(c) < 2:
                    valid = False
                    break
                if len(set(c)) != len(c):
                    valid = False
                    break
                # Check clique property: all pairs in c must be edges
                for a in range(len(c)):
                    for b in range(a + 1, len(c)):
                        e = tuple(sorted((c[a], c[b])))
                        if e not in all_edges:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                # Add all pairs to covered
                for a in range(len(c)):
                    for b in range(a + 1, len(c)):
                        e = tuple(sorted((c[a], c[b])))
                        if e in covered:
                            valid = False  # edge in two cliques
                            break
                        covered.add(e)
                    if not valid:
                        break
            if not valid or covered != all_edges:
                # Fallback to seed
                n, k, p, seed = spec
                edges = instance(n, k, p, seed)
                result.append([list(e) for e in edges])
            else:
                result.append(cliques)

    return result
# EVOLVE-BLOCK-END
