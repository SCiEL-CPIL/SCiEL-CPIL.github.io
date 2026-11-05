"""Clique partitioning for two fixed dense random graphs."""
import time
import random
import numpy as np
from scipy.sparse.linalg import eigsh

DEADLINE = 140  # seconds, safety margin before 150s kill


def _generate_graph(n, p, seed):
    """Generate a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _max_clique_bk(adj, n, active_mask, seed_vertex, deadline):
    """
    Find max clique containing seed_vertex in the subgraph induced by active_mask.
    Uses Bron-Kerbosch with pivot and coloring.
    """
    if active_mask == 0:
        return []

    # Ensure seed is in active
    if not (active_mask & (1 << seed_vertex)):
        return []

    best_clique = [seed_vertex]
    best_size = 1

    # Vertices in active_mask
    verts = []
    m = active_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        verts.append(idx)
        m ^= lsb

    if not verts:
        return []

    # Sort by degree in active_mask descending for better coloring/order
    verts.sort(key=lambda v: (adj[v] & active_mask).bit_count(), reverse=True)

    def color_sort(P_mask):
        """Greedy coloring of vertices in P_mask, return list of (color, vertex) sorted by color."""
        pm = P_mask
        verts_p = []
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            verts_p.append(idx)
            pm ^= lsb

        if not verts_p:
            return []

        # Sort by degree in P_mask
        verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

        v_color = {}
        result = []
        for v in verts_p:
            used = set()
            neigh = adj[v] & P_mask
            nm = neigh
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                if u in v_color:
                    used.add(v_color[u])
                nm ^= lsb
            c = 1
            while c in used:
                c += 1
            v_color[v] = c
            result.append((c, v))

        result.sort(key=lambda x: x[0])
        return result

    def bk(R, P, X):
        nonlocal best_clique, best_size

        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return

        if time.time() > deadline:
            return

        colored = color_sort(P)
        for color, v in reversed(colored):
            if len(R) + color <= best_size:
                break

            R.append(v)
            bk(R, P & adj[v], X & adj[v])
            R.pop()
            P ^= (1 << v)
            X |= (1 << v)

            if time.time() > deadline:
                return

    # Initialize with seed
    # P = neighbors of seed in active_mask
    # X = empty
    P = adj[seed_vertex] & active_mask
    X = 0
    R = [seed_vertex]
    
    bk(R, P, X)
    
    return best_clique


def _solve_instance(adj, n, deadline):
    """
    Partition all edges into cliques.
    Strategy: use spectral centrality to guide seed vertex selection,
    then find max clique containing that seed using BK.
    """
    rem = adj[:]  # Copy of adjacency for remaining edges
    cliques = []

    # Compute initial spectral centrality
    # Use adjacency matrix for eigsh
    A = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            A[u, v] = 1.0
            mask ^= lsb
    A = (A + A.T) / 2.0

    try:
        vals, vecs = eigsh(A, k=1, which='LA')
        centrality = np.abs(vecs[:, 0])
        if np.max(centrality) > 0:
            centrality = centrality / np.max(centrality)
        else:
            centrality = np.ones(n) / n
    except Exception:
        centrality = np.ones(n) / n

    # Track active vertices (those with at least one remaining edge)
    active_mask = 0
    for i in range(n):
        if rem[i]:
            active_mask |= (1 << i)

    while active_mask and time.time() < deadline:
        # Select seed vertex: highest centrality among active vertices
        best_seed = -1
        max_cent = -1.0
        
        m = active_mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if centrality[u] > max_cent:
                max_cent = centrality[u]
                best_seed = u
            m ^= lsb
        
        if best_seed == -1:
            break

        # Find max clique containing best_seed in the subgraph of active_mask
        clique = _max_clique_bk(rem, n, active_mask, best_seed, deadline)

        if not clique or len(clique) < 2:
            # Fallback: pick any neighbor
            nb = rem[best_seed] & active_mask
            if nb:
                lsb = nb & -nb
                v = lsb.bit_length() - 1
                clique = [best_seed, v]
            else:
                # Should not happen if active_mask is correct, but safety
                break

        cliques.append(clique)

        # Remove edges of the clique from rem
        for u in clique:
            for v in clique:
                if u != v:
                    rem[u] &= ~(1 << v)

        # Update active_mask
        active_mask = 0
        for i in range(n):
            if rem[i]:
                active_mask |= (1 << i)

        # If edge density drops significantly, recompute centrality
        # This adapts the ordering to the residual graph structure
        total_edges = sum(rem[i].bit_count() for i in range(n)) // 2
        if total_edges < 0.4 * (n * (n-1) // 2):
            # Recompute centrality on remaining graph
            A_rem = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                mask = rem[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    A_rem[u, v] = 1.0
                    mask ^= lsb
            A_rem = (A_rem + A_rem.T) / 2.0
            try:
                vals, vecs = eigsh(A_rem, k=1, which='LA')
                centrality = np.abs(vecs[:, 0])
                if np.max(centrality) > 0:
                    centrality = centrality / np.max(centrality)
                else:
                    centrality = np.ones(n) / n
            except Exception:
                pass

    # Fallback: cover remaining edges with pairs if time runs out or logic fails
    if active_mask:
        # Iterate through remaining edges
        for u in range(n):
            nb = rem[u]
            while nb:
                lsb = nb & -nb
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
                nb ^= lsb

    return cliques


def construct():
    """
    Return two lists of cliques, one per instance.
    Each clique is a list of vertices forming a clique.
    Every edge appears in exactly one clique.
    Minimize total number of cliques.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj = _generate_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
        if time.time() > deadline:
            # If we ran out of time, we need to ensure validity
            # For now, just return what we have
            break

    # If we didn't process both instances, fill in with edge-by-edge
    while len(results) < 2:
        idx = len(results)
        n, p, seed = ((32, 0.6, 601), (64, 0.55, 602))[idx]
        adj = _generate_graph(n, p, seed)
        # Simple fallback: each edge as its own clique
        edge_cliques = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edge_cliques.append([u, v])
                mask ^= lsb
        results.append(edge_cliques)

    return results


if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total cliques: {len(res[0]) + len(res[1])}")
