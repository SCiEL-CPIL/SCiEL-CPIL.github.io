import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    n = 20
    rng = random.Random(42)
    
    # Precompute all triples for efficiency
    all_triples = list(combinations(range(n), 3))
    
    def count_k33s(adj_bits):
        """Count the number of K3,3 subgraphs in the graph.
        A K3,3 is formed by two disjoint triples A, B such that all 9 edges between them exist.
        To avoid double counting (A,B) and (B,A), we can just count all pairs of disjoint triples
        where all cross-edges exist, and divide by 2 at the end, or just use the raw count as energy.
        For speed, we count for each triple A, the number of common neighbors C_A.
        Then the number of triples B in C_A that are disjoint from A is what we need?
        No, B must be a subset of C_A. And B must be disjoint from A.
        Since C_A consists of neighbors of all vertices in A, and vertices in A are not neighbors of themselves,
        C_A is automatically disjoint from A (in a simple graph without loops).
        So for each triple A, let c_A = |C_A|.
        The number of triples B in C_A is C(c_A, 3).
        Each K3,3 {A, B} is counted twice: once as (A, B) and once as (B, A).
        So total K3,3s = 0.5 * sum_{A in all_triples} C(c_A, 3).
        """
        total = 0
        for a1, a2, a3 in all_triples:
            common = adj_bits[a1] & adj_bits[a2] & adj_bits[a3]
            c = common.bit_count()
            if c >= 3:
                # C(c, 3) = c * (c-1) * (c-2) / 6
                total += c * (c - 1) * (c - 2) // 6
        return total // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def get_edges(adj_bits):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def get_missing_edges(adj_bits):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj_bits[i] >> j) & 1):
                    edges.append((i, j))
        return edges

    # Strategy: Simulated Annealing
    # 1. Initialize with a dense graph (e.g., K20)
    # 2. Compute initial energy (K3,3 count)
    # 3. Iterate:
    #    - Pick a random move: add a missing edge OR remove an existing edge
    #    - Compute delta energy
    #    - Accept if delta < 0 or with probability exp(-delta / T)
    #    - Cool T over time
    
    # We need to efficiently compute delta energy.
    # When adding edge (u, v):
    #   Only triples containing u or v can have their common neighbor count change.
    #   Specifically, for any triple A containing u, if v is in C_A, then c_A increases by 1.
    #   But v is in C_A iff v is connected to all vertices in A.
    #   This is still a bit expensive to check for all triples containing u or v.
    #   Number of triples containing u is C(19, 2) = 171.
    #   So for each move, we check ~342 triples.
    #   For each such triple, we check if the new edge affects its common neighbor count.
    #   If we add (u, v):
    #     For a triple A containing u:
    #       v is in C_A iff v is connected to all other two vertices in A.
    #       If so, c_A increases by 1.
    #     Similarly for triples containing v.
    #   We need to update the sum of C(c_A, 3).
    #   Let f(c) = C(c, 3).
    #   Delta = f(c_A + 1) - f(c_A) for each affected triple.
    #   f(c+1) - f(c) = C(c+1, 3) - C(c, 3) = C(c, 2).
    #   So delta = sum of C(c_A, 2) for all affected triples A where the condition holds.
    
    # Let's implement this incremental update.
    
    def delta_energy_add(adj_bits, u, v, c_counts):
        """Compute change in K3,3 count when adding edge (u, v).
        c_counts[i] is the common neighbor count for triple i.
        """
        delta = 0
        # Triples containing u
        for idx, A in enumerate(all_triples):
            if u in A:
                # Check if v is in C_A
                # v is in C_A iff v is connected to all vertices in A
                ok = True
                for a in A:
                    if a == u:
                        continue
                    if not ((adj_bits[v] >> a) & 1):
                        ok = False
                        break
                if ok:
                    c = c_counts[idx]
                    delta += c * (c - 1) // 2  # C(c, 2)
            elif v in A:
                # Check if u is in C_A
                ok = True
                for a in A:
                    if a == v:
                        continue
                    if not ((adj_bits[u] >> a) & 1):
                        ok = False
                        break
                if ok:
                    c = c_counts[idx]
                    delta += c * (c - 1) // 2
        return delta // 2

    def delta_energy_remove(adj_bits, u, v, c_counts):
        """Compute change in K3,3 count when removing edge (u, v).
        This is the negative of delta_energy_add, but we must check the state BEFORE removal.
        Actually, it's easier to just compute the change.
        When removing (u, v), c_A decreases by 1 for triples A containing u (or v) where the other vertex is in C_A.
        Delta = - C(c_A - 1, 2) for each such triple?
        f(c) - f(c-1) = C(c, 3) - C(c-1, 3) = C(c-1, 2).
        So delta = - sum C(c_A - 1, 2).
        """
        delta = 0
        for idx, A in enumerate(all_triples):
            if u in A:
                ok = True
                for a in A:
                    if a == u:
                        continue
                    if not ((adj_bits[v] >> a) & 1):
                        ok = False
                        break
                if ok:
                    c = c_counts[idx]
                    delta -= (c - 1) * (c - 2) // 2  # C(c-1, 2)
            elif v in A:
                ok = True
                for a in A:
                    if a == v:
                        continue
                    if not ((adj_bits[u] >> a) & 1):
                        ok = False
                        break
                if ok:
                    c = c_counts[idx]
                    delta -= (c - 1) * (c - 2) // 2
        return delta // 2

    # Initialize with K20
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            add_edge(adj_bits, i, j)
    
    # Compute initial c_counts
    c_counts = [0] * len(all_triples)
    for idx, (a1, a2, a3) in enumerate(all_triples):
        common = adj_bits[a1] & adj_bits[a2] & adj_bits[a3]
        c_counts[idx] = common.bit_count()
    
    current_energy = count_k33s(adj_bits)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.9995
    best_adj = adj_bits[:]
    best_energy = current_energy
    
    # Precompute list of all edges and missing edges? No, it changes.
    # We'll pick a random pair (u, v) and decide to add or remove based on current state.
    
    steps = 0
    while time.time() < deadline - 5:
        # Pick a random pair
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        has_edge = (adj_bits[u] >> v) & 1
        
        if has_edge:
            # Try to remove
            de = delta_energy_remove(adj_bits, u, v, c_counts)
            # Acceptance
            if de < 0 or rng.random() < np.exp(-de / T):
                # Accept
                remove_edge(adj_bits, u, v)
                # Update c_counts
                for idx, A in enumerate(all_triples):
                    if u in A:
                        ok = True
                        for a in A:
                            if a == u:
                                continue
                            if not ((adj_bits[v] >> a) & 1):
                                ok = False
                                break
                        if ok:
                            c_counts[idx] -= 1
                    elif v in A:
                        ok = True
                        for a in A:
                            if a == v:
                                continue
                            if not ((adj_bits[u] >> a) & 1):
                                ok = False
                                break
                        if ok:
                            c_counts[idx] -= 1
                current_energy += de
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj_bits[:]
        else:
            # Try to add
            de = delta_energy_add(adj_bits, u, v, c_counts)
            if de < 0 or rng.random() < np.exp(-de / T):
                # Accept
                add_edge(adj_bits, u, v)
                # Update c_counts
                for idx, A in enumerate(all_triples):
                    if u in A:
                        ok = True
                        for a in A:
                            if a == u:
                                continue
                            if not ((adj_bits[v] >> a) & 1):
                                ok = False
                                break
                        if ok:
                            c_counts[idx] += 1
                    elif v in A:
                        ok = True
                        for a in A:
                            if a == v:
                                continue
                            if not ((adj_bits[u] >> a) & 1):
                                ok = False
                                break
                        if ok:
                            c_counts[idx] += 1
                current_energy += de
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj_bits[:]
        
        T *= cooling_rate
        steps += 1
        if steps % 10000 == 0:
            # Optional: restart if stuck?
            pass

    # Final check: if best_energy > 0, we need to repair.
    # But the task requires a K3,3-free graph.
    # If SA didn't reach 0, we must fall back to a valid graph.
    # We can use the "deletion" heuristic on the best_adj found by SA.
    
    if best_energy > 0:
        adj_bits = best_adj[:]
        # Greedy removal
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            # Find a K3,3
            k33_found = None
            for a1, a2, a3 in all_triples:
                common = adj_bits[a1] & adj_bits[a2] & adj_bits[a3]
                verts_in_common = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    verts_in_common.append(v)
                    mask ^= lsb
                
                verts_in_common = [v for v in verts_in_common if v not in (a1, a2, a3)]
                
                if len(verts_in_common) >= 3:
                    b1, b2, b3 = verts_in_common[0], verts_in_common[1], verts_in_common[2]
                    k33_found = ((a1, a2, a3), (b1, b2, b3))
                    break
            if not k33_found:
                break
            A, B = k33_found
            a1, a2, a3 = A
            b1, b2, b3 = B
            u, v = rng.choice([(a1,b1), (a1,b2), (a1,b3), (a2,b1), (a2,b2), (a2,b3), (a3,b1), (a3,b2), (a3,b3)])
            remove_edge(adj_bits, u, v)
        
        # Verify it's free
        if count_k33s(adj_bits) > 0:
            # Fallback to a known valid construction: K2,18 bipartite
            adj_bits = [0] * n
            for i in range(2):
                for j in range(2, 20):
                    add_edge(adj_bits, i, j)
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
