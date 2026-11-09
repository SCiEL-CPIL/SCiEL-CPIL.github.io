# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum square-difference-free set modulo 205 using a bitset-based
    branch-and-bound Maximum Independent Set solver with a valid greedy coloring bound.

    The forbidden differences are the nonzero quadratic residues mod 205.
    We model the problem as finding a maximum independent set in the Cayley
    graph Cay(Z/205Z, F) where F is the set of forbidden differences.
    """
    m = 205
    k = 2

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build adjacency bitsets using Python's arbitrary-precision integers as bitsets
    # adj[i] is a bitset where bit j is set if i and j are adjacent (forbidden difference)
    adj = [0] * m
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
    # Remove self-loops
    for i in range(m):
        adj[i] &= ~(1 << i)

    # Precompute full mask
    full_mask = (1 << m) - 1

    # Greedy coloring of the complement graph to get a valid upper bound on MIS size
    # Equivalent to finding a sequence of maximal cliques in G that cover V
    def greedy_color_bound(candidates_bitset):
        """
        Returns an upper bound on the size of the Maximum Independent Set in G[candidates].
        This is done by computing a greedy coloring of the complement graph \bar{G}[candidates].
        The number of colors is an upper bound on the clique number of \bar{G}, which is the
        independence number of G.
        
        A color class in \bar{G} is an independent set in \bar{G}, which is a clique in G.
        So we greedily find maximal cliques in G[candidates].
        """
        if candidates_bitset == 0:
            return 0
        
        remaining = candidates_bitset
        color_count = 0
        
        while remaining != 0:
            # Find a maximal clique in G induced by 'remaining'
            clique = 0
            temp = remaining
            while temp != 0:
                # Pick the lowest set bit (or any vertex)
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                
                # Check if v can be added to the clique
                # v must be adjacent to all vertices already in the clique
                if clique == 0:
                    # First vertex, always add
                    clique |= lb
                    # Restrict future candidates to neighbors of v (within 'temp')
                    # Any vertex not adjacent to v cannot be in the same clique
                    temp &= adj[v]
                else:
                    # v must be adjacent to all vertices in clique
                    # (adj[v] & clique) == clique
                    if (adj[v] & clique) == clique:
                        clique |= lb
                        # Restrict future candidates to neighbors of v
                        temp &= adj[v]
                
                # Remove v from temp (it's processed for this color class)
                temp &= ~(1 << v)
                
                # If we added v, we already restricted temp to adj[v].
                # If we didn't add v, we just skip it.
                # Wait, if we don't add v, should we remove it? Yes, it's not in this clique.
                # But should we remove its non-neighbors? No, they might be compatible with other clique members.
                # Actually, the standard greedy maximal clique algorithm:
                # If v is compatible, add it and intersect candidate set with N(v).
                # If v is not compatible, just discard v from consideration for this clique.
                
                # The line `temp &= ~(1 << v)` handles removing v.
                # The line `temp &= adj[v]` inside the if handles pruning.
                
            color_count += 1
            # Remove the clique from remaining
            remaining &= ~clique
        
        return color_count

    # Branch and bound
    best_size = 0
    best_set = 0  # bitset representation of the best independent set found

    def branch_and_bound(candidates_bitset, current_set, current_size):
        nonlocal best_size, best_set
        
        if candidates_bitset == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Upper bound: current_size + max possible from candidates
        # Use greedy coloring bound on complement (valid upper bound for MIS)
        ub = current_size + greedy_color_bound(candidates_bitset)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with highest degree in the candidate subgraph
        temp = candidates_bitset
        best_v = -1
        best_deg = -1
        while temp != 0:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            # Degree in candidate subgraph
            deg = bin(adj[v] & candidates_bitset).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # Remove best_v and all its neighbors from candidates
        new_candidates = candidates_bitset & ~(1 << best_v) & ~adj[best_v]
        new_set = current_set | (1 << best_v)
        branch_and_bound(new_candidates, new_set, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_candidates = candidates_bitset & ~(1 << best_v)
        branch_and_bound(new_candidates, current_set, current_size)

    # Initialize with a greedy solution to get a good lower bound
    # Greedy: pick vertices in order, add if no conflict
    greedy_set = 0
    for v in range(m):
        if (adj[v] & greedy_set) == 0:
            greedy_set |= (1 << v)
    best_size = bin(greedy_set).count('1')
    best_set = greedy_set

    # Run branch and bound
    full_candidates = (1 << m) - 1
    branch_and_bound(full_candidates, 0, 0)

    # Extract the best set
    result = []
    for v in range(m):
        if (best_set >> v) & 1:
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
