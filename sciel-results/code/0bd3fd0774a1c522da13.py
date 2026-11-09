import time
from math import gcd
import random

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.5


def _check_pair(r1, m1, r2, m2):
    """Two residue classes are disjoint iff r1 mod gcd(m1,m2) != r2 mod gcd(m1,m2)."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g


def _is_valid_solution(solution):
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if not _check_pair(solution[i][0], solution[i][1], solution[j][0], solution[j][1]):
                return False
    return True


def _build_conflict_graph():
    """
    Build node list and conflict bitsets.
    Node index: for modulus m (2..128), residue r in [0, m):
        idx = (m-2)*256 + r   (256 >= max modulus 128)
    conflict_bits[i] is a Python int whose bit j is set iff nodes i and j conflict.
    """
    nodes = []           # list of (r, m)
    idx_of = {}          # (r, m) -> node index
    for m in range(2, MAXIMUM + 1):
        for r in range(m):
            idx = (m - 2) * 256 + r
            nodes.append((r, m))
            idx_of[(r, m)] = idx
    N = len(nodes)
    conflict_bits = [0] * N
    for i in range(N):
        r1, m1 = nodes[i]
        for j in range(i + 1, N):
            r2, m2 = nodes[j]
            if m1 == m2:
                conflict = True
            else:
                conflict = not _check_pair(r1, m1, r2, m2)
            if conflict:
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    return nodes, idx_of, conflict_bits, N


def _greedy_seed(deadline):
    """Dynamic greedy: repeatedly pick the modulus with the fewest valid residues (most constrained)."""
    solution = []
    remaining = list(range(2, MAXIMUM + 1))
    while remaining and time.time() < deadline:
        best_m = None
        best_r = None
        best_count = None
        for m in remaining:
            valid_count = 0
            first_valid = None
            for r in range(m):
                ok = True
                for (rp, mp) in solution:
                    if not _check_pair(r, m, rp, mp):
                        ok = False
                        break
                if ok:
                    valid_count += 1
                    if first_valid is None:
                        first_valid = r
            if valid_count > 0:
                if best_count is None or valid_count < best_count:
                    best_count = valid_count
                    best_m = m
                    best_r = first_valid
        if best_m is None:
            break
        solution.append([best_r, best_m])
        remaining.remove(best_m)
    return solution


def _branch_and_bound(nodes, idx_of, conflict_bits, N, initial_solution, deadline):
    """
    Maximum independent set search on the conflict graph, one node per (r,m).
    An independent set = set of nodes with no conflicts = pairwise disjoint residue classes.
    We also need distinct moduli; conflicts already include same-modulus pairs, so
    an independent set automatically has distinct moduli.
    Bitset-accelerated branch and bound.
    """
    best_size = len(initial_solution)
    best_set = set(idx_of[(r, m)] for (r, m) in initial_solution)
    best_nodes = list(best_set)

    # Order nodes: a good static order helps. Sort by (modulus, residue) but we
    # process in an order that tends to find large sets early.
    # We'll use a fixed processing order = node index order for determinism,
    # but the branching uses dynamic candidate selection.
    # For speed, precompute nothing else; use bit operations.

    # We implement a recursive B&B over the node order.
    # State: P = set of candidate nodes (as a Python int bitset) that are all
    #         compatible with the chosen set and with each other's future choices.
    #         chosen = set of chosen node indices.
    # We branch on a node in P: include it (remove its conflicts from P) or exclude it.

    # To keep it manageable, we use a simple but effective B&B:
    #   - Greedy upper bound: size of P (trivial) plus a better bound via a
    #     greedy independent-set on P.
    #   - Branch on the node in P with highest degree within P (most constrained).

    def greedy_is_size(P):
        """Greedy independent set size within candidate set P (bitset)."""
        count = 0
        cand = P
        while cand:
            # pick node with max degree in cand
            # find a node
            # pick the lowest set bit for speed, but prefer high-degree for bound quality
            # For a bound, any maximal independent set works; use low-bit for speed.
            lb = cand & (-cand)
            node = lb.bit_length() - 1
            count += 1
            # remove node and its neighbors
            cand &= ~(lb | conflict_bits[node])
            cand &= cand  # keep only bits in original cand
            # actually we must mask to cand: cand = cand & ~(lb | conflict_bits[node])
            cand = cand & ~(lb | conflict_bits[node])
        return count

    # Precompute: for each node, its conflict bitset (already have conflict_bits).

    def search(P, chosen_count, chosen_set):
        nonlocal best_size, best_set, best_nodes
        if time.time() > deadline:
            return
        if P == 0:
            if chosen_count > best_size:
                best_size = chosen_count
                best_set = set(chosen_set)
                best_nodes = list(best_set)
            return
        # Upper bound: chosen_count + greedy independent set size of P
        ub = chosen_count + greedy_is_size(P)
        if ub <= best_size:
            return
        # Choose branching node: highest degree within P
        # Find node in P with max (popcount of P & conflict_bits[node])
        # For speed, iterate set bits of P.
        cand = P
        best_node = -1
        best_deg = -1
        # Limit the scan: if P is large, scanning all bits is costly.
        # We'll scan but cap: if too many, just pick a representative.
        # Count bits in P.
        # Use a heuristic: pick the node with the largest index? No.
        # Let's scan up to a cap of candidate nodes for degree computation.
        # We'll scan all set bits but break early if P has many.
        bitcount = P.bit_count()
        if bitcount > 200:
            # P is large; pick a few candidates to avoid slow degree scan
            # Just pick the lowest set bit as a reasonable branch.
            lb = P & (-P)
            best_node = lb.bit_length() - 1
        else:
            c = P
            while c:
                lb = c & (-c)
                node = lb.bit_length() - 1
                deg = (P & conflict_bits[node]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
                c &= c - 1

        node = best_node
        node_bit = 1 << node
        # Branch 1: include node
        new_P = P & ~(node_bit | conflict_bits[node])
        chosen_set.add(node)
        search(new_P, chosen_count + 1, chosen_set)
        chosen_set.discard(node)
        # Branch 2: exclude node
        new_P2 = P & ~node_bit
        search(new_P2, chosen_count, chosen_set)

    # Initial candidate set: all nodes compatible with the initial solution.
    # A node is compatible with the initial solution iff it does not conflict with
    # any chosen node. Also, the initial solution nodes are already chosen.
    # So P = all nodes that are (a) not in the initial solution and (b) compatible
    # with every node in the initial solution.
    P = 0
    init_set = set(idx_of[(r, m)] for (r, m) in initial_solution)
    for i in range(N):
        if i in init_set:
            continue
        compatible = True
        for j in init_set:
            if (conflict_bits[i] >> j) & 1:
                compatible = False
                break
        if compatible:
            P |= (1 << i)

    search(P, len(initial_solution), set(init_set))

    # Convert best_set (node indices) back to (r, m) pairs
    result = [list(nodes[i]) for i in best_nodes]
    return result


def _local_search_add(solution, deadline):
    """Try to add more moduli to the solution by scanning unused moduli."""
    best = [list(x) for x in solution]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used = set(m for _, m in best)
        for m in range(2, MAXIMUM + 1):
            if m in used:
                continue
            for r in range(m):
                ok = True
                for (rp, mp) in best:
                    if not _check_pair(r, m, rp, mp):
                        ok = False
                        break
                if ok:
                    best.append([r, m])
                    improved = True
                    break
            if improved:
                break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Build conflict graph (one-time cost, ~8256^2/2 checks, fast enough)
    nodes, idx_of, conflict_bits, N = _build_conflict_graph()

    # Phase 1: dynamic greedy seed
    seed_solution = _greedy_seed(deadline)

    # Phase 2: local search to add more moduli to the seed
    seed_solution = _local_search_add(seed_solution, deadline)

    if not seed_solution:
        return [[0, 2]]

    # Phase 3: branch and bound to find the global maximum
    try:
        bb_solution = _branch_and_bound(nodes, idx_of, conflict_bits, N, seed_solution, deadline)
        if len(bb_solution) > len(seed_solution):
            seed_solution = bb_solution
    except Exception:
        pass

    # Final local search polish
    seed_solution = _local_search_add(seed_solution, deadline)

    if not seed_solution:
        return [[0, 2]]

    return seed_solution
