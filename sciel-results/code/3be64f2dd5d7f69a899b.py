import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)
    
    # Node index: (m_idx, residue) -> global index
    # Total nodes = sum of moduli = sum(2..128) = 128*129/2 - 1 = 8256 - 1 = 8255
    # That's a lot. Let me reconsider.
    
    # Actually sum from 2 to 128 = (2+128)*127/2 = 130*127/2 = 8255
    # 8255 nodes, MIS on 8255 nodes with bitsets... might be too slow for exact.
    
    # Let me use a smarter approach: the structure by gcd.
    # For each modulus m, residues r and r' are "equivalent" for conflict purposes
    # if for all n, r%gcd(m,n) == r'%gcd(m,n). This is true iff r ≡ r' (mod m)
    # which is always true for same m. So no reduction within a modulus.
    
    # Alternative: think in terms of "which moduli to include" and "which residue to pick".
    # For a fixed set of moduli S, can we pick residues so all pairs are disjoint?
    # This is a constraint satisfaction: for each pair (m,n) in S with gcd g > 1,
    # we need r_m % g ≠ r_n % g.
    
    # This is like graph coloring on the "gcd graph" where edges labeled by gcd value.
    
    # Let me use branch and bound on the set of moduli, with feasibility check.
    
    # First, build the gcd matrix
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    gcd_mat = [[0]*n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i+1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # For a set of moduli, check if feasible assignment exists.
    # Use backtracking with constraint propagation.
    
    # For branch and bound on moduli:
    # Order moduli by "constrainedness" - moduli with many gcd>1 connections first.
    
    # Let me try a different approach: since the problem is structured,
    # use the fact that we want max independent set in the conflict graph.
    
    # Build conflict graph as bitsets. 8255 nodes, each with a bitset of conflicts.
    # 8255 bits = ~1032 bytes per node, total ~8.5 MB. Feasible.
    
    # Node ordering: group by modulus
    node_to_mod = []  # node_idx -> mod_idx
    mod_to_nodes = []  # mod_idx -> list of node indices
    offset = [0]
    for m_idx in range(n_mod):
        m = moduli[m_idx]
        mod_to_nodes.append(list(range(offset[-1], offset[-1] + m)))
        node_to_mod.extend([m_idx] * m)
        offset.append(offset[-1] + m)
    total_nodes = offset[-1]
    
    # Build conflict bitsets
    # conflicts[i] is a Python int (bitmask) of nodes conflicting with i
    conflicts = [0] * total_nodes
    for i in range(total_nodes):
        m_idx_i = node_to_mod[i]
        m_i = moduli[m_idx_i]
        r_i = i - offset[m_idx_i]
        for j in range(i+1, total_nodes):
            m_idx_j = node_to_mod[j]
            m_j = moduli[m_idx_j]
            r_j = j - offset[m_idx_j]
            g = gcd_mat[m_idx_i][m_idx_j]
            if r_i % g == r_j % g:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Now solve MIS using branch and bound with bitsets.
    # Use the standard "maximum independent set" algorithm:
    # - Pick a vertex, either include it (remove it and its neighbors) or exclude it.
    # - Upper bound: current_size + number of remaining candidates.
    
    # For 8255 nodes, this might be too slow for exact MIS.
    # Let me use a greedy + local search approach instead, with bitsets.
    
    # Greedy: repeatedly pick the node with minimum degree among candidates.
    # Then local search: try to add nodes, or swap.
    
    # Actually, let me try a smarter greedy with bitsets.
    
    def greedy_mis(candidates, conflicts):
        """Greedy MIS: pick min-degree node, add to solution, remove it and neighbors."""
        solution = set()
        cand = set(candidates)
        while cand:
            # Find node with min degree within cand
            best_node = None
            best_deg = float('inf')
            for node in cand:
                deg = bin(conflicts[node] & 0).count('1')  # placeholder
                # Actually need to count bits in conflicts[node] & cand_mask
                # This is expensive for 8255 nodes. Let me use a different approach.
                pass
            # For speed, use a different strategy: random or fixed order.
            # Let me just pick the first node.
            node = next(iter(cand))
            solution.add(node)
            # Remove node and its neighbors from cand
            to_remove = {node} | (set() if conflicts[node] == 0 else set())
            # This is getting complicated with sets of 8255 elements.
            pass
        return solution
    
    # The bitset approach with Python ints is fast for bit operations.
    # Let me represent candidate set as a bitmask (Python int).
    
    # Full mask
    full_mask = (1 << total_nodes) - 1  # This is a huge int, 8255 bits
    
    # Greedy MIS with bitsets
    def greedy_mis_bitset(order=None):
        if order is None:
            order = list(range(total_nodes))
        solution = []
        cand_mask = full_mask
        for node in order:
            if not (cand_mask & (1 << node)):
                continue
            # Check if node is still a candidate
            solution.append(node)
            # Remove node and its neighbors
            cand_mask &= ~(conflicts[node] | (1 << node))
        return solution
    
    # But this greedy depends heavily on order. Let me try multiple orders.
    
    # Better: use a "minimum remaining values" style greedy.
    # At each step, among candidates, pick the one that removes the fewest candidates.
    
    def greedy_mis_smart():
        solution = []
        cand_mask = full_mask
        # Precompute for each node, the mask of nodes it conflicts with (already have conflicts[node])
        while cand_mask:
            # Find candidate node with minimum "impact" (fewest neighbors in cand)
            best_node = -1
            best_impact = float('inf')
            # Iterate through candidates
            temp = cand_mask
            while temp:
                # Get lowest set bit
                lsb = temp & (-temp)
                node = lsb.bit_length() - 1
                temp ^= lsb
                # Count neighbors in cand
                neighbors_in_cand = conflicts[node] & cand_mask
                impact = bin(neighbors_in_cand).count('1')
                if impact < best_impact:
                    best_impact = impact
                    best_node = node
                    if impact == 0:
                        break  # Can't do better
            if best_node < 0:
                break
            solution.append(best_node)
            cand_mask &= ~(conflicts[best_node] | (1 << best_node))
        return solution
    
    # This is O(n^2) in the worst case per step, and we do up to n steps.
    # With n=8255, this could be slow. Let me time it.
    
    # Alternative: use a simpler but effective approach.
    # Since the problem has structure (gcd-based), let me exploit it.
    
    # Key insight: if two moduli are coprime, ALL their residues conflict.
    # So I can only pick residues from a set of moduli where no two are coprime? No,
    # that's not right either. I pick ONE residue per modulus (or skip a modulus).
    
    # Wait, I can pick multiple residues from the same modulus! The problem says
    # "distinct moduli" - each modulus can appear at most once in the solution.
    
    # Re-reading: "distinct moduli 2..maximum=128" - yes, each modulus used at most once.
    
    # So I pick at most one residue per modulus. This reduces the problem significantly!
    # n_mod = 127, and I pick a subset of moduli and assign one residue to each.
    
    # This is now: max subset of moduli with feasible residue assignment.
    
    # For a fixed subset S of moduli, feasibility: for each pair (m,n) in S,
    # need r_m % gcd(m,n) ≠ r_n % gcd(m,n).
    
    # This is a constraint satisfaction problem. Use backtracking with propagation.
    
    # Branch and bound: order moduli, try to include/exclude each, with feasibility check.
    
    # Let me use a different strategy: since we want max cardinality,
    # and the constraints are pairwise, use a "conflict graph on moduli" approach.
    
    # Two moduli m,n "can coexist" if there exist residues r_m, r_n such that
    # r_m % gcd(m,n) ≠ r_n % gcd(m,n). This is always true if gcd(m,n) > 1
    # (pick r_m=0, r_n=1, then 0%g=0, 1%g=1, different for g>1).
    # If gcd(m,n)=1, they CANNOT coexist.
    
    # So the "coexistence graph" has edge between m,n iff gcd(m,n)>1.
    # I need a clique in this graph? No, I need an independent set in the
    # "conflict graph" where m,n conflict iff gcd(m,n)=1.
    
    # Wait, that's not quite right. Even if gcd(m,n)>1, the residues I pick
    # for m and n might conflict with residues for other moduli.
    
    # But for the purpose of an upper bound: the maximum number of moduli I can pick
    # is bounded by the maximum clique in the "gcd>1" graph, or more precisely,
    # the maximum subset where no two are coprime... no, that's not a standard graph problem.
    
    # Let me just do backtracking on moduli with constraint propagation.
    
    # Order moduli by "constrainedness": moduli with many coprime partners first.
    coprime_count = [0] * n_mod
    for i in range(n_mod):
        for j in range(n_mod):
            if i != j and gcd_mat[i][j] == 1:
                coprime_count[i] += 1
    
    # Sort by descending coprime_count (most constrained first)
    mod_order = sorted(range(n_mod), key=lambda i: -coprime_count[i])
    
    # Backtracking with constraint propagation
    # State: assignment of residues to some moduli, candidate set of remaining moduli.
    
    # For speed, use a greedy approach first, then local search.
    
    # Greedy: process moduli in some order, for each, try all residues, pick one
    # that doesn't conflict with already-assigned moduli.
    
    def greedy_assign(order):
        assignment = {}  # mod_idx -> residue
        for m_idx in order:
            m = moduli[m_idx]
            # Try all residues, pick one that doesn't conflict
            for r in range(m):
                ok = True
                for other_idx, other_r in assignment.items():
                    g = gcd_mat[m_idx][other_idx]
                    if r % g == other_r % g:
                        ok = False
                        break
                if ok:
                    assignment[m_idx] = r
                    break
        return assignment
    
    # Try multiple orderings
    best_assignment = {}
    best_size = 0
    
    orderings = [
        mod_order,
        list(range(n_mod)),
        list(range(n_mod-1, -1, -1)),
        sorted(range(n_mod), key=lambda i: moduli[i]),
        sorted(range(n_mod), key=lambda i: -moduli[i]),
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        assignment = greedy_assign(order)
        if len(assignment) > best_size:
            best_size = len(assignment)
            best_assignment = dict(assignment)
    
    # Local search: try to improve by swapping or adding.
    # For each unassigned modulus, try to add it (with some residue).
    # For each assigned modulus, try changing its residue to free up space for others.
    
    def try_add(assignment, m_idx):
        """Try to add m_idx to assignment with some residue."""
        m = moduli[m_idx]
        for r in range(m):
            ok = True
            for other_idx, other_r in assignment.items():
                g = gcd_mat[m_idx][other_idx]
                if r % g == other_r % g:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def local_search(assignment, max_iters=1000):
        assignment = dict(assignment)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try to add unassigned moduli
            for m_idx in range(n_mod):
                if m_idx in assignment:
                    continue
                r = try_add(assignment, m_idx)
                if r is not None:
                    assignment[m_idx] = r
                    improved = True
            # Try to reassign to enable additions
            # For each pair of assigned moduli, try swapping residues
            assigned_list = list(assignment.keys())
            for i in range(len(assigned_list)):
                for j in range(i+1, len(assigned_list)):
                    m1, m2 = assigned_list[i], assigned_list[j]
                    r1, r2 = assignment[m1], assignment[m2]
                    # Try swapping
                    if r1 == r2:
                        continue
                    # Check if swap is valid
                    ok = True
                    for other_idx, other_r in assignment.items():
                        if other_idx == m1 or other_idx == m2:
                            continue
                        if r2 % gcd_mat[m1][other_idx] == other_r % gcd_mat[m1][other_idx]:
                            ok = False
                            break
                        if r1 % gcd_mat[m2][other_idx] == other_r % gcd_mat[m2][other_idx]:
                            ok = False
                            break
                    if ok:
                        # Check if swap enables adding a new modulus
                        assignment[m1] = r2
                        assignment[m2] = r1
                        added = False
                        for m3 in range(n_mod):
                            if m3 in assignment:
                                continue
                            r3 = try_add(assignment, m3)
                            if r3 is not None:
                                assignment[m3] = r3
                                added = True
                                break
                        if not added:
                            # Revert
                            assignment[m1] = r1
                            assignment[m2] = r2
                        else:
                            improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return assignment
    
    if time.time() < deadline - 10:
        best_assignment = local_search(best_assignment)
    
    # Convert to result
    result = []
    for m_idx, r in best_assignment.items():
        result.append([r, moduli[m_idx]])
    
    # Validate
    def is_valid(result):
        for i in range(len(result)):
            r1, m1 = result[i]
            for j in range(i+1, len(result)):
                r2, m2 = result[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    if not is_valid(result):
        # Fallback: just return first few that are valid
        valid = []
        for pair in result:
            ok = True
            for existing in valid:
                r1, m1 = existing
                r2, m2 = pair
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid.append(pair)
        result = valid
    
    if not result:
        result = [[0, 2]]
    
    return result

