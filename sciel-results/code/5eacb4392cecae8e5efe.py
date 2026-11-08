import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Rank = 105 - (number of disjoint 2x2x2 Strassen blocks).
    Uses branch-and-bound with maximal clique lower bounds to find maximum
    packing of disjoint Strassen blocks, combined with greedy initialization
    and local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total = n * m * p  # 105

    # Precompute all blocks: each = (params, frozenset of (i,j,k), bitmask)
    blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            triples = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = (i * m + j) * p + k
                                        mask |= (1 << idx)
                                        triples.append((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(triples), mask))

    B = len(blocks)
    
    # Build conflict graph: two blocks conflict if they share any triple
    # conflict[a] is a bitmask of all blocks conflicting with block a
    conflict = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        bits = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]):
                bits |= (1 << b)
        conflict[a] = bits

    # Independent set = set of blocks with no conflicts among them
    # We want maximum independent set in the conflict graph

    def greedy_is(nodes, order=None):
        """Greedy independent set: pick nodes in order if they don't conflict with already picked."""
        if order is None:
            order = list(range(len(nodes)))
        picked = []
        picked_set = set()
        for node in order:
            if node in picked_set:
                continue
            # Check if node conflicts with any already picked
            ok = True
            for p in picked:
                if (conflict[node] >> p) & 1:
                    ok = False
                    break
            if ok:
                picked.append(node)
                picked_set.add(node)
        return picked

    def local_search_is(packing, max_iters=50):
        """Local search: try to replace subsets of packing with larger subsets."""
        used_mask = 0
        for b in packing:
            used_mask |= blocks[b][2]
        
        cur_len = len(packing)
        best_packing = list(packing)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            
            improved = False
            
            # Try single removal + multiple additions
            for c in packing:
                used2 = used_mask & ~blocks[c][2]
                # Find all blocks that fit after removing c
                candidates = [a for a in range(B) if a not in packing and (used2 & blocks[a][2]) == 0]
                
                if len(candidates) >= 1:
                    # Greedily add as many as possible
                    new_packing = [x for x in packing if x != c]
                    new_used = used2
                    added = 0
                    for a in sorted(candidates, key=lambda x: random.random()):
                        if (new_used & blocks[a][2]) == 0:
                            new_packing.append(a)
                            new_used |= blocks[a][2]
                            added += 1
                    
                    if len(new_packing) > cur_len:
                        packing = new_packing
                        used_mask = new_used
                        cur_len = len(packing)
                        best_packing = list(packing)
                        improved = True
                        break
            
            if not improved:
                # Try double removal
                for i in range(len(packing)):
                    for j in range(i+1, len(packing)):
                        c1, c2 = packing[i], packing[j]
                        used2 = used_mask & ~blocks[c1][2] & ~blocks[c2][2]
                        candidates = [a for a in range(B) if a not in packing and (used2 & blocks[a][2]) == 0]
                        
                        if len(candidates) >= 2:
                            new_packing = [x for x in packing if x != c1 and x != c2]
                            new_used = used2
                            for a in sorted(candidates, key=lambda x: random.random()):
                                if (new_used & blocks[a][2]) == 0:
                                    new_packing.append(a)
                                    new_used |= blocks[a][2]
                            
                            if len(new_packing) > cur_len:
                                packing = new_packing
                                used_mask = new_used
                                cur_len = len(packing)
                                best_packing = list(packing)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_packing

    def branch_and_bound(nodes, current_set, best_set, depth=0):
        """
        Branch-and-bound for maximum independent set.
        nodes: list of candidate nodes to consider
        current_set: set of nodes already in the independent set
        best_set: best solution found so far
        """
        if time.time() > deadline - 20.0:
            return best_set
        
        if len(current_set) + len(nodes) <= len(best_set):
            return best_set
        
        if not nodes:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            return best_set
        
        # Lower bound: greedy on remaining nodes
        remaining = list(nodes)
        greedy_result = greedy_is(remaining)
        if len(current_set) + len(greedy_result) <= len(best_set):
            return best_set
        
        # Pick a node to branch on
        # Heuristic: pick node with highest degree in remaining subgraph
        best_node = 0
        best_degree = -1
        for node in nodes:
            deg = 0
            for other in nodes:
                if other != node and ((conflict[node] >> other) & 1):
                    deg += 1
            if deg > best_degree:
                best_degree = deg
                best_node = node
        
        # Branch 1: include best_node
        new_current = set(current_set)
        new_current.add(best_node)
        new_nodes = [n for n in nodes if n != best_node and not ((conflict[best_node] >> n) & 1)]
        best_set = branch_and_bound(new_nodes, new_current, best_set, depth + 1)
        
        # Branch 2: exclude best_node
        new_nodes2 = [n for n in nodes if n != best_node]
        best_set = branch_and_bound(new_nodes2, current_set, best_set, depth + 1)
        
        return best_set

    best_packing = []
    best_len = 0
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy initializations with local search
    for trial in range(3000):
        if time.time() > deadline - 25.0:
            break
        
        order = list(range(B))
        rng.shuffle(order)
        cand = greedy_is(order)
        
        if len(cand) > best_len:
            best_packing = cand
            best_len = len(cand)
        
        # Local search refinement
        if len(cand) >= best_len - 1:
            cand2 = local_search_is(list(cand))
            if len(cand2) > best_len:
                best_packing = cand2
                best_len = len(cand2)
        
        if best_len >= 13:
            break
    
    # Phase 2: Branch-and-bound from best packing
    if best_len < 13 and time.time() < deadline - 20.0:
        # Use the current best as a lower bound
        best_bb = list(best_packing)
        
        # Try branch-and-bound on subsets of blocks
        # Start with blocks that don't conflict with any in best_packing
        compatible = [b for b in range(B) if b not in best_packing and 
                      not any((conflict[b] >> c) & 1 for c in best_packing)]
        
        if compatible:
            # Run B&B on compatible blocks, adding to best_packing
            # This is a simplified approach: just try to extend
            for trial in range(200):
                if time.time() > deadline - 15.0:
                    break
                
                # Random subset of compatible blocks
                subset_size = min(len(compatible), rng.randint(10, 25))
                subset = rng.sample(compatible, subset_size)
                
                result = branch_and_bound(subset, set(best_packing), best_bb)
                if len(result) > best_len:
                    best_packing = result
                    best_len = len(result)
                
                if best_len >= 13:
                    break
    
    # Phase 3: Further refinement with more aggressive local search
    if best_len < 13 and time.time() < deadline - 10.0:
        for trial in range(1000):
            if time.time() > deadline - 8.0:
                break
            
            # Perturb and re-optimize
            perturbed = list(best_packing)
            num_remove = min(2, len(perturbed))
            for _ in range(num_remove):
                if perturbed:
                    idx = rng.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
            
            # Greedy fill
            used = 0
            for b in perturbed:
                used |= blocks[b][2]
            
            order = list(range(B))
            rng.shuffle(order)
            for b in order:
                if (used & blocks[b][2]) == 0:
                    perturbed.append(b)
                    used |= blocks[b][2]
            
            cand2 = local_search_is(perturbed)
            if len(cand2) > best_len:
                best_packing = cand2
                best_len = len(cand2)
                
                if best_len >= 13:
                    break
    
    # Build the matrices from the best packing
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bidx in best_packing:
        i1, i2, j1, j2, k1, k2 = blocks[bidx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for t in blocks[bidx][1]:
            covered.add(t)

    # Schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
