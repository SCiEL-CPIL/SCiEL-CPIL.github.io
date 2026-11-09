"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math
from math import gcd as _gcd
import sys
sys.setrecursionlimit(100000)

def _build_graph():
    # Build coprime graph on nodes 0..510 (representing values 1..511)
    n = 511
    # coprime_mask[i] = bitmask of nodes j != i such that gcd(i+1, j+1) == 1
    coprime_mask = [0] * n
    for i in range(n):
        vi = i + 1
        mask = 0
        for j in range(n):
            if j == i:
                continue
            if _gcd(vi, j + 1) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask
    return coprime_mask

def _clique_cover(coprime_mask, nodes, n):
    # Greedy: repeatedly pick a node with max degree within current nodes,
    # form a "clique" = {v} union {u in nodes : u coprime to v} (all pairwise
    # coprime since each is coprime to v and we'll verify pairwise).
    # Actually we need true cliques. Let's build cliques properly:
    # A clique in coprime graph = set of pairwise coprime nodes.
    # Greedy clique construction:
    cliques = []
    remaining = set(nodes)
    while remaining:
        # Pick node with max degree in remaining
        best = None
        best_deg = -1
        for v in remaining:
            deg = bin(coprime_mask[v] & _bitmask(remaining)).count('1')
            if deg > best_deg:
                best_deg = deg
                best = v
        # Build clique greedily
        clique = [best]
        cand = remaining - {best}
        for u in sorted(cand):
            # check u coprime to all in clique
            ok = True
            for c in clique:
                if not (coprime_mask[u] >> c) & 1:
                    ok = False
                    break
            if ok:
                clique.append(u)
        cliques.append(clique)
        remaining -= set(clique)
    return cliques

def _bitmask(nodes):
    m = 0
    for x in nodes:
        m |= (1 << x)
    return m

def _solve_mwis(coprime_mask, weights, n, deadline):
    # Maximum weight independent set via branch and bound
    # Upper bound: sum over cliques in a clique cover of max weight in each clique
    # (since independent set can pick at most 1 from each clique)
    best_weight = [0.0]
    best_set = [set()]
    
    # Precompute: for each node, its neighbors mask
    # Initial greedy solution for lower bound
    def greedy_lower_bound():
        remaining = set(range(n))
        chosen = set()
        total = 0.0
        # Sort by weight descending
        order = sorted(range(n), key=lambda x: -weights[x])
        for v in order:
            if v not in remaining:
                continue
            # Check coprime to all chosen
            ok = True
            for c in chosen:
                if not (coprime_mask[v] >> c) & 1:
                    ok = False
                    break
            if ok:
                chosen.add(v)
                total += weights[v]
        return total, chosen
    
    glb, gset = greedy_lower_bound()
    best_weight[0] = glb
    best_set[0] = set(gset)
    
    # Compute initial clique cover for bounding
    all_nodes = set(range(n))
    cliques = _clique_cover(coprime_mask, all_nodes, n)
    
    # For each clique, precompute member list
    clique_members = [set(c) for c in cliques]
    
    def upper_bound(available_mask):
        # For each clique, max weight among available nodes in that clique
        ub = 0.0
        for cm in clique_members:
            mx = 0.0
            for v in cm:
                if (available_mask >> v) & 1:
                    if weights[v] > mx:
                        mx = weights[v]
            ub += mx
        return ub
    
    # Branch and bound
    # State: (available_mask, current_weight, current_set)
    # We branch on a node: include it or exclude it
    # To keep it fast, use recursion with memoization-free approach
    
    # Actually, for 511 nodes this is too large for exact B&B.
    # Let's use a smarter approach: local search + limited B&B on reduced problem.
    
    # Strategy: 
    # 1. Get good initial solution via multiple greedy + local search
    # 2. Use B&B on a reduced neighborhood to try to improve
    
    # Multiple greedy starts
    best_w = glb
    best_s = set(gset)
    
    # Greedy with different orderings
    import random
    rng = random.Random(42)
    
    # Greedy by weight desc
    for attempt in range(20):
        if _time.time() > deadline:
            break
        if attempt == 0:
            order = sorted(range(n), key=lambda x: -weights[x])
        elif attempt == 1:
            order = sorted(range(n), key=lambda x: x)  # ascending value
        elif attempt == 2:
            order = list(range(n))
            rng.shuffle(order)
        else:
            # Weight with random tiebreak
            order = sorted(range(n), key=lambda x: (-weights[x] + rng.random()*0.001, x))
        
        chosen = set()
        total = 0.0
        for v in order:
            ok = True
            for c in chosen:
                if not (coprime_mask[v] >> c) & 1:
                    ok = False
                    break
            if ok:
                chosen.add(v)
                total += weights[v]
        if total > best_w:
            best_w = total
            best_s = set(chosen)
    
    # Local search: try to swap or add
    def local_search(start_set):
        cur_set = set(start_set)
        cur_w = sum(weights[v] for v in cur_set)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # Try adding each unchosen node
            for v in range(n):
                if v in cur_set:
                    continue
                ok = True
                for c in cur_set:
                    if not (coprime_mask[v] >> c) & 1:
                        ok = False
                        break
                if ok:
                    cur_set.add(v)
                    cur_w += weights[v]
                    improved = True
                    break
            # Try swapping: remove one, add two?
            # Simple: try removing one node and see if we can add better ones
            if not improved:
                for rem in list(cur_set):
                    new_set = cur_set - {rem}
                    new_w = cur_w - weights[rem]
                    # Try to add nodes
                    added = False
                    for v in range(n):
                        if v in new_set:
                            continue
                        ok = True
                        for c in new_set:
                            if not (coprime_mask[v] >> c) & 1:
                                ok = False
                                break
                        if ok:
                            new_set.add(v)
                            new_w += weights[v]
                            added = True
                    if added and new_w > cur_w:
                        cur_set = new_set
                        cur_w = new_w
                        improved = True
                        break
        return cur_w, cur_set
    
    lw, ls = local_search(best_s)
    if lw > best_w:
        best_w = lw
        best_s = ls
    
    # Try B&B on a subset: focus on high-weight nodes
    # Select top K nodes by weight, try exact MWIS on them
    top_k = 80
    top_nodes = sorted(range(n), key=lambda x: -weights[x])[:top_k]
    top_set = set(top_nodes)
    
    # Build subgraph
    # Map top nodes to 0..K-1
    node_to_idx = {v: i for i, v in enumerate(top_nodes)}
    K = len(top_nodes)
    sub_weights = [weights[v] for v in top_nodes]
    
    # Sub coprime masks
    sub_coprime = [0] * K
    for i, vi in enumerate(top_nodes):
        for j, vj in enumerate(top_nodes):
            if i != j and (coprime_mask[vi] >> vj) & 1:
                sub_coprime[i] |= (1 << j)
    
    # Exact MWIS on K nodes via B&B
    def exact_mwis(sub_coprime, sub_weights, K, deadline):
        best_w = [0.0]
        best_mask = [0]
        
        # Greedy lower bound
        order = sorted(range(K), key=lambda x: -sub_weights[x])
        chosen_mask = 0
        cur_w = 0.0
        for v in order:
            # Check coprime to all in chosen
            ok = True
            cm = chosen_mask
            while cm:
                lsb = cm & (-cm)
                c = lsb.bit_length() - 1
                if not (sub_coprime[v] >> c) & 1:
                    ok = False
                    break
                cm ^= lsb
            if ok:
                chosen_mask |= (1 << v)
                cur_w += sub_weights[v]
        best_w[0] = cur_w
        best_mask[0] = chosen_mask
        
        # Upper bound: clique cover
        all_nodes_sub = set(range(K))
        def clique_cover_sub(nodes):
            cliques = []
            remaining = set(nodes)
            while remaining:
                # Pick node with max degree
                best = None
                best_deg = -1
                for v in remaining:
                    deg = bin(sub_coprime[v] & _bitmask(remaining)).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best = v
                clique = [best]
                cand = sorted(remaining - {best})
                for u in cand:
                    ok = all((sub_coprime[u] >> c) & 1 for c in clique)
                    if ok:
                        clique.append(u)
                cliques.append(clique)
                remaining -= set(clique)
            return cliques
        
        sub_cliques = clique_cover_sub(all_nodes_sub)
        sub_clique_sets = [set(c) for c in sub_cliques]
        
        def ub(available_mask):
            ubv = 0.0
            for cs in sub_clique_sets:
                mx = 0.0
                for v in cs:
                    if (available_mask >> v) & 1:
                        if sub_weights[v] > mx:
                            mx = sub_weights[v]
                ubv += mx
            return ubv
        
        # B&B
        def bb(available_mask, cur_w, chosen_mask):
            if _time.time() > deadline:
                return
            if available_mask == 0:
                if cur_w > best_w[0]:
                    best_w[0] = cur_w
                    best_mask[0] = chosen_mask
                return
            # Upper bound
            if cur_w + ub(available_mask) <= best_w[0]:
                return
            # Pick a node to branch on: highest weight available
            am = available_mask
            best_v = -1
            best_wv = -1.0
            while am:
                lsb = am & (-am)
                v = lsb.bit_length() - 1
                if sub_weights[v] > best_wv:
                    best_wv = sub_weights[v]
                    best_v = v
                am ^= lsb
            if best_v < 0:
                return
            # Branch 1: include best_v
            new_avail = available_mask & ~(sub_coprime[best_v] | (1 << best_v))
            bb(new_avail, cur_w + sub_weights[best_v], chosen_mask | (1 << best_v))
            # Branch 2: exclude best_v
            bb(available_mask & ~(1 << best_v), cur_w, chosen_mask)
        
        full_mask = (1 << K) - 1
        bb(full_mask, 0.0, 0)
        return best_w[0], best_mask[0]
    
    bw, bm = exact_mwis(sub_coprime, sub_weights, K, deadline)
    # Convert back
    sub_solution = set()
    for i, v in enumerate(top_nodes):
        if (bm >> i) & 1:
            sub_solution.add(v)
    
    # Combine: try to add nodes outside top_k to sub_solution
    combined = set(sub_solution)
    for v in range(n):
        if v in combined:
            continue
        ok = all((coprime_mask[v] >> c) & 1 for c in combined)
        if ok:
            combined.add(v)
    
    cw = sum(weights[v] for v in combined)
    if cw > best_w:
        best_w = cw
        best_s = combined
    
    # Final local search
    lw2, ls2 = local_search(best_s)
    if lw2 > best_w:
        best_w = lw2
        best_s = ls2
    
    return [v + 1 for v in sorted(best_s)]

def construct():
    start_time = _time.time()
    deadline = start_time + 140.0  # safety margin
    
    coprime_mask = _build_graph()
    n = 511
    weights = [1.0 / (512 - (i + 1)) for i in range(n)]
    
    result = _solve_mwis(coprime_mask, weights, n, deadline)
    return result
# EVOLVE-BLOCK-END
