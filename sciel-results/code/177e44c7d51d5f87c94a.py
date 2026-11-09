"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_conflict_graph(triples, max_n):
    adj = [set() for _ in range(max_n + 1)]
    for t in triples:
        a, b, c = t
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj

def _greedy_mis(adj, max_n, seed=0):
    rng = random.Random(seed)
    remaining = set(range(1, max_n + 1))
    selected = set()
    deg = {x: len(adj[x] & remaining) for x in remaining}
    
    while remaining:
        min_deg = min(deg[x] for x in remaining)
        candidates = [x for x in remaining if deg[x] == min_deg]
        x = rng.choice(candidates)
        selected.add(x)
        remaining.discard(x)
        nbrs = set()
        for y in adj[x]:
            if y in remaining:
                nbrs.add(y)
        for y in nbrs:
            remaining.discard(y)
        for z in remaining:
            lost = (1 if x in adj[z] else 0) + sum(1 for y in nbrs if y in adj[z])
            deg[z] -= lost
    return selected

def _repair_and_add(sel, triples, max_n):
    """Try to add any element that doesn't create a bad triple."""
    # For each element not in sel, check if it can be added
    sel_set = set(sel)
    added = True
    while added:
        added = False
        for x in range(1, max_n + 1):
            if x in sel_set:
                continue
            # Check if adding x creates any bad triple
            ok = True
            for t in triples:
                if x in t:
                    other1, other2 = [v for v in t if v != x]
                    if other1 in sel_set and other2 in sel_set:
                        ok = False
                        break
            if ok:
                sel_set.add(x)
                added = True
    return sel_set

def _local_search(selected, triples, adj, max_n, rng, time_limit):
    start = time.time()
    sel = set(selected)
    best_sel = set(sel)
    best_size = len(sel)
    
    # For each element, precompute the pairs that would form a bad triple with it
    x_pairs = {i: [] for i in range(1, max_n + 1)}
    for t in triples:
        a, b, c = t
        x_pairs[a].append((b, c))
        x_pairs[b].append((a, c))
        x_pairs[c].append((a, b))
    
    def is_addable(x, S):
        for (a, b) in x_pairs[x]:
            if a in S and b in S:
                return False
        return True
    
    # Phase 1: Greedy add
    excluded = set(range(1, max_n + 1)) - sel
    improved = True
    while improved:
        improved = False
        for x in sorted(excluded, key=lambda y: len(x_pairs[y])):
            if is_addable(x, sel):
                sel.add(x)
                excluded.discard(x)
                improved = True
                if len(sel) > best_size:
                    best_size = len(sel)
                    best_sel = set(sel)
        if time.time() - start > time_limit:
            break
    
    # Phase 2: Random swaps
    T = 3.0
    cooling = 0.999
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 50000:
        step += 1
        x = rng.choice(list(sel))
        y = rng.choice(list(excluded))
        sel_no_x = sel - {x}
        if is_addable(y, sel_no_x):
            sel = sel_no_x | {y}
            excluded.discard(y)
            excluded.add(x)
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
        T *= cooling
    
    return best_sel

def _sa_polish(selected, triples, max_n, rng, time_limit):
    start = time.time()
    all_elems = set(range(1, max_n + 1))
    sel = set(selected)
    
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for ti, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(ti)
    
    triple_count = [0] * len(triples)
    for ti, t in enumerate(triples):
        for x in t:
            if x in sel:
                triple_count[ti] += 1
    current_conflicts = sum(1 for c in triple_count if c == 3)
    
    best_sel = set(sel)
    best_conflicts = current_conflicts
    best_size = len(sel)
    
    T = 5.0
    cooling = 0.9995
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 150000:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.35:
            cands = all_elems - sel
            if not cands:
                continue
            x = rng.choice(list(cands))
            affected = elem_to_triples[x]
            delta_conf = 0
            for ti in affected:
                if triple_count[ti] == 2:
                    delta_conf += 1
                triple_count[ti] += 1
            new_conf = current_conflicts + delta_conf
            delta = delta_conf * 100 - 1
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                sel.add(x)
                current_conflicts = new_conf
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(sel) > best_size):
                    best_conflicts = current_conflicts
                    best_size = len(sel)
                    best_sel = set(sel)
            else:
                for ti in affected:
                    triple_count[ti] -= 1
        elif move_type < 0.6:
            if not sel:
                continue
            x = rng.choice(list(sel))
            affected = elem_to_triples[x]
            delta_conf = 0
            for ti in affected:
                if triple_count[ti] == 3:
                    delta_conf -= 1
                triple_count[ti] -= 1
            new_conf = current_conflicts + delta_conf
            delta = delta_conf * 100 + 1
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                sel.discard(x)
                current_conflicts = new_conf
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(sel) > best_size):
                    best_conflicts = current_conflicts
                    best_size = len(sel)
                    best_sel = set(sel)
            else:
                for ti in affected:
                    triple_count[ti] += 1
        else:
            if not sel:
                continue
            x = rng.choice(list(sel))
            cands = all_elems - sel
            if not cands:
                continue
            y = rng.choice(list(cands))
            aff_x = elem_to_triples[x]
            aff_y = elem_to_triples[y]
            all_aff = list(set(aff_x + aff_y))
            old = {ti: triple_count[ti] for ti in all_aff}
            for ti in aff_x:
                triple_count[ti] -= 1
            for ti in aff_y:
                triple_count[ti] += 1
            delta_conf = 0
            for ti in all_aff:
                if old[ti] == 3 and triple_count[ti] != 3:
                    delta_conf -= 1
                elif old[ti] != 3 and triple_count[ti] == 3:
                    delta_conf += 1
            new_conf = current_conflicts + delta_conf
            delta = delta_conf * 100
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                sel.discard(x)
                sel.add(y)
                current_conflicts = new_conf
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(sel) > best_size):
                    best_conflicts = current_conflicts
                    best_size = len(sel)
                    best_sel = set(sel)
            else:
                for ti in all_aff:
                    triple_count[ti] = old[ti]
        T *= cooling
    
    if best_conflicts > 0:
        fixed = set(best_sel)
        for _ in range(500):
            act = [t for t in triples if t[0] in fixed and t[1] in fixed and t[2] in fixed]
            if not act:
                break
            freq = {}
            for t in act:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            mf = max(freq.values())
            cands = [x for x, f in freq.items() if f == mf]
            fixed.discard(rng.choice(cands))
        if len(fixed) >= best_size:
            best_sel = fixed
            best_size = len(fixed)
    
    return best_sel

def construct():
    start_time = time.time()
    deadline = start_time + 82
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    adj = _build_conflict_graph(bad_triples, max_n)
    
    best_selected = None
    best_size = -1
    
    # Phase 1: Greedy MIS with many restarts
    num_greedy = 400
    for i in range(num_greedy):
        if time.time() > deadline - 25:
            break
        selected = _greedy_mis(adj, max_n, seed=i)
        # Repair: try to add back elements
        selected = _repair_and_add(selected, bad_triples, max_n)
        size = len(selected)
        if size > best_size:
            best_size = size
            best_selected = selected
    
    if best_selected is None:
        return [1]
    
    # Phase 2: Local search
    if time.time() < deadline - 12:
        remaining_time = deadline - time.time() - 12
        num_ls = max(1, int(remaining_time / 3.0))
        for i in range(num_ls):
            if time.time() > deadline - 10:
                break
            rng = random.Random(5000 + i)
            polished = _local_search(best_selected, bad_triples, adj, max_n, rng, 2.5)
            size = len(polished)
            if size > best_size:
                best_size = size
                best_selected = polished
    
    # Phase 3: SA polish
    if time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 3
        num_sa = max(1, int(remaining_time / 5.0))
        for i in range(num_sa):
            if time.time() > deadline - 2:
                break
            rng = random.Random(10000 + i)
            polished = _sa_polish(best_selected, bad_triples, max_n, rng, 4.5)
            size = len(polished)
            if size > best_size:
                best_size = size
                best_selected = polished
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END
