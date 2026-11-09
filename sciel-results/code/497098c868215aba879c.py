"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    """
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
    """
    Build conflict graph: two elements conflict if they appear together in some bad triple.
    Also build element -> list of triple indices.
    """
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for ti, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(ti)
    
    # Conflict graph: adj[x] = set of elements that conflict with x
    adj = {i: set() for i in range(1, max_n + 1)}
    for t in triples:
        a, b, c = t
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    
    return adj, elem_to_triples

def _count_conflicts_fast(selected, triples, elem_to_triples):
    """
    Count bad triples fully contained in selected using the element-to-triples index.
    """
    count = 0
    # Use a set for O(1) lookup
    sel_set = selected
    for t in triples:
        if t[0] in sel_set and t[1] in sel_set and t[2] in sel_set:
            count += 1
    return count

def _greedy_max_independent_set(adj, max_n, seed=0):
    """
    Greedy: repeatedly pick the element with minimum degree (conflicts),
    add to independent set, remove it and its neighbors.
    """
    rng = random.Random(seed)
    remaining = set(range(1, max_n + 1))
    selected = set()
    
    # Work with a mutable degree count
    deg = {x: len(adj[x] & remaining) for x in remaining}
    
    while remaining:
        # Pick element with min degree, break ties randomly
        min_deg = min(deg[x] for x in remaining)
        candidates = [x for x in remaining if deg[x] == min_deg]
        x = rng.choice(candidates)
        
        selected.add(x)
        remaining.discard(x)
        
        # Remove neighbors from remaining
        for y in list(adj[x]):
            if y in remaining:
                remaining.discard(y)
                for z in remaining:
                    if y in adj[z]:
                        deg[z] = deg.get(z, 0) - 1
                # Also update deg for remaining elements that had x as neighbor
                # Actually we already removed x, so just remove y's contribution
                for z in remaining:
                    if y in adj[z]:
                        deg[z] -= 1
        
        # Update degrees for remaining elements (they lost x)
        for z in remaining:
            if x in adj[z]:
                deg[z] -= 1
    
    return selected

def _local_search(selected, triples, adj, max_n, rng, time_limit):
    """
    Local search: try to add elements back that were excluded, and swap.
    State: selected set (valid independent set).
    Moves:
      1. Try adding an element not in selected (if it doesn't create a bad triple)
      2. Swap: remove x from selected, add y not in selected
    """
    start = time.time()
    sel = set(selected)
    best_sel = set(sel)
    best_size = len(sel)
    
    # Precompute: for each element, which triples does it participate in?
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for ti, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(ti)
    
    # For quick conflict check: given a set S and element x, is S ∪ {x} valid?
    # x creates a conflict if there exist a, b in S such that (a,b,x) is a bad triple.
    # Precompute for each x, the pairs (a,b) in bad triples containing x.
    x_conflict_pairs = {}
    for t in triples:
        a, b, c = t
        x_conflict_pairs.setdefault(a, []).append((b, c))
        x_conflict_pairs.setdefault(b, []).append((a, c))
        x_conflict_pairs.setdefault(c, []).append((a, b))
    
    def is_addable(x, S):
        """Check if adding x to S keeps it valid."""
        for (a, b) in x_conflict_pairs.get(x, []):
            if a in S and b in S:
                return False
        return True
    
    def conflicts_after_add(x, S):
        """Count how many bad triples would be created by adding x to S."""
        cnt = 0
        for (a, b) in x_conflict_pairs.get(x, []):
            if a in S and b in S:
                cnt += 1
        return cnt
    
    # Phase 1: Try to add elements
    excluded = set(range(1, max_n + 1)) - sel
    improved = True
    while improved:
        improved = False
        for x in sorted(excluded, key=lambda y: len(x_conflict_pairs.get(y, []))):
            if is_addable(x, sel):
                sel.add(x)
                excluded.discard(x)
                improved = True
                if len(sel) > best_size:
                    best_size = len(sel)
                    best_sel = set(sel)
        if time.time() - start > time_limit:
            break
    
    # Phase 2: Swap-based local search
    T = 3.0
    cooling = 0.999
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 100000:
        step += 1
        x = rng.choice(list(sel))
        y = rng.choice(list(excluded))
        
        # Check if swap is valid
        sel_test = set(sel)
        sel_test.discard(x)
        sel_test.add(y)
        
        # Count conflicts in sel_test
        new_conflicts = 0
        for t in triples:
            if t[0] in sel_test and t[1] in sel_test and t[2] in sel_test:
                new_conflicts += 1
        
        if new_conflicts == 0:
            # Valid swap
            sel = sel_test
            excluded = set(range(1, max_n + 1)) - sel
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
        
        T *= cooling
    
    return best_sel

def _simulated_annealing_polish(selected, triples, adj, max_n, rng, time_limit):
    """
    SA polish: work with the complement (removed set) to minimize |removed|.
    """
    start = time.time()
    all_elems = set(range(1, max_n + 1))
    removed = all_elems - set(selected)
    sel = set(selected)
    
    # Precompute conflict pairs
    x_conflict_pairs = {}
    for t in triples:
        a, b, c = t
        x_conflict_pairs.setdefault(a, []).append((b, c))
        x_conflict_pairs.setdefault(b, []).append((a, c))
        x_conflict_pairs.setdefault(c, []).append((a, b))
    
    def count_conflicts(S):
        cnt = 0
        for t in triples:
            if t[0] in S and t[1] in S and t[2] in S:
                cnt += 1
        return cnt
    
    current_conflicts = count_conflicts(sel)
    current_cost = len(removed) + 1000 * current_conflicts
    
    best_removed = set(removed)
    best_cost = current_cost
    best_conflicts = current_conflicts
    
    T = 5.0
    cooling = 0.9995
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 200000:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.4 and removed:
            # Remove from removed (add to selected)
            x = rng.choice(list(removed))
            new_removed = set(removed)
            new_removed.discard(x)
        elif move_type < 0.7 and len(removed) < max_n:
            # Add to removed
            candidates = list(all_elems - removed)
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_removed = set(removed)
            new_removed.add(x)
        else:
            # Swap
            if removed and len(removed) < max_n:
                x_out = rng.choice(list(removed))
                candidates_add = list(all_elems - removed)
                if not candidates_add:
                    continue
                x_in = rng.choice(candidates_add)
                new_removed = set(removed)
                new_removed.discard(x_out)
                new_removed.add(x_in)
            else:
                if removed:
                    x = rng.choice(list(removed))
                    new_removed = set(removed)
                    new_removed.discard(x)
                else:
                    x = rng.choice(list(all_elems))
                    new_removed = set(removed)
                    new_removed.add(x)
        
        new_sel = all_elems - new_removed
        new_conflicts = count_conflicts(new_sel)
        new_cost = len(new_removed) + 1000 * new_conflicts
        
        delta = new_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            removed = new_removed
            sel = new_sel
            current_conflicts = new_conflicts
            current_cost = new_cost
            
            if new_conflicts == 0 and len(new_removed) < len(best_removed):
                best_removed = set(new_removed)
                best_cost = len(new_removed)
                best_conflicts = 0
        
        T *= cooling
    
    # Fix if best has conflicts
    if best_conflicts > 0:
        fixed_removed = set(best_removed)
        fixed_sel = all_elems - fixed_removed
        active = [t for t in triples if t[0] in fixed_sel and t[1] in fixed_sel and t[2] in fixed_sel]
        while active:
            freq = {}
            for t in active:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            max_f = max(freq.values())
            cands = [x for x, f in freq.items() if f == max_f]
            x = rng.choice(cands)
            fixed_removed.add(x)
            fixed_sel.discard(x)
            active = [t for t in active if x not in t]
        best_removed = fixed_removed
    
    return best_removed

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit (90s kill, leave margin)
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    adj, elem_to_triples = _build_conflict_graph(bad_triples, max_n)
    
    best_selected = None
    best_size = -1
    
    # Phase 1: Multiple greedy max independent set attempts
    num_greedy = 200
    for i in range(num_greedy):
        if time.time() > deadline - 15:
            break
        selected = _greedy_max_independent_set(adj, max_n, seed=i)
        size = len(selected)
        if size > best_size:
            best_size = size
            best_selected = selected
    
    if best_selected is None:
        # Fallback: single element
        return [1]
    
    # Phase 2: Local search polish on best greedy result
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        num_ls = max(1, int(remaining_time / 5))
        
        for i in range(num_ls):
            if time.time() > deadline - 3:
                break
            rng = random.Random(5000 + i)
            polished = _local_search(best_selected, bad_triples, adj, max_n, rng, 4.0)
            size = len(polished)
            if size > best_size:
                best_size = size
                best_selected = polished
    
    # Phase 3: SA polish
    if time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 2
        num_sa = max(1, int(remaining_time / 8))
        
        for i in range(num_sa):
            if time.time() > deadline - 2:
                break
            rng = random.Random(10000 + i)
            removed = _simulated_annealing_polish(best_selected, bad_triples, adj, max_n, rng, 7.0)
            selected = set(range(1, max_n + 1)) - removed
            size = len(selected)
            if size > best_size:
                best_size = size
                best_selected = selected
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END
