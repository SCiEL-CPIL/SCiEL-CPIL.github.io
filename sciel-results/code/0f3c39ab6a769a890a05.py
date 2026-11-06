"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import lcm

def construct():
    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute lcm conflicts
    conflict = {a: set() for a in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            if lcm(a, b) <= n:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Greedy: smallest numbers first (highest reciprocal)
    selected = set()
    for a in sorted(candidates):
        if not (conflict[a] & selected):
            selected.add(a)
    
    inv = {a: 1.0 / a for a in candidates}
    
    def score(s):
        return sum(inv[x] for x in s)
    
    current_score = score(selected)
    improved = True
    max_iter = 50
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for v in candidates:
            if v in selected:
                continue
            conf_in_sel = conflict[v] & selected
            if not conf_in_sel:
                new_score = current_score + inv[v]
                if new_score > current_score + 1e-15:
                    selected.add(v)
                    current_score = new_score
                    improved = True
            else:
                removed_sum = sum(inv[x] for x in conf_in_sel)
                if inv[v] > removed_sum + 1e-15:
                    new_score = current_score - removed_sum + inv[v]
                    if new_score > current_score + 1e-15:
                        selected -= conf_in_sel
                        selected.add(v)
                        current_score = new_score
                        improved = True
    
    return list(selected)
# EVOLVE-BLOCK-END
