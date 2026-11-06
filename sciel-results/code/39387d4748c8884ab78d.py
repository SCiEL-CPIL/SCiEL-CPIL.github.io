"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def construct():
    n = 256
    
    # Maximum weight 3-AP-free subset using branch-and-bound with local search.
    
    def greedy_3ap_free(n):
        """Greedy: add x if it doesn't complete a 3-AP with any two selected elements."""
        selected = set()
        for x in range(1, n + 1):
            ok = True
            for a in selected:
                for b in selected:
                    if a == b:
                        continue
                    # Check if {a, b, x} contains a 3-term AP
                    if 2*a == b + x or 2*b == a + x or 2*x == a + b:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.add(x)
        return selected
    
    # Start with greedy solution
    selected = greedy_3ap_free(n)
    current_score = sum(1.0 / x for x in selected)
    
    # Local search: try swaps (remove one, add one) and pure additions
    improved = True
    max_iters = 300
    it = 0
    
    while improved and it < max_iters:
        improved = False
        it += 1
        sel = set(selected)
        
        # Try removing each element x, then try to add the best possible y
        for x in sorted(sel):
            trial = sel - {x}
            trial_score = sum(1.0 / y for y in trial)
            
            best_y = None
            best_score = trial_score
            for y in range(1, n + 1):
                if y in trial:
                    continue
                # Check if trial ∪ {y} is 3-AP-free
                ok = True
                for a in trial:
                    for b in trial:
                        if a == b:
                            continue
                        if 2*a == b + y or 2*b == a + y or 2*y == a + b:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    ns = trial_score + 1.0 / y
                    if ns > best_score:
                        best_score = ns
                        best_y = y
            
            if best_y is not None and best_score > current_score + 1e-15:
                selected = list(trial | {best_y})
                current_score = best_score
                improved = True
        
        # Try pure additions
        if not improved:
            sel = set(selected)
            for y in range(1, n + 1):
                if y in sel:
                    continue
                ok = True
                for a in sel:
                    for b in sel:
                        if a == b:
                            continue
                        if 2*a == b + y or 2*b == a + y or 2*y == a + b:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    ns = current_score + 1.0 / y
                    if ns > current_score + 1e-15:
                        selected = list(sel | {y})
                        current_score = ns
                        improved = True
                        break
    
    return sorted(selected)
# EVOLVE-BLOCK-END
