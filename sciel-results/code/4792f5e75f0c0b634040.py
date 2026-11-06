"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    n = 256
    # Perfect squares up to 512
    squares = set()
    for k in range(1, 23):
        squares.add(k * k)
    
    # Build conflict: conflict[a] = set of b where a+b is a perfect square
    conflict = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in squares:
            b = s - a
            if 1 <= b <= n:
                conflict[a].add(b)
    
    # Greedy initial: sort by degree ascending, add if no conflict with already selected
    degrees = [len(conflict[i]) for i in range(1, n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: degrees[x - 1])
    
    selected = set()
    for v in order:
        # Check conflicts
        ok = True
        for u in conflict[v]:
            if u in selected:
                ok = False
                break
        if ok:
            selected.add(v)
    
    best = selected.copy()
    best_size = len(selected)
    
    # Fast local search using bitmasks or efficient sets
    # Try: for each unselected v, compute how many selected it conflicts with
    # If 0, add it. If >0, try removing those and adding v if net gain.
    
    random.seed(123)
    
    for outer in range(200):
        if time.time() > deadline:
            break
        
        unselected = [v for v in range(1, n + 1) if v not in selected]
        if not unselected:
            break
        
        # Phase 1: add any vertex with 0 conflicts
        changed = True
        while changed:
            changed = False
            for v in unselected:
                has_conflict = False
                for u in conflict[v]:
                    if u in selected:
                        has_conflict = True
                        break
                if not has_conflict:
                    selected.add(v)
                    unselected.remove(v)
                    changed = True
                    if len(selected) > best_size:
                        best = selected.copy()
                        best_size = len(selected)
        
        if time.time() > deadline:
            break
        
        # Phase 2: try swaps - remove a set of k vertices, add 1
        # For each unselected v, let C(v) = conflicts in selected
        # If we remove C(v) and add v, net change = 1 - len(C(v))
        # Only beneficial if len(C(v)) == 0 (handled above) or we can find other gains
        
        # Better: try removing 1 vertex and adding 1-2 others
        for trial in range(50):
            if time.time() > deadline:
                break
            
            if not selected:
                break
            
            v_remove = random.choice(list(selected))
            temp_selected = selected - {v_remove}
            
            # Find vertices that can now be added
            candidates = []
            for v in unselected:
                ok = True
                for u in conflict[v]:
                    if u in temp_selected:
                        ok = False
                        break
                if ok:
                    candidates.append(v)
            
            if candidates:
                # Add all candidates that don't conflict with each other
                added = []
                for v in candidates:
                    ok = True
                    for u in conflict[v]:
                        if u in temp_selected or u in added:
                            ok = False
                            break
                    if ok:
                        added.append(v)
                
                if len(added) > 1:  # net gain
                    selected = temp_selected | set(added)
                    unselected = [v for v in range(1, n + 1) if v not in selected]
                    if len(selected) > best_size:
                        best = selected.copy()
                        best_size = len(selected)
    
    # Final cleanup: add any remaining zero-conflict vertices
    for v in range(1, n + 1):
        if v not in selected:
            ok = True
            for u in conflict[v]:
                if u in selected:
                    ok = False
                    break
            if ok:
                selected.add(v)
    
    if len(selected) > len(best):
        best = selected.copy()
    
    return sorted(best)
# EVOLVE-BLOCK-END
