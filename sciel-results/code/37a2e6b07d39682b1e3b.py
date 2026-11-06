"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 115
    best = [None]
    best_max = [float('inf')]
    
    def solve():
        # Backtracking: build Sidon set containing 0, covering diffs 1..30
        # State: sorted list of elements, set of used positive differences
        elements = [0]
        used_diffs = set()
        need = set(range(1, 31))
        
        def backtrack():
            if time.time() > deadline:
                return False
            if not need:
                # All differences covered
                mx = elements[-1]
                if mx < best_max[0]:
                    best_max[0] = mx
                    best[0] = list(elements)
                return True
            
            # Try adding a new element x > elements[-1]
            # x must be at least 1 (since 0 is already there)
            # We want to keep x small to minimize the max element
            # x must not create duplicate differences
            # For each existing element a_i, x - a_i must not be in used_diffs
            # and x - a_i > 0 (guaranteed since x > all a_i)
            
            # Lower bound: x must be >= elements[-1] + 1
            # Upper bound: we want to minimize, so try small x first
            # But x can't be too large or we won't beat best_max[0]
            upper = min(best_max[0], 2000)  # safety cap
            
            for x in range(elements[-1] + 1, upper + 1):
                if time.time() > deadline:
                    return False
                # Check if x creates any duplicate difference
                valid = True
                new_diffs = []
                for a_i in elements:
                    d = x - a_i
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                
                # This x is valid; add it
                elements.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                    if d in need:
                        need.discard(d)
                
                if backtrack():
                    # Undo and return success
                    elements.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                    return True
                
                # Undo
                elements.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        backtrack()
    
    solve()
    
    if best[0] is not None:
        return best[0]
    
    # Fallback: use the original greedy approach
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
# EVOLVE-BLOCK-END
