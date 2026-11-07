"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    # Greedy: pick largest numbers first, check coprimality
    selected = []
    for a in range(n, 0, -1):
        if time.time() > deadline:
            break
        if all(math.gcd(a, b) == 1 for b in selected):
            selected.append(a)
    
    # Local search improvement: try to replace one element with a larger one
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        selected_set = set(selected)
        for i in range(len(selected)):
            for j in range(n, selected[i], -1):
                if j in selected_set:
                    continue
                # Check if j is coprime to all other selected elements (excluding selected[i])
                ok = True
                for k in range(len(selected)):
                    if k != i and math.gcd(j, selected[k]) != 1:
                        ok = False
                        break
                if ok:
                    selected[i] = j
                    selected_set.remove(selected[i])  # stale, but we rebuild next iteration
                    selected_set.add(j)
                    improved = True
                    break
            if improved:
                break
    
    # Try to add more elements: check if we can insert additional coprime numbers
    for a in range(n, 0, -1):
        if time.time() > deadline:
            break
        if a in selected:
            continue
        if all(math.gcd(a, b) == 1 for b in selected):
            selected.append(a)
    
    # Final verification
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # Fallback to pure greedy
                selected = []
                for a in range(n, 0, -1):
                    if all(math.gcd(a, b) == 1 for b in selected):
                        selected.append(a)
                break
    
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
