"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 192
MAX_SIZE = 48

def count_violations(selected):
    """Count total violations and find the most violating element."""
    sel = selected
    n = len(sel)
    violation_counts = {a: 0 for a in sel}
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                violation_counts[a] += 1
    return violation_counts

def is_valid(selected):
    sel = selected
    n = len(sel)
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def can_add(selected, x):
    if x in selected:
        return False
    candidate = selected + [x]
    n = len(candidate)
    for i in range(n):
        a = candidate[i]
        others = [candidate[j] for j in range(n) if j != i]
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Start with upper half
    selected = list(range(96, N + 1))  # 97 elements
    
    # Iteratively remove most-violating element
    while time.time() < deadline:
        if is_valid(selected):
            break
        vc = count_violations(selected)
        # Remove element with most violations
        worst = max(vc, key=vc.get)
        selected.remove(worst)
    
    # Now try to add more elements
    if len(selected) < MAX_SIZE:
        for x in range(1, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > deadline:
                break
            if x not in selected:
                if can_add(selected, x):
                    selected.append(x)
    
    # Local search
    if len(selected) < MAX_SIZE and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for x in range(1, N + 1):
                if len(selected) >= MAX_SIZE:
                    break
                if x not in selected:
                    if can_add(selected, x):
                        selected.append(x)
                        improved = True
                        break
            # Try swap
            if not improved:
                for r in selected:
                    if time.time() > deadline:
                        break
                    temp = [v for v in selected if v != r]
                    for a in range(1, N + 1):
                        if a not in temp:
                            if can_add(temp, a):
                                if len(temp) + 1 >= len(selected):
                                    selected = temp + [a]
                                    improved = True
                                    break
                    if improved:
                        break
    
    return selected
# EVOLVE-BLOCK-END
