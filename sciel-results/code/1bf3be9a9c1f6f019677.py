import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill
    N = 256
    
    # Precompute conflict matrix
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    evens = [i for i in range(2, N + 1) if i % 2 == 0]
    
    # All odds are mutually compatible (odd*odd=odd, odd+odd=even, even∤odd)
    base_set = set(odds)
    
    # Find evens compatible with all odds
    compatible_evens = []
    for e in evens:
        is_compat = True
        for o in odds:
            if conflict[e][o]:
                is_compat = False
                break
        if is_compat:
            compatible_evens.append(e)
    
    # Build conflict graph among compatible evens
    n_comp = len(compatible_evens)
    # Map index to value and back
    comp_to_val = {i: compatible_evens[i] for i in range(n_comp)}
    val_to_comp = {v: i for i, v in compatible_evens.items()}
    
    # conflict_comp[i][j] = True if compatible_evens[i] and compatible_evens[j] conflict
    conflict_comp = [[False] * n_comp for _ in range(n_comp)]
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            v1 = compatible_evens[i]
            v2 = compatible_evens[j]
            if conflict[v1][v2]:
                conflict_comp[i][j] = True
                conflict_comp[j][i] = True
    
    # Degrees
    degrees = [sum(conflict_comp[i]) for i in range(n_comp)]
    
    def greedy_iset(order):
        """Greedy independent set given an order of vertices."""
        selected = set()
        for idx in order:
            if idx in selected:
                continue
            can_add = True
            for s in selected:
                if conflict_comp[idx][s]:
                    can_add = False
                    break
            if can_add:
                selected.add(idx)
        return selected
    
    # Try multiple greedy orders to find a good starting point
    best_selected = set()
    best_size = 0
    
    # Order 1: By degree (ascending)
    order1 = sorted(range(n_comp), key=lambda x: degrees[x])
    sel1 = greedy_iset(order1)
    if len(sel1) > best_size:
        best_selected = sel1
        best_size = len(sel1)
    
    # Order 2: Random
    for _ in range(50):
        order2 = list(range(n_comp))
        random.shuffle(order2)
        sel2 = greedy_iset(order2)
        if len(sel2) > best_size:
            best_selected = sel2
            best_size = len(sel2)
    
    # Local search: try removing one element and re-greedy
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        for i in list(best_selected):
            new_set = set(best_selected)
            new_set.remove(i)
            # Re-greedy: try to add any unselected vertex
            changed = True
            while changed:
                changed = False
                for j in range(n_comp):
                    if j in new_set:
                        continue
                    can_add = True
                    for s in new_set:
                        if conflict_comp[j][s]:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(j)
                        changed = True
            if len(new_set) > best_size:
                best_selected = new_set
                best_size = len(best_size)
                improved = True
            elif len(new_set) > best_size:
                best_selected = new_set
                best_size = len(new_set)
                improved = True
    
    # Also try 2-swap: remove two, add up to three
    for i in best_selected:
        if time.time() > DEADLINE - 5:
            break
        for j in best_selected:
            if i == j:
                continue
            new_set = set(best_selected)
            new_set.discard(i)
            new_set.discard(j)
            changed = True
            while changed:
                changed = False
                for k in range(n_comp):
                    if k in new_set:
                        continue
                    can_add = True
                    for s in new_set:
                        if conflict_comp[k][s]:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(k)
                        changed = True
            if len(new_set) > best_size:
                best_selected = new_set
                best_size = len(new_set)
                break
    
    # Convert back to values
    added_evens = [compatible_evens[i] for i in best_selected]
    
    result = odds + added_evens
    
    # Final verification
    for idx_a, a in enumerate(result):
        for b in result[idx_a + 1:]:
            assert (a * b) % (a + b) != 0, f"Conflict found: {a}, {b}"
    
    return result
