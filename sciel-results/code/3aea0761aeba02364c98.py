import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Heuristic: Sort by descending number of divisors
    # Moduli with more divisors have more potential conflicts (larger gcds with more numbers)
    def num_divisors(num):
        count = 0
        for i in range(1, int(num**0.5) + 1):
            if num % i == 0:
                count += 1
                if i != num // i:
                    count += 1
        return count

    # Order indices by difficulty (most constrained first)
    order = sorted(range(n), key=lambda i: (-num_divisors(moduli[i]), moduli[i]))
    
    # Reverse order index for quick lookup? Not needed if we use order list.
    
    best_solution = []
    best_len = 0
    nodes_visited = 0

    def is_compatible(idx, r, selected):
        for (idx2, r2) in selected:
            if (r - r2) % gcd_tab[idx][idx2] == 0:
                return False
        return True

    def forward_check(pos, selected):
        # Check if all unassigned moduli from pos onwards have at least one valid residue
        # selected contains assignments for order[0:pos]
        for k in range(pos, n):
            idx = order[k]
            # Check if moduli[idx] has any valid residue
            found = False
            for r in range(moduli[idx]):
                if is_compatible(idx, r, selected):
                    found = True
                    break
            if not found:
                return False
        return True

    def backtrack(pos, selected):
        nonlocal best_solution, best_len, nodes_visited
        nodes_visited += 1
        
        if time.time() > deadline - 5:
            return

        if len(selected) > best_len:
            best_solution = selected[:]
            best_len = len(selected)

        if pos == n:
            return

        idx = order[pos]
        
        # Option 1: Try to assign a residue to idx
        # We try all residues. To speed up, we can just pick the first valid one?
        # No, we need to explore if the first valid one leads to a dead end later.
        # But for maximizing size, if we can assign, we should.
        # If we assign and later fail, we backtrack.
        
        for r in range(moduli[idx]):
            if is_compatible(idx, r, selected):
                new_selected = selected + [(idx, r)]
                # Forward check
                if forward_check(pos + 1, new_selected):
                    backtrack(pos + 1, new_selected)
                    # If we found a full solution or a very good one, we might prune?
                    # For now, continue to see if we can do better?
                    # Actually, if we found a solution of size n, we can stop.
                    if best_len == n:
                        return

        # Option 2: Skip idx
        # Only skip if we couldn't assign? Or always try skip?
        # If we always try skip, we explore 2^n.
        # If we only skip when assignment fails, we are doing a greedy with backtracking on residues.
        # This finds a maximal set for the order, but not necessarily the maximum size set
        # if skipping an early "easy" modulus allows adding two "hard" later ones.
        
        # However, with "hard first" ordering, skipping a hard modulus is less likely to be beneficial
        # than assigning it, unless it blocks many others.
        
        # Let's try skip only if no residue was compatible?
        # No, even if a residue is compatible, it might block later ones.
        # But forward_check should catch that.
        # If forward_check(pos+1, new_selected) is true, it means later ones *have* options.
        # It doesn't guarantee a solution, but it's a good heuristic.
        
        # If we exhausted all residues for idx (none compatible), we MUST skip.
        # If some were compatible, we tried them. If they all led to dead ends (backtracked),
        # then we should try skipping idx.
        
        # The loop above tries all compatible residues.
        # If we are here, it means either:
        # 1. No residue was compatible.
        # 2. All compatible residues led to dead ends.
        
        # In both cases, we should try skipping idx.
        if forward_check(pos + 1, selected):
            backtrack(pos + 1, selected)

    backtrack(0, [])

    # Fallback to greedy if CSP didn't find anything (shouldn't happen)
    if not best_solution:
        selected = []
        for idx in order:
            for r in range(moduli[idx]):
                if is_compatible(idx, r, selected):
                    selected.append((idx, r))
                    break
        best_solution = selected

    return [[r, moduli[idx]] for (idx, r) in best_solution]
