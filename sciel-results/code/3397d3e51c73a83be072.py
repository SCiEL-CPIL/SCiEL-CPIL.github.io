import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0
    p = 1009
    m = 5

    # Precompute roots for all nonzero quadratic residues
    # root[r] = smallest x in [1, p-1] such that x^2 % p == r
    root = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if root[r] == 0 or x < root[r]:
            root[r] = x

    # Build boolean array for O(1) residue check
    is_residue = [False] * p
    for r in range(1, p):
        if root[r] != 0:
            is_residue[r] = True

    best_s = None
    best_roots = None

    # Search for minimal s
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        
        ok = True
        for i in range(m):
            if not is_residue[s + i]:
                ok = False
                break
        
        if ok:
            best_s = s
            best_roots = [root[s + i] for i in range(m)]
            break

    if best_s is None:
        # Fallback if deadline hit or no solution found in forward pass (unlikely)
        # Search backwards just in case, though forward should find min
        for s in range(p - m, 0, -1):
            if all(is_residue[s + i] for i in range(m)):
                best_s = s
                best_roots = [root[s + i] for i in range(m)]
                break

    if best_s is None:
        raise RuntimeError('No run found')

    return {'start': best_s, 'roots': best_roots}

# EVOLVE-BLOCK-END
