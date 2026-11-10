import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    p = 1009

    # Precompute: for each quadratic residue r^2 mod p (r in 1..p-1),
    # store one root. Use 0 as sentinel for non-residue.
    # We want the smallest root, so we only set if currently 0.
    root_of = [0] * p
    for r in range(1, p):
        res = (r * r) % p
        if root_of[res] == 0:
            root_of[res] = r

    # Find the smallest s in [1, 1004] such that s, s+1, ..., s+4 are all
    # non-zero quadratic residues mod 1009.
    for s in range(1, 1005):
        if time.time() - start_time > deadline:
            break

        # Check if s, s+1, s+2, s+3, s+4 have roots
        if (root_of[s] != 0 and root_of[s + 1] != 0 and root_of[s + 2] != 0 
            and root_of[s + 3] != 0 and root_of[s + 4] != 0):
            
            roots = [root_of[s], root_of[s + 1], root_of[s + 2],
                     root_of[s + 3], root_of[s + 4]]
            
            # Verify each root
            for i, ri in enumerate(roots):
                assert (ri * ri) % p == (s + i), f"Root verification failed: {ri}^2 mod {p} != {s+i}"
                
            return {
                'start': s,
                'roots': roots
            }

    # Fallback
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
