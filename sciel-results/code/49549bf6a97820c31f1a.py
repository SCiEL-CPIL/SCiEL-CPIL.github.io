import time

def construct():
    p = 1009
    deadline = time.time() + 85.0 # Safety margin for 90s limit

    # Precompute roots for each quadratic residue
    roots = [0] * p
    for r in range(1, p):
        if time.time() > deadline:
            break
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r
            
    is_res = [False] * p
    for res in range(1, p):
        if roots[res] != 0:
            is_res[res] = True
            
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            candidate_roots = [roots[s+i] for i in range(5)]
            # Verification pass
            valid = True
            for i in range(5):
                if (candidate_roots[i] * candidate_roots[i]) % p != (s + i) % p:
                    valid = False
                    break
            if valid:
                return {
                    'start': s,
                    'roots': candidate_roots
                }
            
    # Fallback: if we timed out or didn't find a verified solution, return best guess or raise
    # Since we break on deadline, we might not have found the min s.
    # But the loop is very fast, so this is unlikely.
    raise RuntimeError("No solution found or timeout")
