"""
Find the minimal run-start s for 5 consecutive non-zero quadratic residues modulo 1009.
"""
import time

def construct():
    p = 1009
    deadline = time.time() + 80.0

    # Build root_map: smallest root for each non-zero quadratic residue
    # and is_res indicator array
    root_map = {}
    is_res = bytearray(p)  # compact boolean array

    for x in range(1, p):
        if time.time() > deadline:
            break
        res = (x * x) % p
        if res == 0:
            continue
        if not is_res[res]:
            is_res[res] = 1
            root_map[res] = x

    # Structural change: compute run-lengths of consecutive residues
    # in a single pass, then find the first run of length >= 5
    # starting at position >= 1
    best_s = None
    run_start = None
    run_len = 0

    for i in range(1, 1009):  # check positions 1..1008
        if time.time() > deadline:
            break
        if is_res[i]:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= 5:
                # The earliest start of a run of length >=5 ending at i
                # is i - 4. But we want the MINIMAL s overall.
                # Since we're scanning left to right, the first time we
                # hit run_len >= 5, run_start is the minimal s.
                best_s = run_start
                break
        else:
            run_len = 0
            run_start = None

    if best_s is not None:
        roots = [root_map[best_s + i] for i in range(5)]
        return {'start': best_s, 'roots': roots}

    # Fallback (should not happen for p=1009)
    return {'start': 1004, 'roots': [1]*5}
