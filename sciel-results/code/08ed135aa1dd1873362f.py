"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 256

def _generate_aps():
    """Generate all arithmetic progressions as index lists."""
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N + 1):
            length = (N - start) // d + 1
            if length >= 1:
                indices = list(range(start - 1, start - 1 + length * d, d))
                aps.append(indices)
    return aps

def construct():
    deadline = time.time() + 78.0
    ap_indices = _generate_aps()
    
    def eval_max(signs):
        max_abs = 0
        for indices in ap_indices:
            s = 0
            for idx in indices:
                s += signs[idx]
            a = abs(s)
            if a > max_abs:
                max_abs = a
        return max_abs
    
    def eval_max_worst(signs):
        max_abs = 0
        worst = -1
        for i, indices in enumerate(ap_indices):
            s = 0
            for idx in indices:
                s += signs[idx]
            a = abs(s)
            if a > max_abs:
                max_abs = a
                worst = i
        return max_abs, worst
    
    rng = random.Random(999)
    best_signs = None
    best_max = float('inf')
    
    while time.time() < deadline:
        signs = [rng.choice((-1, 1)) for _ in range(N)]
        cur_max, _ = eval_max_worst(signs)
        
        stalled = 0
        while stalled < 30 and time.time() < deadline:
            cur_max, widx = eval_max_worst(signs)
            if widx < 0 or cur_max <= 1:
                break
            
            waps = ap_indices[widx]
            best_pos = -1
            best_nm = cur_max
            
            for p in waps:
                signs[p] = -signs[p]
                nm, _ = eval_max_worst(signs)
                signs[p] = -signs[p]
                if nm < best_nm:
                    best_nm = nm
                    best_pos = p
            
            if best_pos >= 0 and best_nm < cur_max:
                signs[best_pos] = -signs[best_pos]
                cur_max = best_nm
                stalled = 0
            else:
                stalled += 1
        
        if cur_max < best_max:
            best_max = cur_max
            best_signs = signs[:]
            
            if best_max <= 4 and time.time() < deadline - 5:
                # Polish
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    cm, wi = eval_max_worst(signs)
                    if wi < 0 or cm <= 1:
                        break
                    imp = False
                    for p in ap_indices[wi]:
                        signs[p] = -signs[p]
                        nm, _ = eval_max_worst(signs)
                        signs[p] = -signs[p]
                        if nm < cm:
                            signs[p] = -signs[p]
                            cm = nm
                            imp = True
                            break
                    if not imp:
                        break
                if cm < best_max:
                    best_max = cm
                    best_signs = signs[:]
    
    return best_signs
# EVOLVE-BLOCK-END
