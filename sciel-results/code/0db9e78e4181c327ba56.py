"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_HARD_LIMIT = 110.0
_SOFT_MARGIN = 5.0

def _check_deadline():
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _check_deadline():
        # Fallback: Pick the one with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    min_durs = []
    max_durs = []
    rem_works = []
    rem_ops = []
    flexes = []
    
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durs = list(mo.values())
            min_durs.append(min(durs))
            max_durs.append(max(durs))
            flexes.append(len(mo))
        else:
            min_durs.append(0)
            max_durs.append(0)
            flexes.append(0)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        
    # Normalize
    def norm(vals):
        v = np.array(vals)
        if v.max() == v.min():
            return np.ones_like(v)
        return (v - v.min()) / (v.max() - v.min())
        
    n_min = norm(min_durs)
    n_max = norm(max_durs)
    n_rw = norm(rem_works)
    n_ro = norm(rem_ops)
    n_flex = norm(flexes)
    
    # Invert flex: less flex is better
    n_flex_inv = 1.0 - n_flex
    
    # Weights
    # LPT on min duration is strong for open shop
    # Remaining work is also strong
    # Constrained first (flex_inv) is a good tie breaker
    
    scores = 0.5 * n_min + 0.3 * n_rw + 0.1 * n_ro + 0.1 * n_flex_inv
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
