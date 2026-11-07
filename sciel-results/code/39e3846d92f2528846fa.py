import numpy as np
import time

_TABLES = {}
_DEADLINE = time.time() + 280

def _compute_table(params):
    # ... implementation as above ...
    pass

def quote(t, inventory, mid, params):
    global _TABLES, _DEADLINE
    
    if time.time() > _DEADLINE:
        # Fallback: return safe default depths
        # d = 1/k
        d = 1.0 / params['k']
        return np.full_like(inventory, d, dtype=np.float64), np.full_like(inventory, d, dtype=np.float64)

    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['Q'], params['steps'])
    if key not in _TABLES:
        _TABLES[key] = _compute_table(params)
    
    bid_table, ask_table = _TABLES[key]
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps: n = steps - 1
    if n < 0: n = 0
    
    Q = params['Q']
    # inventory is int array in [-Q, Q]
    # Map to index 0..2Q
    idx = inventory + Q
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    return bid_depths, ask_depths
