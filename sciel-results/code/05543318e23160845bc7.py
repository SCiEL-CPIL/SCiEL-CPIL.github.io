"""Difference triangle set (4,7) - improved solver."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_solution = None
    best_scope = float('inf')
    
    # First, try deterministic greedy as baseline
    det_rows = _greedy_deterministic(n_rows, k)
    if det_rows is not None:
        best_solution = det_rows
        best_scope = max(max(r) for r in det_rows)
    
    # Try randomized greedy with multiple restarts and varying biases
    rng = random.Random(12345)
    
    # Try a range of bias parameters
    bias_values = [0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.4, 0.6, 0.8]
    
    for bias in bias_values:
        if time.time() >= deadline:
            break
        for _ in range(100):  # 100 attempts per bias
            if time.time() >= deadline:
                break
            rows = _greedy_randomized(n_rows, k, rng, prefer_small=bias)
            if rows is not None:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = rows
                    if best_scope <= 112:
                        return best_solution
    
    # Final polish: try to reduce the maximum entry
    if best_solution is not None and time.time() < deadline - 10:
        polished = _polish(best_solution, n_rows, k, deadline)
        if polished is not None:
            scope = max(max(r) for r in polished)
            if scope < best_scope:
                best_scope = scope
                best_solution = polished
    
    return best_solution


def _greedy_deterministic(n_rows, k):
    """Original greedy: pick smallest valid x each time."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_randomized(n_rows, k, rng, prefer_small=0.5):
    """
    Randomized greedy: at each step, consider valid x values and pick with bias.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        x = 0
        
        while len(row) < k + 1:
            # Find candidates in a window
            candidates = []
            # Search window: start from x+1, go up to x+40 or until 15 candidates
            for trial in range(40):
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append(x)
                    if len(candidates) >= 15:
                        break
            
            if not candidates:
                return None  # stuck
            
            # Pick with bias toward smaller values
            candidates.sort()
            # Weight: w_i = (1 - prefer_small) + prefer_small * (1 - i/len)
            n_cand = len(candidates)
            weights = []
            for i, c in enumerate(candidates):
                w = (1 - prefer_small) + prefer_small * (1.0 - i / max(1, n_cand - 1))
                weights.append(w)
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[-1]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
    
    return rows


def _polish(rows, n_rows, k, deadline):
    """
    Try to reduce the maximum entry by shifting marks downward.
    """
    # Find the row with the maximum entry
    max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
    max_val = rows[max_row_idx][-1]
    
    # Try to reduce the last few marks
    # This is complex; skip for now and just return original
    # A proper polish would require checking all difference constraints
    return rows
# EVOLVE-BLOCK-END
