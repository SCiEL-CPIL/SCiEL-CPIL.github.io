"""Rewritten program: fast greedy with strict deadline and limited improvements."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Strict deadline to ensure we return before the 120s evaluator kill
    deadline = time.time() + 100
    best_rows = None
    best_scope = None

    def run_greedy(rng=None):
        """Runs a standard greedy construction. If rng is provided, it may be used for tie-breaking or skipping, but for simplicity and speed, we stick to deterministic greedy first."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x is valid
                valid = True
                for a in row:
                    diff = x - a
                    if diff in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    # 1. Baseline Deterministic Greedy
    rows = run_greedy()
    best_rows = rows
    best_scope = max(max(r) for r in rows)

    # 2. Try a few randomized variants if time permits
    # Strategy: Sometimes skip the smallest valid candidate to allow better packing later?
    # This is risky and hard to do greedily. 
    # Instead, let's try a "Greedy with Randomized Start" for the last few elements?
    # Or just multiple restarts with a slightly different heuristic: 
    # Sort candidates by frequency of use? No, too slow.
    
    # A known better heuristic for DSTs is to use a "Backtracking" with pruning, 
    # but that's dangerous for time.
    
    # Let's try a simple Local Search / Hill Climbing on the generated rows?
    # No, changing a row breaks the global difference constraint.
    
    # Given the time constraint, the deterministic greedy is the safest.
    # To potentially improve, we can try a "Greedy with Lookahead 1":
    # When choosing x, check if choosing x+1 allows the rest of the row to be completed with a smaller max?
    # This is expensive.
    
    # Let's just ensure the deadline is respected.
    # If we have time, we can try a few random permutations of the *order* of rows?
    # The order of rows doesn't matter for the set of differences, only the content.
    
    # Let's try a simple random restart of the greedy but with a "bias" to pick larger numbers occasionally?
    # This usually increases the scope, so it's not helpful for minimization.
    
    # The most effective simple improvement is often just trying different "seeds" for a randomized greedy 
    # that picks from the top N valid candidates.
    
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for _ in range(100): # Limit attempts to avoid infinite loop
            if time.time() > deadline - 5:
                break
            
            used = set()
            rows = []
            ok = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Find the next few valid candidates
                    candidates = []
                    # Look ahead a bit
                    for cand in range(x + 1, x + 50):
                        valid = True
                        new_diffs = []
                        for a in row:
                            diff = cand - a
                            if diff in used:
                                valid = False
                                break
                            new_diffs.append(diff)
                        if valid:
                            candidates.append((cand, new_diffs))
                        if len(candidates) >= 5:
                            break
                    
                    if not candidates:
                        ok = False
                        break
                    
                    # Pick one. Deterministically pick the smallest (first one).
                    # To add randomness, we can pick randomly from the top 3 with low probability.
                    if rng.random() < 0.1 and len(candidates) > 1:
                        idx = rng.randint(0, min(2, len(candidates)-1))
                    else:
                        idx = 0
                    
                    cand, new_diffs = candidates[idx]
                    row.append(cand)
                    x = cand
                    used.update(new_diffs)
                
                if not ok:
                    break
                rows.append(row)
            
            if ok:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]

    return best_rows
# EVOLVE-BLOCK-END
