"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 192
MAX_K = 48
DEADLINE_MARGIN = 5.0  # seconds to keep before the 90s kill


def _is_valid_set(selected):
    """Full validation of a set. Used only for final check."""
    for a in selected:
        others = [v for v in selected if v != a]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True


def _valid_addition(x, selected):
    """Check if adding x to selected keeps the set valid.
    Optimized: Check x against existing triples, and existing elements against new triples involving x.
    """
    sel = selected
    # Condition 1: x must not divide sum of any 3 distinct others in sel
    if len(sel) >= 3:
        for t in combinations(sel, 3):
            if sum(t) % x == 0:
                return False
    
    # Condition 2: For each a in sel, x + (two others) must not be divisible by a
    # This is equivalent to: for all pairs (u, v) in sel, u != v, (x + u + v) % a != 0 for all a in sel?
    # No, the condition is: a does not divide (x + u + v) where u, v are distinct from a and x.
    # So for each a in sel, we check pairs (u, v) from sel \ {a}.
    for a in sel:
        others = [v for v in sel if v != a]
        if len(others) >= 2:
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (x + others[i] + others[j]) % a == 0:
                        return False
    return True


def _greedy_removal(pool_start, pool_end):
    """Start with all elements in [pool_start, pool_end], remove those causing violations."""
    selected = list(range(pool_start, pool_end + 1))
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            a = selected[i]
            others = [v for v in selected if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        selected.pop(i)
                        changed = True
                        break
            if changed:
                break
    return selected


def _simple_local_search(selected, time_limit, candidates):
    """Fast local search: try adding candidates, then remove-one-add-two if stuck."""
    selected = list(selected)
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Step 1: Try to add any candidate
        for x in candidates:
            if time.time() >= deadline:
                break
            if x not in selected:
                if _valid_addition(x, selected):
                    selected.append(x)
                    improved = True
                    break
        
        # Step 2: If no addition possible, try remove-one-add-two
        if not improved:
            for i in range(len(selected)):
                if time.time() >= deadline:
                    break
                a = selected[i]
                sel_without_a = selected[:i] + selected[i+1:]
                added = []
                # Try to add 2 candidates
                for x in candidates:
                    if time.time() >= deadline:
                        break
                    if x == a or x in sel_without_a or x in added:
                        continue
                    if _valid_addition(x, sel_without_a + added):
                        added.append(x)
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    selected = sel_without_a + added
                    improved = True
                    break
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    # Phase 1: Greedy removal from dense upper pool
    # Start with a range that is likely to yield high cardinality
    best = _greedy_removal(120, 192)
    best_size = len(best)
    
    # Phase 2: Try different pool starts if time permits
    if time.time() < deadline - 15:
        for ps in [100, 80, 60]:
            if time.time() >= deadline - 15:
                break
            candidate = _greedy_removal(ps, 192)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search to add more elements
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        # Candidates are all numbers not in best
        candidates = [x for x in range(1, N + 1) if x not in best]
        # Sort candidates descending to prioritize larger numbers (heuristic)
        candidates.sort(reverse=True)
        best = _simple_local_search(best, remaining, candidates)
        best_size = len(best)
        
    # Final validation
    if not _is_valid_set(best):
        # Fallback to a safe smaller set if validation fails
        best = _greedy_removal(144, 192)
        
    return best
# EVOLVE-BLOCK-END
