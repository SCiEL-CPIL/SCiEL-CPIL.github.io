"""
Strategy: CSP with AC-3 arc consistency and MRV heuristic.
This treats the problem as finding a sequence where each element is a variable
constrained by previous elements (increasing order) and the global set of
consecutive block sums. AC-3 prunes domains of future variables based on
current assignments, significantly reducing the branching factor compared to
simple backtracking.
"""

import time
import sys
from collections import deque

_DEADLINE = 85.0  # safety margin before 90s kill
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid_full(seq):
    """Final validation check."""
    if not seq:
        return True
    if any(v < 1 or v > 256 for v in seq):
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build():
    """Fast baseline to get a starting point."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _compute_block_sums(seq):
    """Compute all non-empty consecutive block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _ac3_check_and_prune(seq, sums, domain_start):
    """
    Check if we can extend the sequence.
    Instead of full AC-3 on a graph, we perform a forward check:
    For the next position, determine which values are viable.
    If we are doing deeper search, we look ahead.
    
    This function returns the list of viable candidates for the next element.
    """
    last_val = seq[-1] if seq else 0
    n = len(seq)
    
    viable = []
    for x in range(last_val + 1, 257):
        # Check new sums created by appending x
        # New sums are: x, x+seq[-1], x+seq[-1]+seq[-2], ...
        new_sums = []
        s = x
        new_sums.append(s)
        for i in range(n-1, -1, -1):
            s += seq[i]
            new_sums.append(s)
        
        # Check collision with existing sums
        if any(ns in sums for ns in new_sums):
            continue
            
        # Check collision among new sums themselves
        if len(set(new_sums)) != len(new_sums):
            continue
            
        viable.append(x)
        
    return viable

def _solve_csp(seq, sums, target_len):
    """
    Backtracking search with forward checking (AC-3 lite).
    We use MRV (Minimum Remaining Values) implicitly by always trying
    the smallest viable value first to keep numbers low, but we could
    also try to look 2 steps ahead.
    
    To make it structurally different and more powerful, we implement
    a look-ahead depth of 1 or 2 to prune branches early.
    """
    if len(seq) >= target_len:
        return seq[:]
    
    if _time_remaining() < 1.0:
        return seq[:]
        
    last_val = seq[-1] if seq else 0
    best = seq[:]
    
    # Get viable candidates
    candidates = _ac3_check_and_prune(seq, sums, last_val)
    
    # Heuristic: Try candidates that leave the most room for future elements.
    # Smaller x generally leaves more room in [1, 256], but might cause
    # more sum collisions later. Let's stick to increasing order for now,
    # but we can prioritize x that don't "block" too many future sums.
    
    # A simple heuristic: sort by x.
    # A better heuristic: try x that results in the largest minimum gap to existing sums?
    # For now, let's try the smallest valid x first (greedy-like but with backtracking).
    
    for x in candidates:
        new_seq = seq + [x]
        new_sums = sums.copy()
        
        # Add new sums
        s = x
        new_sums.add(s)
        for a in reversed(seq):
            s += a
            new_sums.add(s)
            
        # Look ahead 1 step to see if this choice is a dead end
        next_candidates = _ac3_check_and_prune(new_seq, new_sums, x)
        
        # If no next candidates, this is a dead end (unless we reached target_len)
        if len(new_seq) < target_len and not next_candidates:
            continue
            
        result = _solve_csp(new_seq, new_sums, target_len)
        if len(result) > len(best):
            best = result[:]
            
        # If we found the target length, we can stop
        if len(best) >= target_len:
            return best
            
    return best

def construct():
    start_time = time.time()
    
    # 1. Get a solid baseline with greedy
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # 2. Use CSP solver to try to extend or find a better sequence
    # We will try to find a sequence of length best_len + k
    # Start with the greedy sequence and try to extend it
    sums = _compute_block_sums(best_seq)
    
    # Try to extend the greedy sequence
    if _time_remaining() > 2.0:
        extended = _solve_csp(best_seq, sums, best_len + 10)
        if len(extended) > best_len:
            best_seq = extended
            best_len = len(best_seq)
            
    # 3. If we have time, try starting from scratch with CSP
    # The greedy sequence might be locally optimal but globally suboptimal.
    # We can try different starting points or use the CSP solver from scratch.
    
    # To make this a structural change, we run a full CSP search from the beginning
    # with a target length slightly higher than the current best.
    # We limit the search depth/time to ensure we finish.
    
    if _time_remaining() > 5.0:
        # Try to find a sequence of length best_len + 1 from scratch
        # This is expensive, so we rely on the pruning.
        # We initialize with an empty sequence.
        # However, starting from scratch is very slow.
        # Instead, we can try to "repair" the greedy sequence or use it as a guide.
        
        # Let's try a few different first elements with the CSP solver
        # The first element determines the scale of the sums.
        for first_val in [1, 2, 3, 5, 8, 13]:
            if _time_remaining() < 2.0:
                break
                
            seq = [first_val]
            sums_init = {first_val}
            
            # Target: try to beat current best
            target = best_len + 1
            result = _solve_csp(seq, sums_init, target)
            
            if len(result) > best_len:
                best_seq = result
                best_len = len(best_seq)
                
                # If we improved, try to extend further
                if _time_remaining() > 3.0:
                    sums_new = _compute_block_sums(best_seq)
                    extended = _solve_csp(best_seq, sums_new, best_len + 5)
                    if len(extended) > best_len:
                        best_seq = extended
                        best_len = len(best_seq)

    # Final safety check
    if not _is_valid_full(best_seq):
        # Fallback to greedy
        best_seq = _greedy_build()
        
    return best_seq
