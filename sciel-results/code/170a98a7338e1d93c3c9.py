"""B_3[1] set construction: maximize cardinality with unique unordered triple sums.

Strategy:
  1. Fast bytearray-based conflict checking for O(1) sum lookups.
  2. Recursive backtracking search with MRV (Minimum Remaining Values) heuristic.
  3. Multi-phase approach: quick greedy baseline, then targeted backtracking
     to extend or replace elements to increase cardinality.
"""
import time
import random
import sys

import numpy as np

MAX_N = 4096
MAX_SUM = 3 * MAX_N  # 12288

def _build_sums_bitset(S):
    """Build a bytearray bitset of all triple sums for set S."""
    bitset = bytearray(MAX_SUM + 1)
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                bitset[S[i] + S[j] + S[k]] = 1
    return bitset

def _check_valid(x, S, bitset):
    """Check if x can be added to S without conflict. Uses bitset for O(1) lookups."""
    n = len(S)
    # Check (x, x, x)
    if bitset[3 * x]:
        return False
    # Check (x, x, y) and (x, y, y) for y in S
    for y in S:
        if bitset[2 * x + y]:
            return False
        if bitset[x + 2 * y]:
            return False
    # Check (x, y, z) for y <= z in S
    for i in range(n):
        for j in range(i, n):
            if bitset[x + S[i] + S[j]]:
                return False
    return True

def _add_to_bitset(x, S, bitset):
    """Add x to S's bitset (caller must also append x to S)."""
    n = len(S)
    bitset[3 * x] = 1
    for y in S:
        bitset[2 * x + y] = 1
        bitset[x + 2 * y] = 1
    for i in range(n):
        for j in range(i, n):
            bitset[x + S[i] + S[j]] = 1

def _remove_from_bitset(x, S, bitset):
    """Remove x from S's bitset. Rebuilds bitset for correctness."""
    # Rebuild from scratch (S already has x removed)
    n = len(S)
    bitset[:] = b'\x00' * (MAX_SUM + 1)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                bitset[S[i] + S[j] + S[k]] = 1

def _greedy_bitset(candidates, deadline):
    """Fast greedy using bytearray bitset for O(1) conflict checks."""
    S = []
    bitset = bytearray(MAX_SUM + 1)
    
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        if _check_valid(x, S, bitset):
            S.append(x)
            _add_to_bitset(x, S, bitset)
    
    return S, bitset

def _backtrack_extend(S, bitset, deadline, max_depth=15):
    """
    Try to extend S by finding additional elements via backtracking.
    Uses MRV: at each step, pick the smallest valid candidate.
    Returns (S, bitset) possibly extended.
    """
    S = list(S)
    bitset = bytearray(bitset)
    in_S = set(S)
    
    # Find all currently valid candidates
    def find_valid_candidates():
        valid = []
        for x in range(1, MAX_N + 1):
            if x not in in_S and _check_valid(x, S, bitset):
                valid.append(x)
        return valid
    
    # Greedy extension first
    candidates = find_valid_candidates()
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        S.append(x)
        _add_to_bitset(x, S, bitset)
        in_S.add(x)
    
    # Now try backtracking to find more
    best_size = len(S)
    
    def bt(depth, current_S, current_bitset, current_in_S):
        nonlocal best_size
        if deadline and time.time() > deadline:
            return
        if depth >= max_depth:
            return
        if len(current_S) > best_size:
            best_size = len(current_S)
            # Copy current state as best
            S[:] = current_S
            bitset[:] = current_bitset
            in_S = set(current_S)
        
        # Find valid candidates, sorted by "constraint" (fewer future options = more constrained)
        # Approximation: smaller values tend to be more constrained
        candidates = []
        for x in range(1, MAX_N + 1):
            if x not in current_in_S and _check_valid(x, current_S, current_bitset):
                candidates.append(x)
        
        if not candidates:
            return
        
        # Try a limited number of candidates (MRV: try most constrained first)
        # Use smallest values as proxy for most constrained
        for x in candidates[:min(5, len(candidates))]:
            if deadline and time.time() > deadline:
                return
            
            new_S = current_S + [x]
            new_bitset = bytearray(current_bitset)
            _add_to_bitset(x, current_S, new_bitset)
            new_in_S = current_in_S | {x}
            
            bt(depth + 1, new_S, new_bitset, new_in_S)
    
    bt(0, S, bitset, in_S)
    return S, bitset

def _swap_improve(S, bitset, deadline, rng, max_swaps=100):
    """
    Try to improve S by swapping elements: remove one, add another.
    Uses bitset for fast conflict checking.
    """
    S = list(S)
    bitset = bytearray(bitset)
    in_S = set(S)
    best_size = len(S)
    
    for _ in range(max_swaps):
        if deadline and time.time() > deadline:
            break
        
        idx_remove = rng.randint(0, len(S) - 1)
        x_remove = S[idx_remove]
        
        # Try to find a replacement
        candidates = [x for x in range(1, MAX_N + 1) if x not in in_S]
        if not candidates:
            break
        
        # Try a few random candidates
        tried = 0
        while tried < 5:
            if deadline and time.time() > deadline:
                break
            x_add = rng.choice(candidates)
            tried += 1
            
            # Temporarily remove x_remove
            S.pop(idx_remove)
            _remove_from_bitset(x_remove, S, bitset)
            in_S.discard(x_remove)
            
            if _check_valid(x_add, S, bitset):
                S.append(x_add)
                _add_to_bitset(x_add, S, bitset)
                in_S.add(x_add)
                break
            else:
                # Restore
                S.insert(idx_remove, x_remove)
                _add_to_bitset(x_remove, S, bitset)
                in_S.add(x_remove)
    
    return S, bitset

def _is_valid_triples(S):
    """Check if all unordered triple sums (with replacement) are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    rng = random.Random(42)
    
    best_S = []
    
    # Phase 1: Multi-phase greedy to get a strong baseline
    # Phase 1a: ascending
    S, bitset = _greedy_bitset(list(range(1, MAX_N + 1)), deadline)
    best_S = S
    
    # Phase 1b: descending
    if time.time() < deadline:
        S, bitset = _greedy_bitset(list(range(MAX_N, 0, -1)), deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 1c: random shuffles
    for attempt in range(30):
        if time.time() > deadline:
            break
        candidates = list(range(1, MAX_N + 1))
        rng.shuffle(candidates)
        S, bitset = _greedy_bitset(candidates, deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 1d: biased toward small numbers
    for attempt in range(20):
        if time.time() > deadline:
            break
        alpha = rng.uniform(0.5, 3.0)
        weights = np.array([1.0 / (i + 1) ** alpha for i in range(MAX_N)])
        weights /= weights.sum()
        candidates = np.random.choice(MAX_N, size=MAX_N, replace=False, p=weights) + 1
        S, bitset = _greedy_bitset(candidates.tolist(), deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Phase 2: Extend the best set using backtracking
    if time.time() < deadline:
        bitset = _build_sums_bitset(best_S)
        best_S, bitset = _backtrack_extend(best_S, bitset, deadline, max_depth=10)
    
    # Phase 3: Local search to improve further
    if time.time() < deadline:
        bitset = _build_sums_bitset(best_S)
        best_S, bitset = _swap_improve(best_S, bitset, deadline, rng, max_swaps=200)
    
    # Phase 4: More random restarts with local search
    no_improve = 0
    while time.time() < deadline and no_improve < 20:
        seed = rng.randint(0, 10**9)
        rng2 = random.Random(seed)
        
        # Random candidate ordering
        candidates = list(range(1, MAX_N + 1))
        rng2.shuffle(candidates)
        S, bitset = _greedy_bitset(candidates, deadline)
        
        # Extend with backtracking
        if time.time() < deadline:
            S, bitset = _backtrack_extend(S, bitset, deadline, max_depth=8)
        
        # Local search
        if time.time() < deadline:
            S, bitset = _swap_improve(S, bitset, deadline, rng2, max_swaps=100)
        
        if len(S) > len(best_S):
            best_S = S
            no_improve = 0
        else:
            no_improve += 1
    
    # Final verification
    if not _is_valid_triples(best_S):
        # Fallback: rebuild from scratch with a simple greedy
        best_S, _ = _greedy_bitset(list(range(1, MAX_N + 1)), deadline)
    
    return best_S

if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time()-s:.2f}s, Size: {len(res)}")
    if _is_valid_triples(res):
        print("Valid")
    else:
        print("Invalid")
