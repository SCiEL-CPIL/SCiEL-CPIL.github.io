"""Optimized branch-and-bound for 3-uniform hypergraph independent set."""
# EVOLVE-BLOCK-START
import time
import random

_N = 256
_CAND = list(range(2, _N + 1))
_M = len(_CAND)
_IDX = {v: i for i, v in enumerate(_CAND)}

# Precompute conflict masks
# For each vertex v, we store a list of bitmasks mask_u such that
# if u is in the current set, then (mask_u & current_set) != 0 implies conflict.
# Specifically, conflict exists if there is a pair {u, w} in the current set
# such that {v, u, w} is a hyperedge.
# We precompute for each v and each u, a bitset of w's that form a conflict with v and u.
# conflict_mask[v][u] is a bitset of w.

conflict_masks = []
for v_idx in range(_M):
    v_val = _CAND[v_idx]
    masks_for_v = [0] * _M
    for u_idx in range(_M):
        if u_idx == v_idx:
            continue
        u_val = _CAND[u_idx]
        mask_w = 0
        for w_idx in range(_M):
            if w_idx == v_idx or w_idx == u_idx:
                continue
            w_val = _CAND[w_idx]
            # Check if {v, u, w} is a conflict
            # Conflict if v|u*w OR u|v*w OR w|u*v
            if (u_val * w_val) % v_val == 0 or (v_val * w_val) % u_val == 0 or (u_val * v_val) % w_val == 0:
                mask_w |= (1 << w_idx)
        masks_for_v[u_idx] = mask_w
    conflict_masks.append(masks_for_v)

_TIMEOUT = 55.0  # Safety margin for 60s limit

def _valid_solution(indices):
    """Verify a solution is valid."""
    S = set(indices)
    for a in S:
        ai = _IDX[a]
        for b in S:
            if b == a: continue
            bi = _IDX[b]
            for c in S:
                if c == a or c == b: continue
                ci = _IDX[c]
                # Check conflict
                if (b * c) % a == 0:
                    return False
    return True

def _branch_and_bound(order, deadline):
    """
    B&B to find max independent set.
    order: list of vertex indices to consider.
    deadline: time.time() limit.
    """
    best_set = set()
    best_size = 0
    
    # Precompute popcount for small integers? No, Python int.bit_count() is fast.
    
    def _rec(idx, current_set_mask):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        # Upper bound: current size + remaining vertices
        current_size = current_set_mask.bit_count()
        remaining_count = len(order) - idx
        if current_size + remaining_count <= best_size:
            return
            
        if idx == len(order):
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set
                temp_set = set()
                m = current_set_mask
                while m:
                    lsb = m & -m
                    temp_set.add(lsb.bit_length() - 1)
                    m ^= lsb
                best_set = temp_set
            return
            
        v_idx = order[idx]
        
        # Option 1: Exclude v_idx
        _rec(idx + 1, current_set_mask)
        
        # Option 2: Include v_idx
        # Check if v_idx can be added to current_set_mask
        # Conflict if there exists u in current_set such that
        # conflict_masks[v_idx][u] & current_set_mask != 0
        # Note: conflict_masks[v_idx][u] contains w's. If w is also in current_set, conflict.
        # We must iterate u in current_set.
        
        can_add = True
        m = current_set_mask
        v_masks = conflict_masks[v_idx]
        while m:
            lsb = m & -m
            u_idx = lsb.bit_length() - 1
            if v_masks[u_idx] & current_set_mask:
                can_add = False
                break
            m ^= lsb
            
        if can_add:
            _rec(idx + 1, current_set_mask | (1 << v_idx))

    _rec(0, 0)
    
    # Convert best_set (indices) to values
    return [_CAND[i] for i in best_set]

def construct():
    t0 = time.time()
    deadline = t0 + _TIMEOUT
    
    # Order: Descending values (larger numbers have fewer divisors, less likely to be 'a')
    # But also less likely to be factors? 
    # Heuristic: Sort by value descending.
    order = list(range(_M - 1, -1, -1))
    
    # Initial greedy solution
    greedy_set = set()
    greedy_mask = 0
    for v_idx in order:
        # Check if v_idx can be added to greedy_set
        can_add = True
        m = greedy_mask
        v_masks = conflict_masks[v_idx]
        while m:
            lsb = m & -m
            u_idx = lsb.bit_length() - 1
            if v_masks[u_idx] & greedy_mask:
                can_add = False
                break
            m ^= lsb
        if can_add:
            greedy_set.add(v_idx)
            greedy_mask |= (1 << v_idx)
            
    best_result = [_CAND[i] for i in greedy_set]
    best_size = len(greedy_set)
    
    # Run B&B
    if time.time() < deadline - 2.0:
        res = _branch_and_bound(order, deadline - 2.0)
        if len(res) > best_size:
            best_size = len(res)
            best_result = res
            
    # Try a few random orders if time permits
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            shuffled_order = order[:]
            rng.shuffle(shuffled_order)
            res = _branch_and_bound(shuffled_order, deadline - 2.0)
            if len(res) > best_size:
                best_size = len(res)
                best_result = res
                
    # Final validation
    if not _valid_solution(best_result):
        # Fallback: Primes
        primes = []
        for x in range(2, _N + 1):
            is_prime = True
            for p in range(2, int(x**0.5) + 1):
                if x % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(x)
        best_result = primes
        
    return sorted(best_result)
# EVOLVE-BLOCK-END
