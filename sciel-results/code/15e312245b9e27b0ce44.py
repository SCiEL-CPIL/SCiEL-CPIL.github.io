"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}
_DEADLINE = None

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            for k in range(j+1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S without violating the constraint."""
    S_list = list(S)
    m = len(S_list)
    # x does not divide a*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    # a does not divide x*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    return True

def _precompute_conflicts(nums):
    """
    Precompute which triples (a,b,c) are conflicting (a divides b*c).
    Returns a dict mapping each number to the set of pairs (i,j) where 
    nums[i] and nums[j] form a conflict with this number.
    Also returns conflict_count[x] = number of pairs that conflict with x.
    """
    n = len(nums)
    # For each x, store the set of indices (i,j) such that nums[i]*nums[j] % x == 0 
    # or x*nums[j] % nums[i] == 0 (for distinct i,j)
    # Actually, let's think differently: for a set S, x can be added if for all 
    # distinct a,b in S: (a*b) % x != 0 AND (x*b) % a != 0
    
    # Let's precompute for each x, the set of numbers that "conflict" with x in pairs
    # conflict_pairs[x] = set of y such that there exists z where {x,y,z} is a bad triple
    
    # Better approach: for each x, precompute which pairs (a,b) with a != b 
    # would make {x,a,b} invalid
    # {x,a,b} is invalid if: x|a*b OR a|x*b OR b|x*a
    
    # For efficiency, let's compute for each x, the list of (a,b) pairs that conflict
    conflict_map = {}  # x -> list of (a, b) pairs that conflict with x
    for x in nums:
        conflicting_pairs = []
        for i, a in enumerate(nums):
            if a == x:
                continue
            for j, b in enumerate(nums):
                if b == x or b == a:
                    continue
                # Check if {x, a, b} is a bad triple
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    conflicting_pairs.append((a, b))
        conflict_map[x] = conflicting_pairs
    
    return conflict_map

def _max_independent_set_bb(nums, current, remaining, best_size, best_set, deadline, conflict_map):
    """
    Branch and bound for maximum independent set in 3-uniform hypergraph.
    current: set of selected vertices
    remaining: list of candidate vertices not yet decided
    Uses upper bound: current_size + len(remaining)
    Branching heuristic: pick vertex with highest conflict degree in remaining
    """
    if time.time() > deadline:
        return best_size, best_set
    
    if not remaining:
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        return best_size, best_set
    
    # Upper bound
    if len(current) + len(remaining) <= best_size:
        return best_size, best_set
    
    # Pick branching vertex: one with highest "conflict count" among remaining
    # Conflict count for x in context of current: number of pairs in current that conflict with x
    # Plus number of pairs in remaining that conflict with x
    best_x = remaining[0]
    best_conf = -1
    for x in remaining:
        # Count conflicts with current set
        conf_count = 0
        cur_list = list(current)
        for i in range(len(cur_list)):
            a = cur_list[i]
            for j in range(i+1, len(cur_list)):
                b = cur_list[j]
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    conf_count += 1
        if conf_count > best_conf:
            best_conf = conf_count
            best_x = x
            if best_conf > 0:
                break  # Found a conflicting vertex, use it
    
    x = best_x
    new_remaining = [v for v in remaining if v != x]
    
    # Option 1: Add x if possible
    if _can_add(current, x):
        new_current = current | {x}
        best_size, best_set = _max_independent_set_bb(
            nums, new_current, new_remaining, best_size, best_set, deadline, conflict_map
        )
    
    # Option 2: Skip x
    best_size, best_set = _max_independent_set_bb(
        nums, current, new_remaining, best_size, best_set, deadline, conflict_map
    )
    
    return best_size, best_set

def _local_search(S, candidates, deadline, max_iterations=100):
    """Try to improve S by removing one element and adding others, with multiple restarts."""
    best = set(S)
    best_size = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try all removals
        improved = False
        for rem in list(best):
            if time.time() > deadline:
                break
            S2 = best - {rem}
            # Try to add candidates greedily
            for x in candidates:
                if x in S2:
                    continue
                if _can_add(S2, x):
                    S2.add(x)
            if len(S2) > best_size:
                best = set(S2)
                best_size = len(best)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 75  # 75 second budget (15s safety margin)
    
    n = P['n']
    nums = list(range(2, n + 1))
    
    # Start with primes (always a valid set)
    primes = [x for x in nums if _is_prime(x)]
    S = set(primes)
    
    # Try greedy with different orderings
    def _greedy_order(order_func):
        S_greedy = set()
        for x in order_func(nums):
            if _can_add(S_greedy, x):
                S_greedy.add(x)
        return S_greedy
    
    orderings = [
        lambda nums: sorted(nums, key=lambda x: -x),  # descending
        lambda nums: sorted(nums, key=lambda x: x),    # ascending
        lambda nums: sorted(nums, key=lambda x: (x, -x)),  # natural
        # New: sort by number of prime factors (fewer = more likely to be compatible)
        lambda nums: sorted(nums, key=lambda x: (x.bit_count(), x)),
    ]
    
    best_set = set(S)
    best_size = len(S)
    
    for order_func in orderings:
        S_greedy = _greedy_order(order_func)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
    
    # Local search from best
    candidates = [x for x in nums if x not in best_set]
    if time.time() < _DEADLINE:
        best_set = _local_search(best_set, candidates, _DEADLINE)
        best_size = len(best_set)
    
    # Branch and bound with remaining time
    if time.time() < _DEADLINE - 15:  # Leave 15s for verification and return
        # Precompute conflicts for faster branching
        # Use a reduced candidate set for B&B to keep it tractable
        # Focus on numbers that are likely to be in the optimal set
        def _conflict_count(x, S_ref):
            count = 0
            for a in S_ref:
                for b in S_ref:
                    if a == b:
                        continue
                    if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                        count += 1
            return count
        
        # Sort candidates by conflict count (ascending) - try less conflicting ones first
        candidate_list = [x for x in nums if x not in best_set]
        candidate_list.sort(key=lambda x: _conflict_count(x, best_set))
        
        # Limit B&B to a manageable subset
        bb_candidates = candidate_list[:40]
        
        # Reset best for B&B
        bb_best_size = best_size
        bb_best_set = set(best_set)
        
        # Try B&B starting from empty set with these candidates
        if time.time() < _DEADLINE - 25:
            # Precompute conflict map for the candidate set + best_set
            all_nums = list(set(bb_candidates) | best_set)
            
            # Simple B&B without precomputed conflicts (faster setup)
            def _bb_search(cands, current, b_size, b_set, dl):
                if time.time() > dl:
                    return b_size, b_set
                if not cands:
                    if len(current) > b_size:
                        b_size = len(current)
                        b_set = set(current)
                    return b_size, b_set
                if len(current) + len(cands) <= b_size:
                    return b_size, b_set
                
                # Pick vertex with highest conflict in current
                best_x = cands[0]
                best_conf = -1
                for x in cands:
                    conf = 0
                    cur_list = list(current)
                    for i in range(len(cur_list)):
                        a = cur_list[i]
                        for j in range(i+1, len(cur_list)):
                            b = cur_list[j]
                            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                                conf += 1
                    if conf > best_conf:
                        best_conf = conf
                        best_x = x
                        if conf > 0:
                            break
                
                x = best_x
                rest = [v for v in cands if v != x]
                
                # Try adding x
                if _can_add(current, x):
                    b_size, b_set = _bb_search(
                        rest, current | {x}, b_size, b_set, dl
                    )
                
                # Try skipping x
                b_size, b_set = _bb_search(
                    rest, current, b_size, b_set, dl
                )
                
                return b_size, b_set
            
            bb_best_size, bb_best_set = _bb_search(
                bb_candidates, set(), bb_best_size, bb_best_set, _DEADLINE
            )
            
            if bb_best_size > best_size:
                best_set = bb_best_set
                best_size = bb_best_size
    
    # Verify
    S_list = sorted(best_set)
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted(primes)
    
    return list(S_list)
# EVOLVE-BLOCK-END
