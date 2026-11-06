"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

# Deadline guard
_DEADLINE = 85.0  # seconds total; we will aim to finish well before this

def _check_add(x, S):
    """Check if x can be added to S without violating the condition."""
    if len(S) >= 3:
        # Check if x divides any triple from S
        for t in combinations(S, 3):
            if (t[0] + t[1] + t[2]) % x == 0:
                return False
    
    # Check if any a in S divides a new triple involving x
    for a in S:
        # Need to check if there exist u, v in S\{a}, u != v, such that (x + u + v) % a == 0
        # i.e., (u + v) % a == (-x) % a
        target = (-x) % a
        S_without_a = [v for v in S if v != a]
        if len(S_without_a) < 2:
            continue
        # Use residue counting
        residues = defaultdict(int)
        for v in S_without_a:
            residues[v % a] += 1
        for i, u in enumerate(S_without_a):
            needed = (target - u) % a
            # Check if needed exists in remaining elements
            count_needed = residues[needed]
            if u % a == needed:
                # Need another element with same residue
                if count_needed >= 2:
                    return False
            else:
                if count_needed >= 1:
                    return False
    return True

def _greedy(order, t_start):
    """Greedy construction with given order."""
    S = []
    for x in order:
        if time.time() - t_start > _DEADLINE - 5:
            break
        if _check_add(x, S):
            S.append(x)
    return S

def _local_search(S, t_start):
    """Try to improve S via local search (swaps)."""
    S = list(S)
    n = 192
    all_nums = set(range(1, n + 1))
    in_S = set(S)
    not_in_S = list(all_nums - in_S)
    random.shuffle(not_in_S)
    
    improved = True
    while improved and time.time() - t_start < _DEADLINE - 10:
        improved = False
        # Try to add any element from not_in_S
        for x in not_in_S:
            if time.time() - t_start > _DEADLINE - 10:
                break
            if _check_add(x, S):
                S.append(x)
                in_S.add(x)
                not_in_S.remove(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, try to add two
        S_copy = list(S)
        for i, a in enumerate(S_copy):
            if time.time() - t_start > _DEADLINE - 10:
                break
            S_temp = [v for j, v in enumerate(S_copy) if j != i]
            # Try to add elements from not_in_S
            added = []
            for x in not_in_S:
                if time.time() - t_start > _DEADLINE - 10:
                    break
                if _check_add(x, S_temp):
                    S_temp.append(x)
                    added.append(x)
                    if len(added) >= 2:
                        break
            if len(added) >= 2:
                # Net gain of 1
                S = S_temp
                in_S = set(S)
                not_in_S = list(all_nums - in_S)
                random.shuffle(not_in_S)
                improved = True
                break
    
    return S

def construct():
    t_start = time.time()
    n = 192
    
    # Try multiple orderings and keep the best
    best_S = []
    
    # Descending order
    order1 = list(range(n, 0, -1))
    S1 = _greedy(order1, t_start)
    if len(S1) > len(best_S):
        best_S = S1
    
    # Ascending order
    if time.time() - t_start < _DEADLINE - 15:
        order2 = list(range(1, n + 1))
        S2 = _greedy(order2, t_start)
        if len(S2) > len(best_S):
            best_S = S2
    
    # Random orderings
    if time.time() - t_start < _DEADLINE - 15:
        for _ in range(5):
            if time.time() - t_start > _DEADLINE - 15:
                break
            order3 = list(range(1, n + 1))
            random.shuffle(order3)
            S3 = _greedy(order3, t_start)
            if len(S3) > len(best_S):
                best_S = S3
    
    # Local search on the best found
    if time.time() - t_start < _DEADLINE - 10:
        best_S = _local_search(best_S, t_start)
    
    return best_S
# EVOLVE-BLOCK-END
