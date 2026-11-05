"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    # Precompute gcd table
    gcd_table = [[1]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd_table[j][i] = gcd(i, j)
    
    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]
    
    def count_conflicts(x, selected):
        """Count how many pairs in selected would form a bad triple with x"""
        count = 0
        n = len(selected)
        for i in range(n):
            for j in range(i+1, n):
                if is_bad_triple(x, selected[i], selected[j]):
                    count += 1
        return count
    
    # Greedy initialization: add numbers in order of fewest potential conflicts
    # First, compute for each number how many pairs it conflicts with
    conflict_score = [0] * (N+1)
    for x in range(1, N+1):
        for a in range(1, N+1):
            if a == x:
                continue
            for b in range(a+1, N+1):
                if b == x:
                    continue
                if is_bad_triple(x, a, b):
                    conflict_score[x] += 1
    
    # Sort by conflict score (fewest conflicts first)
    candidates = list(range(1, N+1))
    candidates.sort(key=lambda x: conflict_score[x])
    
    selected = []
    for x in candidates:
        if time.time() > deadline:
            break
        if count_conflicts(x, selected) == 0:
            selected.append(x)
    
    # Local search: try to swap out elements to add more
    best_selected = list(selected)
    best_size = len(selected)
    
    # Try adding elements not in selected by removing conflicting ones
    not_selected = [x for x in range(1, N+1) if x not in selected]
    
    for x in not_selected:
        if time.time() > deadline:
            break
        # Find which pairs in selected conflict with x
        conflicting_pairs = []
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if is_bad_triple(x, selected[i], selected[j]):
                    conflicting_pairs.append((i, j))
        
        if not conflicting_pairs:
            selected.append(x)
            best_selected = list(selected)
            best_size = len(selected)
            continue
        
        # Try to remove one element from each conflicting pair to make room
        # Greedy: remove the element that appears in the most conflicting pairs
        removal_count = [0] * len(selected)
        for i, j in conflicting_pairs:
            removal_count[i] += 1
            removal_count[j] += 1
        
        # Find the element to remove that resolves the most conflicts
        # Try removing each candidate and see if we can add x
        for remove_idx in range(len(selected)):
            if removal_count[remove_idx] == 0:
                continue
            new_selected = [s for k, s in enumerate(selected) if k != remove_idx]
            if count_conflicts(x, new_selected) == 0:
                selected = new_selected + [x]
                if len(selected) > best_size:
                    best_selected = list(selected)
                    best_size = len(selected)
                break
    
    # Multiple rounds of local search
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        not_selected = [x for x in range(1, N+1) if x not in selected]
        for x in not_selected:
            if time.time() > deadline:
                break
            conflicting_pairs = []
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if is_bad_triple(x, selected[i], selected[j]):
                        conflicting_pairs.append((i, j))
            if not conflicting_pairs:
                selected.append(x)
                improved = True
                if len(selected) > best_size:
                    best_selected = list(selected)
                    best_size = len(selected)
                continue
            for remove_idx in range(len(selected)):
                new_selected = [s for k, s in enumerate(selected) if k != remove_idx]
                if count_conflicts(x, new_selected) == 0:
                    selected = new_selected + [x]
                    improved = True
                    if len(selected) > best_size:
                        best_selected = list(selected)
                        best_size = len(selected)
                    break
        if not improved:
            break
    
    return best_selected if best_selected else selected
# EVOLVE-BLOCK-END
