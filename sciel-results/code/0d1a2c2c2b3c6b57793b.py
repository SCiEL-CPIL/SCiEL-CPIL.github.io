import time
import sys

# Increase recursion limit if necessary, though we use iterative or shallow depth
sys.setrecursionlimit(2000)

_DEADLINE = 80.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_validity(sums):
    """
    Check if a set of sums satisfies the divisibility constraint.
    No two distinct values in sums divide one another.
    """
    if len(sums) <= 1:
        return True
    
    sum_list = list(sums)
    n = len(sum_list)
    
    # Optimization: sort sums. If a | b, then a <= b.
    # We can iterate and check multiples.
    # But checking all pairs is O(N^2).
    # For N ~ 1000, N^2 = 1e6. This is okay for a single check.
    # We do this check at every step of DFS.
    
    # A faster check:
    # For each s in sums, check if any multiple of s is in sums.
    # Max sum is bounded.
    max_sum = max(sum_list)
    
    for s in sum_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sums:
                return False
            multiple += s
            
    return True

def _add_element(sums, x):
    """
    Add element x to the set. Returns new set of sums.
    sums: set of current non-empty subset sums.
    x: new element.
    """
    new_sums = set(sums)
    new_sums.add(x)
    for s in sums:
        new_sums.add(s + x)
    return new_sums

def construct():
    global _START
    _START = time.time()
    
    # Candidates: 1 to 256
    # Sort descending to try large numbers first (less likely to divide others)
    candidates = list(range(256, 0, -1))
    
    best_set = []
    best_len = 0
    node_count = 0
    max_nodes = 50000  # Safety limit on search nodes
    
    # State: (current_sums, current_elements)
    # We use DFS
    
    def dfs(sums, elements, start_idx):
        nonlocal best_set, best_len, node_count
        
        if _remaining() < 5.0:
            return
        if node_count > max_nodes:
            return
            
        node_count += 1
        
        current_len = len(elements)
        if current_len > best_len:
            best_len = current_len
            best_set = list(elements)
            
        # Try adding more elements
        # Start from start_idx to maintain ordering (symmetry breaking)
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Quick check: if x is too small, it might divide existing sums?
            # Actually, x itself is a sum. If x divides any s in sums, invalid.
            # If any s in sums divides x, invalid.
            
            # Check if x is compatible with current sums
            # 1. x must not divide any s in sums
            # 2. No s in sums must divide x
            
            compatible = True
            for s in sums:
                if x % s == 0 or s % x == 0:
                    compatible = False
                    break
            
            if not compatible:
                continue
                
            # Generate new sums
            new_sums = _add_element(sums, x)
            
            # Check divisibility among new sums
            # We only need to check interactions involving new sums
            # But _check_validity checks all. It's safer and fast enough for small N.
            if _check_validity(new_sums):
                # Recurse
                elements.append(x)
                dfs(new_sums, elements, i + 1)
                elements.pop()
                
                if _remaining() < 5.0:
                    return

    # Initial call
    dfs(set(), [], 0)
    
    # Fallback if DFS didn't find anything or timed out early
    if not best_set:
        # Greedy fallback
        current_sums = set()
        current_elements = []
        for x in candidates:
            new_sums = _add_element(current_sums, x)
            if _check_validity(new_sums):
                current_sums = new_sums
                current_elements.append(x)
        best_set = current_elements
        best_len = len(best_set)
        
    return [int(x) for x in best_set]
