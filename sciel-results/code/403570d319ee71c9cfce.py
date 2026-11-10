"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    N = 256
    # Perfect squares up to 512
    squares = set()
    for k in range(2, 23):  # 2^2=4 to 22^2=484
        sq = k * k
        if sq <= 512:
            squares.add(sq)
    
    # Diagonal constraint: 2a must not be a perfect square
    # a must not be 2*k^2
    diagonal_bad = set()
    for k in range(1, 16):
        val = 2 * k * k
        if val <= N:
            diagonal_bad.add(val)
    
    # Build adjacency: for each a, which b > a conflict with a (a+b is square)
    # Also self-conflict if a in diagonal_bad
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        if a in diagonal_bad:
            adj[a].add(a)  # self-conflict
        for b in range(a + 1, N + 1):
            if (a + b) in squares:
                adj[a].add(b)
                adj[b].add(a)
    
    # Greedy initial solution: process in order, add if no conflict with already selected
    selected = set()
    # Try multiple orderings
    best_set = set()
    best_size = 0
    
    # Ordering 1: ascending
    for a in range(1, N + 1):
        if a in diagonal_bad:
            continue
        conflict = False
        for b in adj[a]:
            if b in selected:
                conflict = True
                break
        if not conflict:
            selected.add(a)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
    
    # Ordering 2: descending
    selected = set()
    for a in range(N, 0, -1):
        if a in diagonal_bad:
            continue
        conflict = False
        for b in adj[a]:
            if b in selected:
                conflict = True
                break
        if not conflict:
            selected.add(a)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
    
    # Local search: try to add, remove, swap elements
    current = set(best_set)
    current_size = best_size
    
    def is_valid(s):
        """Check if set s is a valid independent set."""
        for a in s:
            if a in diagonal_bad:
                return False
            for b in s:
                if a < b and (a + b) in squares:
                    return False
        return True
    
    # Local search with add/remove/swap
    random.seed(42)
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        
        # Try adding each unselected element
        for a in range(1, N + 1):
            if a in current or a in diagonal_bad:
                continue
            # Check if a can be added
            can_add = True
            for b in adj[a]:
                if b in current:
                    can_add = False
                    break
            if can_add:
                current.add(a)
                current_size += 1
                improved = True
        
        # Try removing elements that allow more additions (swap-like)
        # For each element in current, try removing it and see if we can add more
        for a in list(current):
            current.discard(a)
            temp_size = current_size - 1
            
            # Try to greedily add back
            for b in range(1, N + 1):
                if b in current or b in diagonal_bad:
                    continue
                can_add = True
                for c in adj[b]:
                    if c in current:
                        can_add = False
                        break
                if can_add:
                    current.add(b)
                    temp_size += 1
            
            if temp_size > current_size:
                current_size = temp_size
                improved = True
            else:
                # Revert: put a back, remove newly added
                # Actually this is complex, let's just keep the better state
                pass
        
        # Additional: random restarts with different greedy orders
        if not improved and (time.time() - start_time) < deadline - 10:
            for _ in range(20):
                if (time.time() - start_time) > deadline - 5:
                    break
                order = list(range(1, N + 1))
                random.shuffle(order)
                selected = set()
                for a in order:
                    if a in diagonal_bad:
                        continue
                    conflict = False
                    for b in adj[a]:
                        if b in selected:
                            conflict = True
                            break
                    if not conflict:
                        selected.add(a)
                if len(selected) > current_size:
                    current = set(selected)
                    current_size = len(selected)
                    improved = True
    
    # Final check and return
    result = sorted(current)
    if not result:
        # Fallback: just pick valid elements
        result = [a for a in range(1, N + 1) if a not in diagonal_bad][:1]
    
    return result
# EVOLVE-BLOCK-END
