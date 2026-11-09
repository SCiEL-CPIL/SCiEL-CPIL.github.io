import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 256
    
    # Precompute squares up to 2*N = 512
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Diagonal condition: 2a must not be a square
    # 2a in squares means a is in {2,8,18,32,50,72,98,128,162,200,242}
    invalid_diagonal = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            invalid_diagonal.add(a)
    
    # Build conflict graph: edge between a,b (a<b) if a+b is a square
    # Only among valid nodes (not in invalid_diagonal)
    valid_nodes = [v for v in range(1, N + 1) if v not in invalid_diagonal]
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Conflict adjacency list
    conflicts = [set() for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Greedy: pick nodes with lowest degree first
    def greedy_initial():
        remaining = set(range(M))
        selected = set()
        while remaining:
            # Pick node with minimum degree among remaining
            min_deg = float('inf')
            best = None
            for node in remaining:
                deg = len(conflicts[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best = node
            selected.add(best)
            remaining.discard(best)
            # Remove all neighbors of best from remaining
            for neighbor in conflicts[best]:
                remaining.discard(neighbor)
        return selected
    
    # Local search: try to add nodes, or swap
    def local_search(selected, max_iters=50000):
        selected = set(selected)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 2:
                break
            # Try adding a node not in selected
            candidates = [n for n in range(M) if n not in selected]
            random.shuffle(candidates)
            for n in candidates:
                if all(nb in selected or True for nb in conflicts[n]) and not any(nb in selected for nb in conflicts[n]):
                    selected.add(n)
                    improved = True
                    break
            # Try swapping: remove one, add another
            if not improved:
                sel_list = list(selected)
                random.shuffle(sel_list)
                for rem in sel_list:
                    if time.time() > deadline - 2:
                        break
                    candidates = [n for n in range(M) if n not in selected and n != rem]
                    random.shuffle(candidates)
                    for add in candidates:
                        # Check if add is compatible with (selected - {rem})
                        ok = True
                        for nb in conflicts[add]:
                            if nb in selected and nb != rem:
                                ok = False
                                break
                        if ok:
                            selected.discard(rem)
                            selected.add(add)
                            improved = True
                            break
                    if improved:
                        break
        return selected
    
    # Get initial solution
    best = greedy_initial()
    best_size = len(best)
    
    # Try multiple random restarts with different greedy orders
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 5:
        # Randomized greedy
        remaining = set(range(M))
        selected = set()
        while remaining:
            # Pick with some randomness among low-degree nodes
            min_deg = float('inf')
            for node in remaining:
                deg = len(conflicts[node] & remaining)
                min_deg = min(min_deg, deg)
            # Pick randomly from nodes with degree <= min_deg + 2
            low_deg_nodes = [n for n in remaining if len(conflicts[n] & remaining) <= min_deg + 2]
            best = random.choice(low_deg_nodes)
            selected.add(best)
            remaining.discard(best)
            for neighbor in conflicts[best]:
                remaining.discard(neighbor)
        
        # Local search
        selected = local_search(selected, max_iters=20000)
        
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
        
        restarts += 1
        if time.time() > deadline - 5:
            break
    
    # Final local search on best
    best = local_search(best, max_iters=30000)
    
    # Convert back to values
    result = [valid_nodes[i] for i in sorted(best)]
    
    # Verify
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = int(math.isqrt(s))
            if r * r == s:
                # Should not happen, but just in case
                pass
    
    return result
