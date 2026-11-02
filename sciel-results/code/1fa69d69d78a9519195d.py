import time
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    
    # 1. Standard recursive construction (Graham-Lagarias)
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    if n <= 6:
        return "".join(map(str, word))
    
    # 2. Optimization for n >= 7
    # The word is constructed from blocks P_i n P_i.
    # The order of P_i determines the overlaps.
    # We want to maximize the total overlap.
    
    # Extract the permutations of 1..n-1 used in the construction.
    # They are the ones that appeared in the previous step.
    # For n=7, these are perms of 1..6.
    
    perms_n_minus_1 = list(permutations(range(1, n)))
    num_p = len(perms_n_minus_1)
    
    # Precompute overlap matrix.
    # overlap[i][j] = max k such that perms_n_minus_1[i][-k:] == perms_n_minus_1[j][:k]
    # k can be 0..n-2.
    overlap = [[0]*num_p for _ in range(num_p)]
    
    for i in range(num_p):
        pi = perms_n_minus_1[i]
        for j in range(num_p):
            pj = perms_n_minus_1[j]
            max_k = n - 2
            for k in range(max_k, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Initial order: standard recursive order.
    # We need to map the standard order to indices in perms_n_minus_1.
    # The standard order is the order in which they appeared in the n-1 superperm.
    # Let's re-derive it or just use a heuristic start.
    # Actually, we can just extract the order from the current 'word' or 'perms' list.
    # In the loop above, `perms` held the order for the *next* step.
    # So for n=7, `perms` in the last iteration (m=7) was the list of perms of 1..6.
    # But we overwrote it. Let's just use the natural order of itertools or random start?
    # The natural order of itertools is lexicographic. The standard construction uses a specific order.
    # However, any valid Hamiltonian path is a valid superpermutation.
    # The length depends on the overlaps.
    
    # Let's start with the lexicographic order (0..N-1) and optimize.
    # Or better, start with the order that minimizes cost greedily (nearest neighbor) to get a good start.
    
    def get_order_cost(order):
        cost = 0
        for i in range(len(order) - 1):
            cost += overlap[order[i]][order[i+1]]
        return cost
    
    # Nearest Neighbor Heuristic for initial solution
    used = [False] * num_p
    order = [0]
    used[0] = True
    for _ in range(num_p - 1):
        last = order[-1]
        best_next = -1
        best_score = -1
        for j in range(num_p):
            if not used[j]:
                score = overlap[last][j]
                if score > best_score:
                    best_score = score
                    best_next = j
        if best_next == -1:
            break
        order.append(best_next)
        used[best_next] = True
    
    # If NN didn't find a full path (unlikely for this graph), fallback to 0..N-1
    if len(order) < num_p:
        order = list(range(num_p))
    
    best_order = order[:]
    best_score = get_order_cost(order)
    
    # 2-Opt Local Search
    # We want to maximize overlap (minimize length).
    # Length = sum(len(block)) - sum(overlap)
    # len(block) is constant (2*(n-1)+1 = 2n-1).
    # So maximizing overlap minimizes length.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_p - 1):
            if time.time() > deadline:
                break
            for j in range(i+1, num_p):
                # Reverse segment i..j
                # Old links: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # New links: (i-1, j), (j, j-1), (i, i+1), (j+1, i)
                # Wait, 2-opt reverses the path between i and j.
                # Path: ... A, v_i, ..., v_j, B ...
                # New: ... A, v_j, ..., v_i, B ...
                
                # Affected edges:
                # 1. (A, v_i) -> (A, v_j)
                # 2. (v_i, v_{i+1}) ... (v_{j-1}, v_j) -> (v_j, v_{j-1}) ... (v_{i+1}, v_i)
                #    The internal edges are reversed, but since overlap is not symmetric,
                #    overlap[u][v] != overlap[v][u].
                #    So we must recalculate all internal edges.
                #    This makes 2-opt O(N) per move, total O(N^3).
                #    For N=720, 720^3 = 373M. Too slow in Python.
                
                # Alternative: Just try swapping adjacent elements?
                # Swap i, i+1.
                # Affected edges: (i-1, i), (i, i+1), (i+1, i+2).
                # This is O(1) per move.
                # 3-Opt or simple swap local search.
                
                # Let's try simple adjacent swap local search.
                pass
                
        # If we stick to adjacent swaps, it's O(N^2) per pass.
        # 720^2 = 518k. Feasible in Python if optimized.
        
    # Let's implement adjacent swap local search properly.
    order = best_order[:]
    best_score = get_order_cost(order)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_p - 1):
            if time.time() > deadline:
                break
            # Try swap i and i+1
            a = order[i]
            b = order[i+1]
            
            old_score = 0
            new_score = 0
            
            # Link before: i-1 -> i
            if i > 0:
                old_score += overlap[order[i-1]][a]
                new_score += overlap[order[i-1]][b]
            
            # Link middle: i -> i+1
            old_score += overlap[a][b]
            new_score += overlap[b][a]
            
            # Link after: i+1 -> i+2
            if i < num_p - 2:
                old_score += overlap[b][order[i+2]]
                new_score += overlap[a][order[i+2]]
            
            if new_score > old_score:
                # Perform swap
                order[i] = b
                order[i+1] = a
                best_score += (new_score - old_score)
                improved = True
                # Move back to check neighbors of new i?
                # i = max(0, i-1) is a common trick to avoid missing things.
                if i > 0:
                    i -= 1
    
    # Construct word from optimized order
    new_word = []
    for idx in order:
        p = perms_n_minus_1[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(new_word), len(block))
        ov = 0
        for o in range(max_overlap, -1, -1):
            if new_word[len(new_word) - o:] == block[:o]:
                ov = o
                break
        new_word += block[ov:]
    
    if len(new_word) < len(word):
        word = new_word
    
    return "".join(map(str, word))
