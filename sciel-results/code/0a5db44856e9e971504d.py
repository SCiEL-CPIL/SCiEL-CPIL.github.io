# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations as _itertools_permutations

def construct(n):
    """
    Construct a superpermutation for n symbols.
    Returns a string of digits 1..n such that every permutation appears as a substring of length n.
    """
    deadline = time.time() + 115  # safety margin before 120s kill
    start_time = time.time()
    
    # For small n, use known optimal constructions
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return _standard_construction(4)
    if n == 5:
        return _standard_construction(5)
    if n == 6:
        return _standard_construction(6)
    
    # For n >= 7: build up to n-1 using standard construction,
    # then optimize the final stage with local search
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        word = _expand_step(word, m)
    
    # Final stage: add symbol n
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    if not perms:
        return "".join(map(str, word))
    
    # Precompute overlap costs between all pairs of permutations
    # Cost of going from perm i to perm j = block_len - overlap(i, j)
    # block_len = 2*(m-1) + 1 = 2*m - 1
    block_len = 2 * m - 1
    k_perms = len(perms)
    
    # Build index map for speed
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap as a 2D list
    # overlap_matrix[i][j]: max k in 0..m-1 where perms[i][-k:] == perms[j][:k]
    overlap_matrix = [[0] * k_perms for _ in range(k_perms)]
    for i in range(k_perms):
        pi = perms[i]
        for j in range(k_perms):
            if i == j:
                overlap_matrix[i][j] = block_len
                continue
            pj = perms[j]
            # Find max overlap: suffix of pi matches prefix of pj
            best_k = 0
            for k in range(m - 1, 0, -1):
                if pi[-k:] == pj[:k]:
                    best_k = k
                    break
            overlap_matrix[i][j] = best_k
    
    # Greedy nearest-neighbor initialization for better starting point
    if n <= 7 and time.time() < deadline - 5:
        order_indices = _greedy_order(k_perms, overlap_matrix, block_len, deadline)
    else:
        order_indices = list(range(k_perms))
    
    def compute_length_fast(ord_indices):
        if len(ord_indices) <= 1:
            return block_len
        total = block_len
        for t in range(len(ord_indices) - 1):
            total += block_len - overlap_matrix[ord_indices[t]][ord_indices[t + 1]]
        return total
    
    best_len = compute_length_fast(order_indices)
    best_order_indices = list(order_indices)
    
    # Local search with adjacent swaps and 2-opt moves
    if n <= 7 and time.time() < deadline - 15:
        improved = True
        max_rounds = 1000
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 15:
                break
            improved = False
            
            # Adjacent swaps
            for i in range(len(best_order_indices) - 1):
                if time.time() > deadline - 15:
                    break
                a = best_order_indices[i]
                b = best_order_indices[i + 1]
                # Compute delta for swapping i and i+1
                # Before: ... X, a, b, Y ...
                # After:  ... X, b, a, Y ...
                # Delta = (cost[b][a] - cost[a][b]) + effects on neighbors
                old_cost = 0
                new_cost = 0
                # Edge from left neighbor to a, and a to b, and b to right neighbor
                if i > 0:
                    left = best_order_indices[i - 1]
                    old_cost += block_len - overlap_matrix[left][a]
                    new_cost += block_len - overlap_matrix[left][b]
                old_cost += block_len - overlap_matrix[a][b]
                new_cost += block_len - overlap_matrix[b][a]
                if i + 2 < len(best_order_indices):
                    right = best_order_indices[i + 2]
                    old_cost += block_len - overlap_matrix[b][right]
                    new_cost += block_len - overlap_matrix[a][right]
                
                if new_cost < old_cost:
                    best_order_indices[i] = b
                    best_order_indices[i + 1] = a
                    best_len += new_cost - old_cost
                    improved = True
            
            # 2-opt: reverse a segment [i, j]
            if time.time() > deadline - 15:
                break
            seg_len_limit = min(30, len(best_order_indices))
            for i in range(len(best_order_indices) - 1):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, min(i + seg_len_limit, len(best_order_indices))):
                    if time.time() > deadline - 15:
                        break
                    # Reversing segment [i..j]
                    # Old edges: (i-1 -> i), (i -> i+1), ..., (j -> j+1)
                    # New edges: (i-1 -> j), (j -> j-1), ..., (i -> i+1)
                    
                    old_cost = 0
                    new_cost = 0
                    if i > 0:
                        left = best_order_indices[i - 1]
                        old_cost += block_len - overlap_matrix[left][best_order_indices[i]]
                        new_cost += block_len - overlap_matrix[left][best_order_indices[j]]
                    if j + 1 < len(best_order_indices):
                        right = best_order_indices[j + 1]
                        old_cost += block_len - overlap_matrix[best_order_indices[j]][right]
                        new_cost += block_len - overlap_matrix[best_order_indices[i]][right]
                    # Internal edges
                    for t in range(i, j):
                        old_cost += block_len - overlap_matrix[best_order_indices[t]][best_order_indices[t + 1]]
                        new_cost += block_len - overlap_matrix[best_order_indices[t + 1]][best_order_indices[t]]
                    
                    if new_cost < old_cost:
                        # Reverse segment [i..j]
                        best_order_indices[i:j + 1] = best_order_indices[i:j + 1][::-1]
                        best_len += new_cost - old_cost
                        improved = True
                        break  # Restart inner loop after a move
        
            # Also try random restarts with perturbation
            if not improved and time.time() < deadline - 20:
                # Perturb the current best and do a quick local search
                perturbed = list(best_order_indices)
                # Randomly swap a few elements
                for _ in range(min(10, len(perturbed))):
                    i, j = random.sample(range(len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                
                perturbed_len = compute_length_fast(perturbed)
                # Quick local search on perturbed
                for _ in range(50):
                    if time.time() > deadline - 15:
                        break
                    imp = False
                    for i in range(len(perturbed) - 1):
                        a = perturbed[i]
                        b = perturbed[i + 1]
                        old_cost = 0
                        new_cost = 0
                        if i > 0:
                            left = perturbed[i - 1]
                            old_cost += block_len - overlap_matrix[left][a]
                            new_cost += block_len - overlap_matrix[left][b]
                        old_cost += block_len - overlap_matrix[a][b]
                        new_cost += block_len - overlap_matrix[b][a]
                        if i + 2 < len(perturbed):
                            right = perturbed[i + 2]
                            old_cost += block_len - overlap_matrix[b][right]
                            new_cost += block_len - overlap_matrix[a][right]
                        
                        if new_cost < old_cost:
                            perturbed[i] = b
                            perturbed[i + 1] = a
                            perturbed_len += new_cost - old_cost
                            imp = True
                    if not imp:
                        break
                
                if perturbed_len < best_len:
                    best_order_indices = perturbed
                    best_len = perturbed_len
        
        # Build final word
        final_order = [perms[idx] for idx in best_order_indices]
        return "".join(map(str, _build_final_stage(final_order, m)))
    else:
        final_order = [perms[idx] for idx in order_indices]
        return "".join(map(str, _build_final_stage(final_order, m)))


def _greedy_order(k, overlap_matrix, block_len, deadline):
    """Greedy nearest-neighbor: start from first perm, always go to nearest unvisited."""
    visited = [False] * k
    order_indices = [0]
    visited[0] = True
    current = 0
    
    for _ in range(k - 1):
        if time.time() > deadline - 10:
            break
        best_next = -1
        best_cost = float('inf')
        for j in range(k):
            if not visited[j]:
                cost = block_len - overlap_matrix[current][j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
        if best_next == -1:
            break
        order_indices.append(best_next)
        visited[best_next] = True
        current = best_next
    
    # Fill in any unvisited (shouldn't happen)
    for j in range(k):
        if not visited[j]:
            order_indices.append(j)
    
    return order_indices


def _standard_construction(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        word = _expand_step(word, m)
    return "".join(map(str, word))


def _expand_step(word, m):
    """Expand word from level m-1 to level m using standard construction."""
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        overlap = _find_overlap(new, block)
        new += block[overlap:]
    return new


def _build_final_stage(order, m):
    """Build the final stage word given an ordering of (m-1)-permutations."""
    new = []
    for p in order:
        block = list(p) + [m] + list(p)
        overlap = _find_overlap(new, block)
        new += block[overlap:]
    return new


def _find_overlap(current, block):
    """Find the maximum overlap between end of current and start of block."""
    if not current:
        return 0
    max_overlap = min(len(current), len(block))
    for o in range(max_overlap, -1, -1):
        if o == 0:
            return 0
        if current[len(current) - o:] == block[:o]:
            return o
    return 0
# EVOLVE-BLOCK-END
