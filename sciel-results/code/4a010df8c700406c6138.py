# EVOLVE-BLOCK-START
import time
import sys
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
    
    # Local search with adjacent swaps, 2-opt, and perturbation
    if n <= 7 and time.time() < deadline - 15:
        import random
        random.seed(42)  # Deterministic for reproducibility
        best_len_so_far = best_len
        best_order_so_far = list(best_order_indices)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 15 and no_improve_count < max_no_improve:
            improved = False
            
            # Pass 1: Adjacent swaps (multiple passes until no improvement)
            for _ in range(3):  # Multiple passes
                if time.time() > deadline - 15:
                    break
                pass_improved = False
                for i in range(len(best_order_indices) - 1):
                    a = best_order_indices[i]
                    b = best_order_indices[i + 1]
                    old_cost = 0
                    new_cost = 0
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
                        pass_improved = True
                        improved = True
            
            # Pass 2: 2-opt with larger segments
            if time.time() < deadline - 15:
                seg_len_limit = min(50, len(best_order_indices))
                for i in range(len(best_order_indices) - 1):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i + 1, min(i + seg_len_limit, len(best_order_indices))):
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
                        for t in range(i, j):
                            old_cost += block_len - overlap_matrix[best_order_indices[t]][best_order_indices[t + 1]]
                            new_cost += block_len - overlap_matrix[best_order_indices[t + 1]][best_order_indices[t]]
                        
                        if new_cost < old_cost:
                            best_order_indices[i:j + 1] = best_order_indices[i:j + 1][::-1]
                            best_len += new_cost - old_cost
                            improved = True
                            break  # Restart after 2-opt move
            
            # Pass 3: Or-opt (move a segment of length 1-3 to another position)
            if time.time() < deadline - 15:
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline - 15:
                        break
                    for i in range(len(best_order_indices) - seg_len):
                        if time.time() > deadline - 15:
                            break
                        for j in range(len(best_order_indices) - seg_len):
                            if j == i or (j > i and j < i + seg_len):
                                continue
                            # Remove segment [i, i+seg_len) and insert at position j
                            # Adjust j if i < j
                            insert_pos = j
                            if i < j:
                                insert_pos = j - seg_len
                            
                            # Compute cost change
                            old_cost = 0
                            new_cost = 0
                            seg = best_order_indices[i:i + seg_len]
                            
                            # Old edges
                            if i > 0:
                                old_cost += block_len - overlap_matrix[best_order_indices[i - 1]][best_order_indices[i]]
                            if i + seg_len < len(best_order_indices):
                                old_cost += block_len - overlap_matrix[best_order_indices[i + seg_len - 1]][best_order_indices[i + seg_len]]
                            for t in range(i, i + seg_len - 1):
                                old_cost += block_len - overlap_matrix[best_order_indices[t]][best_order_indices[t + 1]]
                            
                            # New edges (after removal and insertion)
                            # This is complex; skip for now and rely on 2-opt
                            pass
            
            # Check for improvement
            if improved:
                no_improve_count = 0
                if best_len < best_len_so_far:
                    best_len_so_far = best_len
                    best_order_so_far = list(best_order_indices)
            else:
                no_improve_count += 1
                
                # Perturbation: random 2-opt move to escape local optimum
                if no_improve_count % 10 == 0 and time.time() < deadline - 15:
                    i = random.randint(0, len(best_order_indices) - 2)
                    j = random.randint(i + 1, min(i + 30, len(best_order_indices) - 1))
                    best_order_indices[i:j + 1] = best_order_indices[i:j + 1][::-1]
                    # Recompute length
                    best_len = compute_length_fast(best_order_indices)
        
        # Use best found
        best_order_indices = best_order_so_far
        best_len = best_len_so_far
        
        # Build final word from best order found
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
