import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Build superpermutation for n-1 using standard construction
    def build_standard(m):
        word = [1]
        for k in range(2, m + 1):
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    word_np1 = build_standard(n - 1)
    
    # Step 2: Extract the sequence of (n-1)! permutations from word_np1
    perms = []
    seen = set()
    for i in range(len(word_np1) - (n - 1) + 1):
        window = tuple(word_np1[i:i + (n - 1)])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    num_perms = len(perms)  # Should be (n-1)!
    
    # Step 3: Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = n - 2
    overlap = np.zeros((num_perms, num_perms), dtype=np.int32)
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap[i][j] = 0
                    break
                if pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Step 4: Greedy maximum-overlap initial ordering
    # Start from the first permutation, then repeatedly pick the unvisited
    # permutation with the maximum overlap with the current last permutation.
    def greedy_order():
        unvisited = set(range(num_perms))
        order = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            best_next = -1
            best_ov = -1
            for v in unvisited:
                ov = overlap[current, v]
                if ov > best_ov:
                    best_ov = ov
                    best_next = v
            order.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        return order
    
    # Also try a few random restarts of greedy
    def greedy_order_random_start():
        unvisited = set(range(num_perms))
        start = random.randint(0, num_perms - 1)
        order = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            best_next = -1
            best_ov = -1
            for v in unvisited:
                ov = overlap[current, v]
                if ov > best_ov:
                    best_ov = ov
                    best_next = v
            order.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        return order
    
    def calc_length(order):
        # First block: 2*(n-1) + 1
        total = 2 * (n - 1) + 1
        for i in range(len(order) - 1):
            u = order[i]
            v = order[i + 1]
            total += (2 * (n - 1) + 1) - overlap[u, v]
        return total
    
    # Get initial order from greedy
    best_order = greedy_order()
    best_len = calc_length(best_order)
    
    # Try a few random greedy restarts
    for _ in range(5):
        if time.time() > deadline - 20.0:
            break
        candidate = greedy_order_random_start()
        cand_len = calc_length(candidate)
        if cand_len < best_len:
            best_len = cand_len
            best_order = candidate
    
    # Step 5: Local search with incremental updates
    # Precompute block_len for each permutation (all same: 2*(n-1)+1)
    block_len = 2 * (n - 1) + 1
    
    def calc_length_from_order(order):
        total = block_len
        for i in range(len(order) - 1):
            total += block_len - overlap[order[i], order[i + 1]]
        return total
    
    # Adjacent swap local search with incremental delta
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline - 15.0:
                break
            # Try swap i, i+1
            # Delta = (block_len - overlap[order[i+1], order[i]]) + (block_len - overlap[order[i-1] if i>0 else -1, ...])
            # More carefully: removing edges (i-1,i) and (i,i+1), adding (i-1,i+1) and (i+1,i)
            u = best_order[i]
            v = best_order[i + 1]
            # Current contribution of edges around i, i+1:
            # edge (i-1, i) if i > 0, edge (i, i+1), edge (i+1, i+2) if i+1 < num_perms-1
            # After swap: edge (i-1, i+1) if i > 0, edge (i+1, i), edge (i, i+2) if i+1 < num_perms-1
            
            delta = 0
            if i > 0:
                prev = best_order[i - 1]
                delta += (block_len - overlap[prev, v]) - (block_len - overlap[prev, u])
            if i + 1 < num_perms - 1:
                nxt = best_order[i + 2]
                delta += (block_len - overlap[u, nxt]) - (block_len - overlap[v, nxt])
            # Edge between u and v changes direction
            delta += (block_len - overlap[v, u]) - (block_len - overlap[u, v])
            
            if delta < 0:
                best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
                best_len += delta
                improved = True
    
    # 2-opt: reverse a segment [i, j]
    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline - 8.0:
                break
            for j in range(i + 2, num_perms):
                if time.time() > deadline - 8.0:
                    break
                # Reversing [i, j]: edges affected are (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # After reversal: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                delta = 0
                if i > 0:
                    prev = best_order[i - 1]
                    delta += (block_len - overlap[prev, best_order[j]]) - (block_len - overlap[prev, best_order[i]])
                if j < num_perms - 1:
                    nxt = best_order[j + 1]
                    delta += (block_len - overlap[best_order[i], nxt]) - (block_len - overlap[best_order[j], nxt])
                # Internal edges: (i,i+1), (i+1,i+2), ..., (j-1,j) become (j,j-1), (j-1,j-2), ..., (i+1,i)
                for k in range(i, j):
                    delta += (block_len - overlap[best_order[k + 1], best_order[k]]) - (block_len - overlap[best_order[k], best_order[k + 1]])
                
                if delta < 0:
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_len += delta
                    improved = True
                    break
    
    # Step 6: Construct final word using optimized ordering
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_ov = min(len(final_word), len(block))
        ov = 0
        for o in range(max_ov, -1, -1):
            if final_word[len(final_word) - o:] == block[:o]:
                ov = o
                break
        final_word += block[ov:]
    
    return "".join(map(str, final_word))
