"""Improved superpermutation construction with local search."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    def classic_word(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    if n <= 5:
        return "".join(map(str, classic_word(n)))
    
    base_word = classic_word(n - 1)
    base_len = len(base_word)
    
    perms, seen = [], set()
    for i in range(base_len - (n - 1) + 1):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    num_perms = len(perms)
    m = n - 1  # length of each perm
    
    # Precompute overlap from base to each perm
    overlap_from_base = [0] * num_perms
    for j in range(num_perms):
        p = perms[j]
        for k in range(min(len(base_word), m), 0, -1):
            if tuple(base_word[-k:]) == p[:k]:
                overlap_from_base[j] = k
                break
    
    # Precompute overlap matrix
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            if i == j:
                continue
            pj = perms[j]
            for k in range(m, 0, -1):
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break
    
    # Added length for an ordering
    def added_length(order):
        total = n - overlap_from_base[order[0]]
        for i in range(1, len(order)):
            total += n - overlap_matrix[order[i-1]][order[i]]
        return total
    
    def total_len(order):
        return base_len + added_length(order)
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Classic order
    best_order = list(range(num_perms))
    best_len = total_len(best_order)
    
    # Greedy
    def greedy():
        visited = [False] * num_perms
        order = []
        best_j = 0
        for j in range(1, num_perms):
            if overlap_from_base[j] > overlap_from_base[best_j]:
                best_j = j
        visited[best_j] = True
        order.append(best_j)
        cur = best_j
        for _ in range(num_perms - 1):
            best_j = -1
            best_ov = -1
            for j in range(num_perms):
                if not visited[j] and overlap_matrix[cur][j] > best_ov:
                    best_ov = overlap_matrix[cur][j]
                    best_j = j
            if best_j == -1:
                for j in range(num_perms):
                    if not visited[j]:
                        best_j = j
                        break
            visited[best_j] = True
            order.append(best_j)
            cur = best_j
        return order
    
    if time.time() < deadline:
        g = greedy()
        gl = total_len(g)
        if gl < best_len:
            best_len = gl
            best_order = g[:]
    
    # Local search with efficient delta computation
    def delta_swap(order, i):
        """Compute change in added_length if we swap positions i and i+1."""
        # Positions affected: i-1->i, i->i+1, i+1->i+2
        # Also if i==0, position 0's overlap_from_base changes
        old = 0
        new = 0
        
        if i == 0:
            old += n - overlap_from_base[order[0]]
            new += n - overlap_from_base[order[1]]
        else:
            old += n - overlap_matrix[order[i-1]][order[i]]
            new += n - overlap_matrix[order[i-1]][order[i+1]]
        
        # Middle: i -> i+1 becomes i+1 -> i
        old += n - overlap_matrix[order[i]][order[i+1]]
        new += n - overlap_matrix[order[i+1]][order[i]]
        
        if i + 1 < num_perms - 1:
            old += n - overlap_matrix[order[i+1]][order[i+2]]
            new += n - overlap_matrix[order[i]][order[i+2]]
        
        return new - old
    
    def local_search(order, time_budget):
        order = order[:]
        cur_len = total_len(order)
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in range(num_perms - 1):
                if time.time() >= end_time:
                    break
                d = delta_swap(order, i)
                if d < 0:
                    order[i], order[i+1] = order[i+1], order[i]
                    cur_len += d
                    improved = True
            # 2-opt: try reversing segments
            attempts = 0
            while attempts < 200 and time.time() < end_time:
                attempts += 1
                i = random.randint(0, num_perms - 3)
                j = random.randint(i + 2, num_perms - 1)
                # Reverse order[i:j+1]
                order[i:j+1] = order[i:j+1][::-1]
                new_len = total_len(order)
                if new_len < cur_len:
                    cur_len = new_len
                    improved = True
                else:
                    order[i:j+1] = order[i:j+1][::-1]
        return order, cur_len
    
    # Run local search multiple times
    for iteration in range(20):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        new_order, new_len = local_search(best_order, remaining)
        if new_len < best_len:
            best_len = new_len
            best_order = new_order
        # Perturb if no improvement
        if new_len == best_len:
            # Random restart from greedy with perturbation
            if time.time() < deadline - 10:
                g = greedy()
                # Perturb
                for _ in range(50):
                    if time.time() >= deadline - 10:
                        break
                    i = random.randint(0, num_perms - 2)
                    g[i], g[i+1] = g[i+1], g[i]
                gl = total_len(g)
                if gl < best_len:
                    best_len = gl
                    best_order = g[:]
    
    # Build final word
    word = list(base_word)
    for idx in best_order:
        p = perms[idx]
        block = [n] + list(p)
        best_overlap = 0
        max_k = min(len(word), len(block))
        for k in range(max_k, 0, -1):
            if word[-k:] == block[:k]:
                best_overlap = k
                break
        word += block[best_overlap:]
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
