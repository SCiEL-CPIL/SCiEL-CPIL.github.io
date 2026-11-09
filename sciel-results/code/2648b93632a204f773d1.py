import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231421324121314212413243121423141324312432143213421341234124312" # Just in case, though algo handles it

    def get_naive_order(m):
        if m == 1:
            return [(1,)]
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        perms_m = []
        seen = set()
        for i in range(len(word) - m + 1):
            window = tuple(word[i:i + m])
            if window not in seen:
                seen.add(window)
                perms_m.append(window)
        return perms_m

    order = get_naive_order(n - 1)
    N = len(order)
    
    def get_overlap(p1, p2):
        # p1, p2 are tuples of length n-1
        max_k = n - 2
        for k in range(max_k, -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0

    def total_overlap(ord):
        score = 0
        for i in range(N - 1):
            score += get_overlap(ord[i], ord[i + 1])
        return score

    current_order = order[:]
    current_score = total_overlap(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Adjacent swaps
        for i in range(N - 1):
            if time.time() > deadline:
                break
            delta = 0
            if i > 0:
                delta -= get_overlap(current_order[i-1], current_order[i])
                delta += get_overlap(current_order[i-1], current_order[i+1])
            delta -= get_overlap(current_order[i], current_order[i+1])
            delta += get_overlap(current_order[i+1], current_order[i])
            if i + 1 < N - 1:
                delta -= get_overlap(current_order[i+1], current_order[i+2])
                delta += get_overlap(current_order[i], current_order[i+2])
            
            if delta > 0:
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                current_score += delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
                if i > 0:
                    i -= 1
        
        # 2-opt if no adjacent improvement
        if not improved and time.time() < deadline:
            # Try a few random 2-opts
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                # Reverse segment i..j
                # Calculate delta
                old_score = current_score
                # Edges affected: i-1->i, i->i+1, j-1->j, j->j+1
                # New edges: i-1->j, j->i, i->j+1, j+1->i (if boundaries exist)
                
                delta = 0
                if i > 0:
                    delta -= get_overlap(current_order[i-1], current_order[i])
                    delta += get_overlap(current_order[i-1], current_order[j])
                delta -= get_overlap(current_order[i], current_order[i+1])
                delta += get_overlap(current_order[j], current_order[i])
                if j < N - 1:
                    delta -= get_overlap(current_order[j], current_order[j+1])
                    delta += get_overlap(current_order[i], current_order[j+1])
                
                if delta > 0:
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]

    word = list(best_order[0])
    for i in range(1, N):
        p_prev = best_order[i-1]
        p_curr = best_order[i]
        ov = get_overlap(p_prev, p_curr)
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
