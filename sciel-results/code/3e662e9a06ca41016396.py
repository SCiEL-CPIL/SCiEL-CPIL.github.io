import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Base cases
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
        
    # Generate all permutations of 1..n-1
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    M = len(perms)
    
    # Precompute overlap matrix using KMP prefix function
    # overlap[i][j] = longest k such that suffix of perms[i] of length k == prefix of perms[j] of length k
    # For each pair, compute prefix function on (perms[i] + sentinel + perms[j])
    # The last value of the prefix function gives the longest suffix of perms[i] that is a prefix of perms[j]
    
    # Build a list of strings for faster comparison
    perm_strs = [''.join(map(str, p)) for p in perms]
    
    overlap = [[0] * M for _ in range(M)]
    for i in range(M):
        s_i = perm_strs[i]
        for j in range(M):
            s_j = perm_strs[j]
            # Build combined string: s_i + sentinel + s_j
            # Sentinel is a character not in '1'..'n'
            combined = s_i + chr(0) + s_j
            # Compute prefix function
            m = len(combined)
            pi = [0] * m
            for idx in range(1, m):
                k = pi[idx - 1]
                while k > 0 and combined[idx] != combined[k]:
                    k = pi[k - 1]
                if combined[idx] == combined[k]:
                    k += 1
                pi[idx] = k
            overlap[i][j] = pi[m - 1]
    
    # Initial order: lexicographic
    current_order = list(range(M))
    
    def calc_score(order):
        score = 0
        for i in range(len(order) - 1):
            u = order[i]
            v = order[i+1]
            score += overlap[u][v]
        return score
    
    current_score = calc_score(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    # Local Search: Full 2-Opt (segment reversal)
    # For reversing segment [i, j] in current_order:
    # Edges removed: (i-1, i) if i>0, (j, j+1) if j<M-1, and internal (k, k+1) for k in i..j-1
    # Edges added: (i-1, j) if i>0, (i, j+1) if j<M-1, and internal (k+1, k) for k in i..j-1
    # Delta = sum of new edge weights - sum of old edge weights
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(M - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, M):
                # Calculate delta for reversing current_order[i..j]
                delta = 0
                
                # Internal edges: for k in i..j-1, remove (k, k+1), add (k+1, k)
                for k in range(i, j):
                    delta -= overlap[current_order[k]][current_order[k+1]]
                    delta += overlap[current_order[k+1]][current_order[k]]
                
                # Boundary: left
                if i > 0:
                    delta -= overlap[current_order[i-1]][current_order[i]]
                    delta += overlap[current_order[i-1]][current_order[j]]
                else:
                    # No left boundary edge removed, no left boundary edge added
                    pass
                
                # Boundary: right
                if j < M - 1:
                    delta -= overlap[current_order[j]][current_order[j+1]]
                    delta += overlap[current_order[i]][current_order[j+1]]
                else:
                    pass
                
                if delta > 0:
                    # Reverse segment
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                    improved = True
    
    # Use best_order to construct the word
    def build_string(order):
        if not order:
            return []
        # First block
        word = list(perms[order[0]]) + [n] + list(perms[order[0]])
        for idx in range(1, len(order)):
            p_curr = perms[order[idx]]
            block_curr = list(p_curr) + [n] + list(p_curr)
            
            ov = overlap[order[idx-1]][order[idx]]
            word += block_curr[ov:]
        return word

    final_word = build_string(best_order)
    
    # Verification
    all_perms = set(itertools.permutations(range(1, n+1)))
    found_perms = set()
    for i in range(len(final_word) - n + 1):
        window = tuple(final_word[i:i+n])
        found_perms.add(window)
        
    if all_perms == found_perms:
        return "".join(map(str, final_word))
    else:
        # Fallback
        def get_classic_superperm(k):
            if k == 1:
                return [1]
            word = [1]
            for m in range(2, k + 1):
                perms, seen = [], set()
                for i in range(len(word) - m + 2):
                    window = tuple(word[i:i + m - 1])
                    if len(set(window)) == m - 1 and window not in seen:
                        seen.add(window)
                        perms.append(window)
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    overlap = 0
                    max_overlap = min(len(new), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                word = new
            return word

        classic_word = get_classic_superperm(n)
        return "".join(map(str, classic_word))
