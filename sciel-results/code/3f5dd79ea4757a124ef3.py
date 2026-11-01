import itertools
import time
import random
import sys

def construct(n):
    if n < 2:
        return "".join(map(str, range(1, n + 1))) if n == 1 else ""

    # For small n, use the classic recursive construction which is optimal
    if n <= 5:
        # Classic recursive construction
        word = [1]
        for m in range(2, n + 1):
            # Get all distinct permutations of 1..m-1 that appear in word
            perms_m, seen_m = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen_m:
                    seen_m.add(window)
                    perms_m.append(window)
            new = []
            for p in perms_m:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n = 6 and n = 7, use a smarter approach
    # The classic recursive construction gives:
    # n=6: 1+2+6+24+120+720 = 873
    # n=7: 873 + 5040 = 5913 (but the known shortest is 5905)
    
    # Strategy: Build the superpermutation for n-1, then insert the new symbol
    # in a way that minimizes the total length.
    
    # For n=7, the known shortest is 5905, which is 7! + 6! + 5! + 4! + 3! + 2! + 1! - 7 = 5905
    # Actually: 5040 + 720 + 120 + 24 + 6 + 2 + 1 = 5913, and 5913 - 8 = 5905
    
    # The standard recursive construction gives 5913 for n=7.
    # We need to find a better arrangement.
    
    if n == 7:
        start_time = time.time()
        time_limit = 140  # Leave 10s margin
        
        # Generate all permutations of 1..6
        perms6 = list(itertools.permutations(range(1, 7)))
        N = len(perms6)  # 720
        perm_to_idx = {p: i for i, p in enumerate(perms6)}
        
        # Build the standard n=6 superpermutation to get an initial ordering
        word6 = [1]
        for m in range(2, 7):
            perms_m, seen_m = [], set()
            for i in range(len(word6) - m + 2):
                window = tuple(word6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen_m:
                    seen_m.add(window)
                    perms_m.append(window)
            new6 = []
            for p in perms_m:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new6), len(block)), -1, -1) if new6[len(new6) - o:] == block[:o])
                new6 += block[overlap:]
            word6 = new6
        
        # Extract the order of 6-permutations from word6
        order = []
        seen_order = set()
        for i in range(len(word6) - 6 + 1):
            window = tuple(word6[i:i + 6])
            if len(set(window)) == 6:
                if window in perm_to_idx:
                    idx = perm_to_idx[window]
                    if idx not in seen_order:
                        seen_order.add(idx)
                        order.append(idx)
        
        # Ensure all permutations are included
        if len(order) != N:
            missing = [i for i in range(N) if i not in seen_order]
            order.extend(missing)
        
        # Precompute overlap matrix M[i][j] = max k such that perms6[i][-k:] == perms6[j][:k]
        M = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms6[i]
            for j in range(N):
                pj = perms6[j]
                for k in range(5, -1, -1):
                    if pi[-k:] == pj[:k]:
                        M[i][j] = k
                        break
        
        # Local search to maximize total overlap
        def calc_total_overlap(ord):
            total = 0
            for i in range(len(ord) - 1):
                total += M[ord[i]][ord[i + 1]]
            return total
        
        current_overlap = calc_total_overlap(order)
        best_order = order[:]
        best_overlap = current_overlap
        
        # Hill Climbing
        improved = True
        while improved and time.time() - start_time < time_limit - 10:
            improved = False
            for i in range(N - 1):
                a = order[i]
                b = order[i + 1]
                prev = order[i - 1] if i > 0 else -1
                nxt = order[i + 2] if i + 2 < N else -1
                
                old_cost = 0
                if prev != -1:
                    old_cost += M[prev][a]
                old_cost += M[a][b]
                if nxt != -1:
                    old_cost += M[b][nxt]
                
                new_cost = 0
                if prev != -1:
                    new_cost += M[prev][b]
                new_cost += M[b][a]
                if nxt != -1:
                    new_cost += M[a][nxt]
                
                if new_cost > old_cost:
                    order[i], order[i + 1] = b, a
                    current_overlap += (new_cost - old_cost)
                    improved = True
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_order = order[:]
        
        # If time remains, do simulated annealing style random swaps
        if time.time() - start_time < time_limit - 5:
            temp = 1.0
            cooling = 0.99999
            while time.time() - start_time < time_limit - 5:
                i = random.randint(0, N - 2)
                a = order[i]
                b = order[i + 1]
                prev = order[i - 1] if i > 0 else -1
                nxt = order[i + 2] if i + 2 < N else -1
                
                old_cost = 0
                if prev != -1:
                    old_cost += M[prev][a]
                old_cost += M[a][b]
                if nxt != -1:
                    old_cost += M[b][nxt]
                
                new_cost = 0
                if prev != -1:
                    new_cost += M[prev][b]
                new_cost += M[b][a]
                if nxt != -1:
                    new_cost += M[a][nxt]
                
                delta = new_cost - old_cost
                if delta >= 0 or random.random() < pow(temp, 1.0 / max(abs(delta), 0.1)):
                    order[i], order[i + 1] = b, a
                    current_overlap += delta
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_order = order[:]
                temp *= cooling
        
        # Construct the word from best_order
        word = []
        prev_idx = -1
        for i, idx in enumerate(best_order):
            p = perms6[idx]
            block = list(p) + [7] + list(p)
            if i == 0:
                word = block
            else:
                overlap = M[prev_idx][idx]
                word.extend(block[overlap:])
            prev_idx = idx
        
        return "".join(map(str, word))
    
    else:
        # For n >= 6, use recursive construction
        word = [1]
        for m in range(2, n + 1):
            perms_m, seen_m = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen_m:
                    seen_m.add(window)
                    perms_m.append(window)
            new = []
            for p in perms_m:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
