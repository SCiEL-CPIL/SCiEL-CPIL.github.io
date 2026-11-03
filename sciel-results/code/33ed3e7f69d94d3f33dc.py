import time
import random
from itertools import permutations

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _get_classic_order(n):
    """
    Returns the order of (n-1)-permutations as they appear in the classic superpermutation of n.
    This is equivalent to the order of (n-2)-permutations in the classic superpermutation of n-1,
    expanded? No.
    
    Let's just generate the classic superpermutation S_{n-1} and extract the order of (n-1)-perms.
    Wait, S_n is built from S_{n-1}.
    The blocks in S_n correspond to the (n-1)-perms in S_{n-1}.
    So the order of (n-1)-perms in S_n is the order of (n-1)-perms in S_{n-1}.
    
    So for n=7, we need the order of 6-perms in S_6.
    We can generate S_6 and extract the order of 6-perms from it.
    """
    if n == 1:
        return [(1,)]
    
    # Generate S_{n-1}
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # Now 'word' is S_{n-1}.
    # Extract order of (n-1)-perms
    perms, seen = [], set()
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    return perms

def construct(n):
    deadline = time.time() + 140  # 140s safety margin
    
    if n < 7:
        # Classic construction is optimal for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # Get classic order of 6-perms
        classic_order = _get_classic_order(7)
        N = len(classic_order) # 720
        
        # Map perms to indices for fast lookup
        perm_to_idx = {p: i for i, p in enumerate(classic_order)}
        
        # Precompute overlap matrix
        # overlaps[i][j] = max k < 6 such that classic_order[i][-k:] == classic_order[j][:k]
        overlaps = [[0] * N for _ in range(N)]
        for i, p1 in enumerate(classic_order):
            for j, p2 in enumerate(classic_order):
                if i == j:
                    overlaps[i][j] = 0 # As analyzed, self-overlap in this context is 0 for distinct blocks
                    # Actually, if p1 == p2, the block is p1 7 p1.
                    # Next block p1 7 p1.
                    # End p1, Start p1.
                    # Suffix 5 of p1 vs Prefix 5 of p1.
                    # p1 is a permutation of 1..6.
                    # e.g., 123456. Suff5: 23456. Pref5: 12345. No match.
                    # So 0 is correct.
                else:
                    # Check k=5 down to 1
                    if p1[-5:] == p2[:5]:
                        overlaps[i][j] = 5
                    elif p1[-4:] == p2[:4]:
                        overlaps[i][j] = 4
                    elif p1[-3:] == p2[:3]:
                        overlaps[i][j] = 3
                    elif p1[-2:] == p2[:2]:
                        overlaps[i][j] = 2
                    elif p1[-1] == p2[0]:
                        overlaps[i][j] = 1
                    else:
                        overlaps[i][j] = 0
        
        # Local Search to maximize sum of overlaps
        # Current order
        current_order = list(range(N))
        
        def calc_score(order):
            s = 0
            for k in range(N - 1):
                s += overlaps[order[k]][order[k+1]]
            return s
        
        current_score = calc_score(current_order)
        best_score = current_score
        best_order = current_order[:]
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Randomize start position or just scan all
            # To avoid getting stuck in local minima, we can do multiple passes
            # or random restarts.
            
            # Pass 1: Forward scan
            for i in range(N - 1):
                if time.time() > deadline - 5:
                    break
                u = current_order[i]
                v = current_order[i+1]
                
                # Old local score contribution:
                # (prev, u) + (u, v) + (v, next)
                # Note: if i=0, no prev. if i=N-2, no next.
                
                prev_idx = current_order[i-1] if i > 0 else -1
                next_idx = current_order[i+2] if i+2 < N else -1
                
                old_local = 0
                new_local = 0
                
                if prev_idx != -1:
                    old_local += overlaps[prev_idx][u]
                    new_local += overlaps[prev_idx][v]
                
                old_local += overlaps[u][v]
                new_local += overlaps[v][u]
                
                if next_idx != -1:
                    old_local += overlaps[v][next_idx]
                    new_local += overlaps[u][next_idx]
                
                if new_local > old_local:
                    current_order[i], current_order[i+1] = v, u
                    current_score += new_local - old_local
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
            
            # Pass 2: Backward scan (optional, but helps)
            if time.time() < deadline - 5:
                for i in range(N - 2, -1, -1):
                    u = current_order[i]
                    v = current_order[i+1]
                    
                    prev_idx = current_order[i-1] if i > 0 else -1
                    next_idx = current_order[i+2] if i+2 < N else -1
                    
                    old_local = 0
                    new_local = 0
                    
                    if prev_idx != -1:
                        old_local += overlaps[prev_idx][u]
                        new_local += overlaps[prev_idx][v]
                    
                    old_local += overlaps[u][v]
                    new_local += overlaps[v][u]
                    
                    if next_idx != -1:
                        old_local += overlaps[v][next_idx]
                        new_local += overlaps[u][next_idx]
                    
                    if new_local > old_local:
                        current_order[i], current_order[i+1] = v, u
                        current_score += new_local - old_local
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_order = current_order[:]
            
            # If no improvement in full passes, try random perturbation
            if not improved:
                # Perturb: swap 10 random pairs
                for _ in range(10):
                    i = random.randint(0, N - 2)
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                # Recalculate score
                current_score = calc_score(current_order)
                improved = True # Force another pass

    # Reconstruct word from best_order
    # Use the best_order found
    if n == 7:
        order = best_order
    else:
        # Fallback
        order = list(range(_factorial(n-1)))
        
    # Reconstruct
    word = []
    for idx in order:
        p = classic_order[idx]
        block = list(p) + [7] + list(p)
        # Find overlap with current word end
        overlap = 0
        max_ov = min(len(word), len(block))
        for o in range(max_ov, -1, -1):
            if o == 0:
                overlap = 0
                break
            if word[len(word) - o:] == block[:o]:
                overlap = o
                break
        word += block[overlap:]
        
    # Final validation check (optional but safe)
    # For n=7, checking all 5040 perms is fast.
    # But we trust the construction.
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
