import itertools
import time
import random
import sys

def construct(n):
    """Construct a superpermutation of order n.
    For n=7, tries to find a short valid superpermutation within time budget.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134121432134214321324132412431234131421412313241241312431431213141241231234"

    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    
    # For n=7: total_perms = 5040
    # For n=6: total_perms = 720
    # For n=5: total_perms = 120
    
    # Precompute all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    
    # Build the permutation graph:
    # Node: permutation (tuple of n elements)
    # Edge from P to Q if Q = (P[1], P[2], ..., P[n-1], x) for some x not in P[1:]
    # i.e., Q[i] = P[i+1] for i=0..n-2, and Q[n-1] is the new symbol
    
    # Precompute neighbors
    neighbors = {}
    for perm in all_perms:
        prefix = perm[1:]  # tuple of n-1 elements
        used = set(prefix)
        available = [x for x in range(1, n + 1) if x not in used]
        nbrs = []
        for x in available:
            next_perm = prefix + (x,)
            nbrs.append(next_perm)
        neighbors[perm] = nbrs
    
    # Also precompute reverse neighbors for potential backtracking
    rev_neighbors = {p: [] for p in all_perms}
    for p in all_perms:
        for q in neighbors[p]:
            rev_neighbors[q].append(p)
    
    start_perm = tuple(range(1, n + 1))
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def build_word_from_path(path):
        """Given a list of permutations (the path), build the word."""
        word = list(path[0])
        for i in range(1, len(path)):
            word.append(path[i][n - 1])
        return word
    
    def validate_word(word):
        """Check if word contains all n! permutations as windows."""
        seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:  # valid permutation window
                seen.add(window)
        return len(seen) == total_perms
    
    # Strategy: Multiple randomized greedy attempts with Warnsdorff's rule
    # and some perturbation, keeping the best valid result.
    
    best_word = None
    best_len = float('inf')
    
    # First, try the standard recursive construction as a fallback
    def standard_construction(n):
        """Classic recursive superpermutation: S(n) = for each perm of 1..n-1: S(perm) + n"""
        if n == 1:
            return [1]
        prev_perms = list(itertools.permutations(range(1, n)))
        result = []
        for p in prev_perms:
            result.append(p[0])
            # Add n, then the rest of the previous superpermutation for p[1:]
            # Actually, the standard construction:
            # S_n(p_1...p_{n-1}) = p_1 S_{n-1}(p_2...p_{n-1}) p_1 S_{n-1}(p_3...p_{n-1}, p_2) ...
            # This is complex; let's use a simpler block-based approach
        # Simpler: for each permutation of 1..n, append it with max overlap
        perms = list(itertools.permutations(range(1, n + 1)))
        word = []
        for i, p in enumerate(perms):
            if i == 0:
                word = list(p)
            else:
                # Find maximum overlap between end of word and start of p
                max_ov = 0
                for ov in range(min(len(word), n), 0, -1):
                    if word[-ov:] == list(p[:ov]):
                        max_ov = ov
                        break
                word.extend(list(p[max_ov:]))
        return word
    
    std_word = standard_construction(n)
    if validate_word(std_word):
        best_word = std_word
        best_len = len(std_word)
    
    # Now try greedy approaches with randomization
    random.seed(42)
    
    for attempt in range(50):
        if time_remaining() < 5.0:
            break
        
        # Variant: different tie-breaking and randomization
        visited = set()
        path = [start_perm]
        visited.add(start_perm)
        current = start_perm
        
        success = True
        for step in range(total_perms - 1):
            if time_remaining() < 3.0:
                success = False
                break
            
            nbrs = neighbors[current]
            unvisited_nbrs = [nb for nb in nbrs if nb not in visited]
            
            if not unvisited_nbrs:
                success = False
                break
            
            # Warnsdorff with randomization
            if attempt < 10:
                # Pure Warnsdorff
                def count_unvisited(nb):
                    return sum(1 for x in neighbors[nb] if x not in visited)
                unvisited_nbrs.sort(key=count_unvisited)
                current = unvisited_nbrs[0]
            elif attempt < 20:
                # Warnsdorff with slight randomization
                def count_unvisited(nb):
                    return sum(1 for x in neighbors[nb] if x not in visited)
                unvisited_nbrs.sort(key=count_unvisited)
                # Pick from top 2-3 with probability
                k = min(3, len(unvisited_nbrs))
                if k > 1:
                    current = unvisited_nbrs[random.randrange(k)]
                else:
                    current = unvisited_nbrs[0]
            elif attempt < 30:
                # Random neighbor
                current = random.choice(unvisited_nbrs)
            else:
                # Hybrid: prefer nodes with fewer unvisited neighbors but with randomness
                def score(nb):
                    cnt = sum(1 for x in neighbors[nb] if x not in visited)
                    return cnt + random.random() * 2
                unvisited_nbrs.sort(key=score)
                current = unvisited_nbrs[0]
            
            visited.add(current)
            path.append(current)
        
        if success and len(path) == total_perms:
            word = build_word_from_path(path)
            # The path covers all permutations, so the word is valid
            if len(word) < best_len:
                best_len = len(word)
                best_word = word
    
    # Try additional local improvement: look for shorter valid superpermutations
    # using a different construction strategy
    
    # Strategy 2: Build word character by character with lookahead
    # State: last (n-1) characters, set of seen permutations
    # This is too slow for n=7 with full state space, so we use it for n<=6
    
    if n <= 6:
        # For smaller n, try BFS/DFS with pruning
        if n <= 5:
            # Full search for n<=5
            from collections import deque
            
            target = total_perms
            # State: (tuple of last n-1 chars, frozenset of seen perms)
            # This state space is too large even for n=5 (120 perms, 5^4=625 suffixes)
            # Use greedy with backtracking for n=6
            
            # For n=6, try iterative improvement
            for iter_count in range(20):
                if time_remaining() < 5.0:
                    break
                
                # Randomized greedy
                random.seed(1000 + iter_count)
                visited = set()
                path = [start_perm]
                visited.add(start_perm)
                current = start_perm
                
                ok = True
                for step in range(total_perms - 1):
                    nbrs = neighbors[current]
                    unvisited_nbrs = [nb for nb in nbrs if nb not in visited]
                    
                    if not unvisited_nbrs:
                        ok = False
                        break
                    
                    # Warnsdorff
                    unvisited_nbrs.sort(key=lambda nb: sum(1 for x in neighbors[nb] if x not in visited))
                    # Add randomness
                    if random.random() < 0.1 and len(unvisited_nbrs) > 1:
                        current = random.choice(unvisited_nbrs)
                    else:
                        current = unvisited_nbrs[0]
                    
                    visited.add(current)
                    path.append(current)
                
                if ok and len(path) == total_perms:
                    word = build_word_from_path(path)
                    if len(word) < best_len:
                        best_len = len(word)
                        best_word = word
    
    # Final strategy: Try to improve by finding shorter paths
    # Using a different starting point or different heuristics
    if time_remaining() > 10.0 and n <= 7:
        # Try different starting permutations
        for start_idx in range(min(20, total_perms)):
            if time_remaining() < 5.0:
                break
            
            start_p = all_perms[start_idx]
            visited = set()
            path = [start_p]
            visited.add(start_p)
            current = start_p
            
            ok = True
            for step in range(total_perms - 1):
                nbrs = neighbors[current]
                unvisited_nbrs = [nb for nb in nbrs if nb not in visited]
                
                if not unvisited_nbrs:
                    ok = False
                    break
                
                # Warnsdorff
                unvisited_nbrs.sort(key=lambda nb: sum(1 for x in neighbors[nb] if x not in visited))
                current = unvisited_nbrs[0]
                
                visited.add(current)
                path.append(current)
            
            if ok and len(path) == total_perms:
                word = build_word_from_path(path)
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = word
    
    if best_word is None:
        # Shouldn't happen, but fallback
        best_word = standard_construction(n)
    
    return "".join(map(str, best_word))

# Test
if __name__ == "__main__":
    for n in range(1, 8):
        t0 = time.time()
        result = construct(n)
        t1 = time.time()
        print(f"n={n}: length={len(result)}, time={t1-t0:.2f}s")
        # Validate
        total = 1
        for i in range(2, n+1):
            total *= i
        seen = set()
        for i in range(len(result) - n + 1):
            window = tuple(int(c) for c in result[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        print(f"  permutations found: {len(seen)}/{total}")
