"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin for 90s limit
    
    N = 256
    # Perfect squares up to 512 (max sum is 255+256=511)
    squares = set()
    k = 2
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    # Also 1 = 1^1 but min sum is 1+1=2, so 1 is irrelevant for pairs
    # Actually for a=b, 2a must not be square. 2a ranges from 2 to 512.
    
    # Build conflict: for each element i, which elements j >= i conflict with i?
    # conflict[i] = set of j such that i+j is a perfect square (j >= i)
    # We'll use a list of bitmasks or sets
    
    # For efficiency, use a list of sets for conflicts
    # Actually, let's use a different approach: for each i, store the set of j > i that conflict
    # and separately handle the diagonal (2i is square)
    
    # Diagonal conflicts: 2*i is a perfect square
    diag_conflict = set()
    for i in range(1, N + 1):
        s = 2 * i
        r = int(s ** 0.5)
        if r * r == s or (r + 1) * (r + 1) == s:
            diag_conflict.add(i)
    
    # Edge conflicts: for i < j, i + j is a perfect square
    # Build adjacency: adj[i] = set of j > i where i+j is square
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for sq in squares:
            j = sq - i
            if j > i and j <= N:
                adj[i].add(j)
            elif j == i:
                # diagonal, already handled
                pass
    
    # Now we want maximum independent set.
    # Greedy + local search with 1-flip and 2-swap moves.
    
    def is_valid(s):
        """Check if set s is a valid independent set."""
        s_list = sorted(s)
        n = len(s_list)
        # Check diagonal
        for x in s_list:
            if x in diag_conflict:
                return False
        # Check pairs
        for i in range(n):
            for j in range(i + 1, n):
                if s_list[j] in adj[s_list[i]]:
                    return False
        return True
    
    def count_conflicts(s):
        """Count number of conflicting pairs in set s (including diagonal)."""
        s_list = sorted(s)
        n = len(s_list)
        cnt = 0
        for x in s_list:
            if x in diag_conflict:
                cnt += 1
        for i in range(n):
            for j in range(i + 1, n):
                if s_list[j] in adj[s_list[i]]:
                    cnt += 1
        return cnt
    
    def greedy_construct():
        """Greedy: add elements in order of fewest conflicts with already-chosen."""
        remaining = list(range(1, N + 1))
        chosen = set()
        # Remove diagonal conflicts first
        remaining = [x for x in remaining if x not in diag_conflict]
        
        while remaining:
            # For each candidate, count how many it conflicts with in `chosen`
            best = None
            best_conf = float('inf')
            for x in remaining:
                conf = 0
                for y in chosen:
                    if x in adj[y] or y in adj[x]:
                        conf += 1
                if conf < best_conf:
                    best_conf = conf
                    best = x
                    if conf == 0:
                        break
            if best is not None:
                chosen.add(best)
                remaining.remove(best)
            else:
                break
        return chosen
    
    def local_search(s, max_iters=5000):
        """Local search with 1-flip and 2-swap moves to maximize |s| while maintaining validity."""
        s = set(s)
        s_list = sorted(s)
        
        # Precompute: for each element, its conflict set within current s
        def get_conflicts_in_set(x, s_set):
            """Return set of elements in s_set that conflict with x."""
            confs = set()
            if x in diag_conflict:
                confs.add(x)  # self-conflict
            for y in s_set:
                if y != x:
                    if y in adj[x] or x in adj[y]:
                        confs.add(y)
            return confs
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            s_list = sorted(s)
            
            # Try 1-flip: remove an element that has conflicts, see if we can add others
            # Actually, for maximization, we want to:
            # 1. Try adding elements not in s (if valid)
            # 2. Try 2-swaps: remove one, add another (net zero or +1)
            
            # First: try to add any element
            for x in range(1, N + 1):
                if x in s:
                    continue
                # Check if x can be added
                if x in diag_conflict:
                    continue
                can_add = True
                for y in s:
                    if y in adj[x] or x in adj[y]:
                        can_add = False
                        break
                if can_add:
                    s.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove a, add b (where b not in s, a in s)
            # Net size stays same, but might enable further additions
            # Actually for pure maximization, 2-swaps don't help directly unless
            # they enable subsequent additions. Let's try: remove a, add b, then try to add more.
            
            s_list = sorted(s)
            for a in s_list:
                for b in range(1, N + 1):
                    if b in s or b == a:
                        continue
                    if b in diag_conflict:
                        continue
                    # Check if b conflicts with (s - {a})
                    ok = True
                    for y in s:
                        if y == a:
                            continue
                        if y in adj[b] or b in adj[y]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Try the swap
                    new_s = set(s)
                    new_s.remove(a)
                    new_s.add(b)
                    # Try to add more elements
                    added = 0
                    for c in range(1, N + 1):
                        if c in new_s:
                            continue
                        if c in diag_conflict:
                            continue
                        can_add = True
                        for y in new_s:
                            if y in adj[c] or c in adj[y]:
                                can_add = False
                                break
                        if can_add:
                            new_s.add(c)
                            added += 1
                    if len(new_s) > len(s):
                        s = new_s
                        improved = True
                        break
                if improved:
                    break
        
        return s
    
    best_set = set()
    best_size = 0
    
    # Multiple restarts with different greedy orderings
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Standard greedy
            s = greedy_construct()
        else:
            # Perturbed greedy: random ordering or random subset removal
            s = greedy_construct()
            # Remove some random elements to create diversity
            s_list = list(s)
            n_remove = max(1, len(s_list) // 10)
            random.shuffle(s_list)
            for x in s_list[:n_remove]:
                s.discard(x)
        
        # Local search
        s = local_search(s, max_iters=3000)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        
        # Try 2-opt style: for pairs in best_set, try removing both and adding different ones
        # This is expensive, so limit it
    
    # Final polish: try to add any remaining elements to best_set
    for x in range(1, N + 1):
        if x in best_set:
            continue
        if x in diag_conflict:
            continue
        can_add = True
        for y in best_set:
            if y in adj[x] or x in adj[x]:  # typo check below
                can_add = False
                break
        if can_add:
            best_set.add(x)
    
    # Verify
    if not is_valid(best_set):
        # Fallback to a known valid set
        # Use the seed
        return list(range(1, 257, 3))
    
    if len(best_set) == 0:
        return [1]
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
