import itertools
import time

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n<=4
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
    
    if n == 5:
        # Standard is optimal (153)
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
        
    if n == 6:
        # Standard is optimal (873)
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
        start_time = time.time()
        deadline = start_time + 110 # Safety margin
        
        # Generate permutations of 1..6
        perms = list(itertools.permutations(range(1, 7)))
        # Map perm to index
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        N = len(perms)
        
        # Pre-compute block strings
        # Block is p 7 p
        block_strs = ["".join(map(str, p)) + "7" + "".join(map(str, p)) for p in perms]
        
        # Function to compute overlap
        def get_overlap(i, j):
            s1 = block_strs[i]
            s2 = block_strs[j]
            max_len = min(len(s1), len(s2))
            # We know max overlap is 5, but let's check up to 6 just in case
            # Actually, we can just check 5, 4, 3, 2, 1, 0
            for k in range(5, -1, -1):
                if s1[-k:] == s2[:k]:
                    return k
            return 0
            
        # Nearest Neighbor Heuristic
        # Start with a random perm
        import random
        random.seed(42)
        
        best_order = None
        best_cost = float('inf')
        
        # Try a few starting points
        for start_idx in range(0, N, 100): # Sample starting points
            if time.time() > deadline:
                break
                
            current = start_idx
            visited = [False] * N
            visited[current] = True
            order = [current]
            
            for _ in range(N - 1):
                min_cost = float('inf')
                next_node = -1
                for j in range(N):
                    if not visited[j]:
                        cost = 13 - get_overlap(current, j)
                        if cost < min_cost:
                            min_cost = cost
                            next_node = j
                if next_node == -1:
                    break
                order.append(next_node)
                visited[next_node] = True
                current = next_node
                
            # Calculate total cost
            total_cost = 0
            for i in range(len(order) - 1):
                total_cost += 13 - get_overlap(order[i], order[i+1])
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_order = order[:]
                
            # 2-opt improvement
            if time.time() > deadline:
                break
                
            improved = True
            while improved:
                improved = False
                for i in range(len(best_order) - 1):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(best_order)):
                        # Check if reversing best_order[i+1..j] improves cost
                        # Cost change:
                        # Old: C(A, B) + C(D, E) where A=best[i], B=best[i+1], D=best[j], E=best[j+1]
                        # New: C(A, D) + C(B, E)
                        # Note: if j is last, E is None, so we don't have C(D, E)
                        # This is a path, not a cycle.
                        
                        A = best_order[i]
                        B = best_order[i+1]
                        D = best_order[j]
                        
                        cost_AB = 13 - get_overlap(A, B)
                        
                        if j < len(best_order) - 1:
                            E = best_order[j+1]
                            cost_DE = 13 - get_overlap(D, E)
                            cost_AD = 13 - get_overlap(A, D)
                            cost_BE = 13 - get_overlap(B, E)
                            if cost_AD + cost_BE < cost_AB + cost_DE:
                                # Reverse
                                best_order[i+1:j+1] = reversed(best_order[i+1:j+1])
                                improved = True
                                # Update best_cost roughly or recalculate later
                        else:
                            # D is the last node
                            # Old: C(A, B) ... C(prev_D, D)
                            # This is complicated for path 2-opt.
                            # Skip 2-opt for the last edge for simplicity, or handle carefully.
                            pass
                            
                if time.time() > deadline:
                    break

        # Recalculate final cost and construct word
        if best_order is None:
            # Fallback to standard
            best_order = list(range(N))
            
        # Construct word
        word = []
        for idx in best_order:
            block = block_strs[idx]
            if not word:
                word_str = block
            else:
                # Find overlap with current end
                current_str = "".join(map(str, word)) # This is slow
                # Better to keep a string
                pass
        
        # Re-implement construction efficiently
        current_str = ""
        for idx in best_order:
            block = block_strs[idx]
            if not current_str:
                current_str = block
            else:
                # Find max overlap
                max_k = 0
                # Max possible overlap is 5
                for k in range(5, -1, -1):
                    if current_str[-k:] == block[:k]:
                        max_k = k
                        break
                current_str += block[max_k:]
                
        return current_str

    # Fallback for n > 7 or other
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
