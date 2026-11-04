import itertools
import sys

def construct(n):
    if n < 7:
        # Standard recursive construction is optimal for n < 7
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
        # Generate S6
        word6 = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word6) - m + 2):
                window = tuple(word6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word6 = new
        
        # Extract permutations of length 6 from S6
        perms6 = []
        seen = set()
        for i in range(len(word6) - 5):
            window = tuple(word6[i:i + 6])
            if window not in seen:
                seen.add(window)
                perms6.append(window)
        
        # P_start is the last permutation in S6
        start_idx = perms6.index(tuple(word6[-6:]))
        
        # Precompute overlaps
        N = len(perms6)
        # overlap[i][j] = length of longest suffix of perms6[i] that is prefix of perms6[j]
        overlaps = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms6[i]
            for j in range(N):
                pj = perms6[j]
                # Check k from 6 down to 0
                for k in range(6, -1, -1):
                    if k == 0:
                        overlaps[i][j] = 0
                        break
                    if pi[6-k:] == pj[:k]:
                        overlaps[i][j] = k
                        break
        
        # Cost matrix
        # Cost to go from i to j is 12 - overlaps[i][j]
        # We want to minimize sum of costs.
        
        # Greedy Nearest Neighbor
        visited = [False] * N
        path = []
        current = start_idx
        visited[current] = True
        path.append(current)
        
        for _ in range(N - 1):
            best_next = -1
            best_cost = float('inf')
            for j in range(N):
                if not visited[j]:
                    cost = 12 - overlaps[current][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = j
            path.append(best_next)
            visited[best_next] = True
            current = best_next
            
        # 2-opt improvement
        # We have a path of indices.
        # Cost of path = sum(cost(path[i], path[i+1]))
        # Note: The first node is start_idx. The cost starts from start_idx to path[1].
        # Actually, the cost is associated with the transition.
        # Total length = len(word6) + sum_{i=0}^{N-2} (12 - overlaps[path[i]][path[i+1]])
        
        def calc_cost(path):
            total = 0
            for i in range(len(path) - 1):
                total += 12 - overlaps[path[i]][path[i+1]]
            return total
        
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 1):
                for j in range(i + 1, len(path)):
                    # Reverse path[i..j]
                    old_cost = calc_cost(path)
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_cost = calc_cost(new_path)
                    if new_cost < old_cost:
                        path = new_path
                        improved = True
                        # Break and restart? Or continue?
                        # Breaking and restarting is safer for 2-opt
                        break
                if improved:
                    break
        
        # Construct the word
        word7 = list(word6)
        for idx in path[1:]: # path[0] is start_idx, which is already in word6
            p = perms6[idx]
            block = list(p) + [7] + list(p)
            # Find overlap with current word7
            max_k = min(len(word7), len(block))
            overlap = 0
            for k in range(max_k, -1, -1):
                if word7[len(word7)-k:] == block[:k]:
                    overlap = k
                    break
            word7 += block[overlap:]
            
        return "".join(map(str, word7))
        
    else:
        # Fallback for n > 7, though task implies n=7
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
