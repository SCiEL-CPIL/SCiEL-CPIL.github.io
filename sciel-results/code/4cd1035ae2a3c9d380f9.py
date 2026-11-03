"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(n):
    if n <= 1:
        return "1"
    
    # For small n, use the classic recursive construction which is optimal
    if n <= 4:
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

    # For n >= 5, use Nearest Neighbor with restarts
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_strs = [''.join(map(str, p)) for p in perms]
    
    # Build prefix map: prefix_string -> list of indices
    # We only need prefixes of length 1 to n-1 for overlaps
    prefix_map = {}
    for i, s in enumerate(perm_strs):
        for k in range(1, n):
            p = s[:k]
            if p not in prefix_map:
                prefix_map[p] = []
            prefix_map[p].append(i)
            
    def run_nn(randomize_ties=False):
        # Start with a random permutation or 0
        start = random.randint(0, N - 1) if randomize_ties else 0
        current = start
        visited = [False] * N
        visited[start] = True
        path = [start]
        
        for _ in range(N - 1):
            s = perm_strs[current]
            found = False
            # Try longest suffix first
            # Suffix of length k corresponds to prefix of length k in next
            for k in range(n - 1, 0, -1):
                suffix = s[n - k:]
                candidates = prefix_map.get(suffix, [])
                if candidates:
                    # Filter unvisited
                    valid = [c for c in candidates if not visited[c]]
                    if valid:
                        if randomize_ties and len(valid) > 1:
                            next_idx = random.choice(valid)
                        else:
                            next_idx = valid[0]
                        path.append(next_idx)
                        visited[next_idx] = True
                        current = next_idx
                        found = True
                        break
            if not found:
                # Fallback: pick any unvisited (should not happen for n>=2)
                for i in range(N):
                    if not visited[i]:
                        path.append(i)
                        visited[i] = True
                        current = i
                        break
                        
        # Construct result string
        result = perm_strs[path[0]]
        for idx in path[1:]:
            s = perm_strs[idx]
            prev = perm_strs[path[idx - 1]]
            # Find overlap
            appended = False
            for k in range(n - 1, 0, -1):
                if prev[n - k:] == s[:k]:
                    result += s[k:]
                    appended = True
                    break
            if not appended:
                result += s
        return result

    best_result = None
    best_len = float('inf')
    
    start_time = time.time()
    limit = 110.0  # Safety margin
    
    # Run multiple restarts
    # First run is deterministic
    res = run_nn(randomize_ties=False)
    if len(res) < best_len:
        best_len = len(res)
        best_result = res
        
    # Randomized restarts
    while time.time() - start_time < limit:
        res = run_nn(randomize_ties=True)
        if len(res) < best_len:
            best_len = len(res)
            best_result = res
            # Optional: break if we hit the theoretical lower bound or known optimal
            if best_len <= 5905: 
                break

    return best_result
# EVOLVE-BLOCK-END
